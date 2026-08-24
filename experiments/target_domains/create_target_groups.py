import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from pydantic import BaseModel
from typing import Union
from transformers import AutoTokenizer

from src.utils import load_env, get_logger, load_json, set_seed, save_to_json
from src.db_config import CoreferenceMention
from src.experiment_config import ExperimentConfig, LLMExperimentConfig
from src.data_loading import DatasetLoader
from src.llm_ann.ollama_utils import Annotator

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

class TargetGroupCreation(BaseModel):
    groups: Union[str, list[str]]


import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
import re



class ContextualWordClustering:
    def __init__(self, model_name="roberta-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device)

        self.model.eval()

    def get_word_embedding(self, sentence, target_word):
        # --- 1. find character spans of target word ---
        # case-insensitive, word-boundary aware
        pattern = re.compile(rf"\b{re.escape(target_word)}\b", re.IGNORECASE)
        matches = [(m.start(), m.end()) for m in pattern.finditer(sentence)]

        if not matches:
            # raise ValueError(f"Word '{target_word}' not found in: {sentence}")
            print(f"Word '{target_word}' not found in: {sentence}")
            return None

        # --- 2. tokenize with offsets ---
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True
        )

        offsets = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden = torch.stack(outputs.hidden_states[-4:]).mean(0)[0]

        # --- 3. collect token vectors overlapping the span ---
        vecs = []

        for (tok_start, tok_end), vec in zip(offsets, hidden):
            if tok_start == tok_end:
                continue  # skip special tokens

            for span_start, span_end in matches:
                # overlap condition
                if not (tok_end <= span_start or tok_start >= span_end):
                    vecs.append(vec)
                    break

        if not vecs:
            raise ValueError(
                f"No token overlap found for '{target_word}' in: {sentence}"
            )

        # --- 4. aggregate ---
        return torch.stack(vecs).mean(0).cpu().numpy()

    def build_dataset(self, data):
        embeddings = []
        emb_keys = []
        for k, info in tqdm(data.items()):
            sent = info["sentence"]
            word = info["target_word"]
            vec = self.get_word_embedding(sent, word)
            if vec is not None:
                embeddings.append(vec)
                emb_keys.append(k)

        return np.array(embeddings), emb_keys

    def preprocess(self, X, remove_pc=10):
        # L2 normalize
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # mean center
        X = X - X.mean(axis=0)

        # remove top PCs
        if remove_pc > 0:
            pca = PCA(n_components=remove_pc)
            X = X - pca.inverse_transform(pca.fit_transform(X))

        return X

    def cluster(self, X, min_k, max_k):

        scores = []
        k_vals = range(min_k, max_k + 1)
        for k in tqdm(k_vals, desc="testing k values"):
            kmeans = KMeans(
                n_clusters=k,
                n_init=50,
                random_state=42
            )
            kmeans.fit(X)
        
            labels = kmeans.labels_

            # Calculate average silhouette score for the current k
            score = silhouette_score(X, labels)
            scores.append(score)

        # calculate best k
        combined_list = zip(k_vals, scores)
        combined_list_sorted = sorted(combined_list, key=lambda x: x[1], reverse=True)
        best_k, best_score = combined_list_sorted[0]

        final_kmeans = KMeans(
            n_clusters=best_k,
            n_init=50,
            random_state=42
        )
        labels = final_kmeans.fit_predict(X)
        return labels, final_kmeans
    
    def get_cluster_dict(self, data, keys, labels):
        # Build clusters dict
        phrases = [f"{data[k]["target_word"]} -> {data[k]["sentence"]}" for k in keys]
        clusters = defaultdict(list)
        for phrase, label in zip(phrases, labels):
            clusters[label].append(phrase)
        
        clusters = dict(sorted(clusters.items()))
    
    def sort_clusters_by_proximity(self, X, labels, kmeans, data, keys, top_25=False):
        """
        Returns clusters sorted by proximity to centroid (closest first)
        """

        centers = kmeans.cluster_centers_

        clusters = {}

        for i in range(len(centers)):
            # indices for this cluster
            idxs = np.where(labels == i)[0]

            if len(idxs) == 0:
                continue

            cluster_vecs = X[idxs]
            center = centers[i]

            # cosine similarity (since X is normalized)
            sims = cluster_vecs @ center

            # sort by similarity (descending)
            sorted_order = np.argsort(-sims)
            if top_25:
                len_top_25 = int(len(sorted_order)*0.25)
                sorted_order = sorted_order[:len_top_25]

            clusters[i] = [
                {
                    "similarity": float(sims[j]),
                    "word": data[keys[idxs[j]]]["target_word"],
                    "sentence": data[keys[idxs[j]]]["sentence"]
                }
                for j in sorted_order
            ]

        return clusters



def construct_docs(ppt_class: str, clusters: dict, dataset: str):
    issue_dict = {
        'tweets_immigration': "U.S. Immigration",
        'subframes_abortion': "U.S. Abortion Rights",
        'subframes_guns': "U.S. Gun Control",
        'podcasts': "U.S. Politics"
    }
    formatted_docs = {}
    for cluster_idx, cluster_items in clusters.items():
        text = [f"{item["word"]} -> {item["sentence"]}" for item in cluster_items]

        formatted_docs[f"{ppt_class}_{cluster_idx}"] = {
            "text": f"ISSUE: {issue_dict[dataset]}" + '\nNOUNS:' + '\n'.join(text)
        }
    
    return formatted_docs
    

def get_target_coref(metaphors: dict, session):
    for met_id, met_info in metaphors.items():
        # get target id from met_id
        doc_id, sent_idx, source_idx, target_idx = met_id.split("_")
        target_idx = int(target_idx)
        sent_id = f"{doc_id}_{sent_idx}"
        
        # get mentions with matching sentence
        mentions = session.query(CoreferenceMention).filter(CoreferenceMention.sentence_id == sent_id).all()

        # iterate over mentions to get the one where the target token is betwen start and end
        cluster_text = met_info["target_word"]
        for m in mentions:
            start_token_idx = int(m.start_token.id.split("_")[-1])
            end_token_idx = int(m.end_token.id.split("_")[-1])

            if target_idx >= start_token_idx and target_idx <= end_token_idx:
                rep_text = m.coreference_chain.representative_text
                cluster_text += f": {rep_text}"


        met_info["target_rep_text"] = cluster_text

    return metaphors

def main(args):

    logger = get_logger("create_target_groups")
    env_vars = load_env()
    set_seed(env_vars["RANDOM_SEED"])

    config = ExperimentConfig(
        task_type="create_target_groups",
        task_name="",
        dataset=args.dataset,
        logger=logger,
        env_vars=env_vars,
        labeled_only=args.labeled_only,
        device=env_vars["DEVICE"],
        random_seed=env_vars["RANDOM_SEED"],
        skip_load=True # do not check preprocessing on db, use as is (saves time)
    )

    # get data so we can query for target representative text
    data_loader = DatasetLoader(config)
    session = data_loader.load_preprocessed_data()

    # first construct clusters, then prompt for cluster labels
    # divide metaphors into person/place/thing based on llm classification results
    ppt_res_path = f"{env_vars["RESULTS_DIR"]}/feature_extractor/llm/{config.dataset_out_name}_target_ppto_class_qwen.json"
    ppt_results = load_json(ppt_res_path)["data"]
    div_ppt_results = {"person": {}, "place": {}, "thing": {}, "organization": {}}

    for met_id, ppt_info in ppt_results.items():
        noun_class = ppt_info["annotation"]["noun_classification"].lower()
        del ppt_info["annotation"]
        ppt_info["target_rep_text"] = None
        try:
            div_ppt_results[noun_class][met_id] = ppt_info
        except KeyError as e:
            logger.warning(f"Skipping invalid noun annotation: '{noun_class}'...")
            pass

    for ppt_class, met_data in div_ppt_results.items():

        # # get target coref rep text to cluster
        # mets = get_target_coref(mets, session)
        # target_coref = [m["target_rep_text"].lower().strip() for m in mets.values()]
        # unique_target_coref = list(set(target_coref))
        # cluster_data = unique_target_coref

        model = ContextualWordClustering()
        X, keys = model.build_dataset(met_data)
        X = model.preprocess(X)

        min_k = 5
        max_k = 15

        logger.info(f"Finding optimal k for clustering {ppt_class} nouns (min k = {min_k}, max k = {max_k})")
        labels, final_kmeans = model.cluster(X, min_k=min_k, max_k=max_k)
        clusters = model.sort_clusters_by_proximity(X, labels, final_kmeans, met_data, keys, args.top_25)


        prompt_file = f"target_{ppt_class}_extraction.json"

        # set up llm config
        config = LLMExperimentConfig(
            dataset=args.dataset,
            labeled_only=args.labeled_only,
            task_type="create_target_groups",
            logger=logger,
            env_vars=env_vars,
            device=env_vars["DEVICE"],
            random_seed=env_vars["RANDOM_SEED"],
            prompt_file=prompt_file,
            out_file=args.out_file,
            model_config_file=args.annotation_config,
            host=args.host,
            port=args.port,
            testing=args.testing,
            sample_size=args.sample_size,
            shots=args.shots,
            skip_load=args.skip_load,
        )
        config.answer_format = TargetGroupCreation

        logger.info("Constructing text for LLM prompting")
        data_formatted = construct_docs(ppt_class, clusters, config.dataset)
        save_to_json(data_formatted, "temp", f"{ppt_class}.json")

        ### SET UP LLM ANNOTATOR
        annotator = Annotator(
            config=config,
            data_entries=data_formatted
        )
        annotator.process_docs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create target groups')
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['tweets_immigration', 'subframes_immigration', 'subframes_guns', 'subframes_abortion', 'lcc-large', 'podcasts'],
        help="The dataset to measure metaphoricity of. Currently only 'wildchat' is supported."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="The number of documents from the dataset to annotate."
    )
    parser.add_argument(
        "--labeled_only",
        action='store_true',
        help="If set, only use documents that have human metaphor annotations (only for tweets_immigration dataset)."
    )
    parser.add_argument(
        "--annotation_config", type=str, default="default.yaml",
        help="Name of config file in src/llm_ann/configs directory"
    )
    parser.add_argument(
        "--out_file", type=str, default=None,
        help="The desired name of the output file."
    )
    parser.add_argument(
        '--host', metavar='HOST', required=True,
        help="The host for the Ollama API."
    )
    parser.add_argument(
        '--port', metavar='PORT',
        help="The port for the Ollama API."
    )
    parser.add_argument(
        "--testing",
        action='store_true',
        help="If set, the script will run in testing mode with a small sample of the data."
    )
    parser.add_argument(
        "--skip_load",
        action='store_true',
        help="If set, won't check dataset before loading."
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="The number of examples to add to the prompt. By default, just uses all examples included in the prompt json. "
    )
    parser.add_argument(
        "--shuffle_docs",
        action='store_true',
        help="If set, shuffles docs before processing."
    )
    parser.add_argument(
        "--top_25",
        action="store_true",
        help="If set, filters to top 25 examples nearest the cluster center for labeling."
    )

    args = parser.parse_args()
    main(args)
