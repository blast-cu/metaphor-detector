import os
from tqdm import tqdm
import argparse
import logging
import pickle
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizerFast, RobertaModel, AutoTokenizer
from datasets import Dataset

from src.utils import set_seed, load_env, get_logger, load_json
from src.data_loading import DatasetLoader
from src.experiment_config import ExperimentConfig

def main(args, env_vars, LOGGER):

    # when using fast tokenizer, must disable parallelism 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load data from xml files
    config = ExperimentConfig(
        task_type="gen_embeddings",
        task_name=args.model,
        dataset=args.dataset,
        logger=LOGGER,
        env_vars=env_vars,
        labeled_only=args.labeled_only,
        device=env_vars["DEVICE"],
        random_seed=seed,
        skip_load=True # SKIPPING LOAD FOR SPEEEED
    )

    data_loader = DatasetLoader(config)

    # read in exported metaphors
    dataset_out_name = config.dataset_out_name

    if args.llm_explanations:
        # read in llm results
        if args.old_results:
            llm_anns_df = data_loader.load_old_metaphor_classification_results(metaphors_only=args.metaphors_only)
        else:
            llm_anns_df = data_loader.load_metaphor_classification_results(metaphors_only=args.metaphors_only)

        texts = llm_anns_df["llm_explanation"].tolist()
        text_ids = llm_anns_df["id"].to_list()

    elif args.frame_entailments:

        fe_path = f"results/gen_entailments/llm/{config.dataset_out_name}_metaphor_framing_entailments_qwen.json"
        fe_results = load_json(fe_path, LOGGER)["data"]


        # the annotation is a dictionary where keys are frame element title, value is corresponding llm generation 
        texts = [
            " ".join(list(v["annotation"].values())) # make a string paragraph of all four sentences
            for v in fe_results.values()
        ]
        
        text_ids = [k for k in fe_results.keys()]

    elif args.ling_mets:

        llm_anns_df = data_loader.load_metaphor_classification_results(metaphors_only=args.metaphors_only)
        texts = llm_anns_df["sentence"].tolist()
        text_ids = llm_anns_df["id"].to_list()



    elif args.full_tweets:
        if args.labeled_only:
            LOGGER.info(f"Getting tweet-level embeddings for hand-annotated subset of tweets...")
            tweet_ann_data = data_loader.load_raw_data(
                metaphors_only=args.metaphors_only, metaphor_thresh=args.metaphor_thresh
            )
            texts = tweet_ann_data["text"].tolist()
            text_ids = tweet_ann_data["id_str"].tolist()

        else: 
            LOGGER.info(f"Getting tweet-level embeddings for unlabelled tweets, filtering to 35k we processed...")
            data = data_loader.load_raw_data(processed_only=True)
            texts = data["text"].tolist()
            text_ids = data["id_str"].tolist()

    else:
        raise NotImplementedError

    # USE SBERT TO EMBED DOCUMENTS
    LOGGER.info(f"Encoding {len(texts)} documents (pbar shows batches)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_list = model.encode(texts, show_progress_bar=True)

    embeddings = {}
    for idx, emb in enumerate(embedding_list):
        embeddings[str(idx)] = {
            "id": text_ids[idx],
            "emb": emb,
            "text": texts[idx]
        }

    out_path = f"{config.task_dir}/{dataset_out_name}_embeddings.pkl"

    if args.metaphors_only:
        out_path = out_path.replace("_embeddings.pkl", "_mets_only_embeddings.pkl")

    if args.llm_explanations:
        out_path = out_path.replace("_embeddings.pkl", "_qwen_expl_embeddings.pkl")

    if args.frame_entailments:
        out_path = out_path.replace("_embeddings.pkl", "_qwen_fe_embeddings.pkl")

    if args.ling_mets:
        out_path = out_path.replace("_embeddings.pkl", "_ling_mets_embeddings.pkl")
    
    if args.full_tweets:
        out_path = out_path.replace("_embeddings.pkl", "_full_tweet_embeddings.pkl")

    if args.old_results:
        out_path = out_path.replace("_embeddings.pkl", "_old_results_embeddings.pkl")

    
    LOGGER.info(f"Saving {len(embeddings)} embeddings as pkl to {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use a rb approach to extract all sentences which could invoke a metaphor.")
    parser.add_argument(
        "--dataset",
        type=str
    )

    parser.add_argument(
        "--labeled_only",
        action="store_true"
    )
    
    parser.add_argument(
        "--model",
        choices=["sbert"],
        default="sbert",
        type=str
    )

    parser.add_argument(
        "--llm_explanations",
        action="store_true"
    )

    parser.add_argument(
        "--frame_entailments",
        action="store_true"
    )

    parser.add_argument(
        "--ling_mets",
        action="store_true"
    )

    parser.add_argument(
        "--metaphors_only",
        action="store_true"
    )

    parser.add_argument( # for tweets only
        "--metaphor_thresh",
        action="store_true"
    )

    parser.add_argument(
        "--full_tweets",
        action="store_true"
    )

    parser.add_argument(
        "--old_results",
        action="store_true"
    )

    env_vars = load_env()
    seed = env_vars["RANDOM_SEED"]
    set_seed(seed)

    LOGGER = get_logger("train_src_class", level=logging.DEBUG)
    LOGGER.info("Starting script to train source classifier...")

    args = parser.parse_args()
    if args.llm_explanations and args.frame_entailments:
        LOGGER.warning("llm_explanations and frame_entailments both set true, switching to frame_entailments only")
        args.llm_explanations = False

    if args.frame_entailments and (not args.metaphors_only):
        LOGGER.warning("metaphors_only not set, but invalid for frame_entailments option, setting metaphors_only = True")
        args.metaphors_only = True

    if args.full_tweets and args.dataset != "tweets_immigration":
        LOGGER.error("Can only process 'full_tweets' option for 'tweets_immigration' dataset...")
        raise ValueError("Invalid argument")

    main(args, env_vars, LOGGER)