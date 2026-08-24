import os
import argparse
import pickle
import logging
import pandas as pd

from src.utils import load_env, get_logger
from src.lr import RegressionModel
import numpy as np

from src.experiment_config import ExperimentConfig
from src.data_loading import DatasetLoader, load_target_prediction_res

"""
Evaluation script for KMeans clustering using Logistic Regression as a downstream task to evaluate the quality of the clusters.
We make predictions over various data labels via LR model trained on feature vectors capturing cluster results
"""
DEFAULT_K_LIST = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 500, 1000]

def add_target_info_to_embs(dataset_name, metaphor_embeddings, env_vars, logger):
    # read in files with target group predictions
    fg_results_path = f"{env_vars["RESULTS_DIR"]}/llm_ann/llm_feature_extractor/{dataset_name}_CLASS_features.json"
    fg_res = pd.DataFrame()
    for c in ["person", "place", "thing"]:
        path = fg_results_path.replace("CLASS", c)
        res = load_target_prediction_res(path)
        fg_res = pd.concat([fg_res, res])
    fg_res = fg_res.set_index("id")

    def get_sdp_target_group(m_id):
        if m_id not in fg_res.index:
            return ""
        else: return fg_res.loc[m_id, "target_group"]

    for _, m_info in metaphor_embeddings.items():
        m_id = m_info["id"]
        m_info["target_group"] = get_sdp_target_group(m_id)

    unique_target_groups = [m_info["target_group"] for m_info in metaphor_embeddings.values()]
    logger.info(f"Collected {len(unique_target_groups)} unique target groups")
    return metaphor_embeddings


def load_filtered_raw_data(
    dataset:str, labeled_only:bool, label_list:list, filter_to_met_docs:bool, 
    logger: logging.Logger, data_loader: DatasetLoader
) -> dict:
    """
    Load the data labels that we will predict.
    """
    if dataset == "tweets_immigration" or dataset == "subframes_abortion" or dataset == "subframes_guns" or dataset =="podcasts": # read in feature data
        if labeled_only:
            data_filtered = data_loader.load_raw_data(metaphors_only=filter_to_met_docs, metaphor_thresh=0.3)
        
        else:
            data_filtered = data_loader.load_raw_data(processed_only=True)

        # check that label we want to predict is in data columns
        for label in label_list:
            if label not in data_filtered.columns:
                logger.error(f"User specified label {label} not in data. Try: {data_filtered.columns}")
                raise ValueError


        data_filtered = data_filtered.set_index('id')
        docs = data_filtered.to_dict(orient="index")
        logger.info(f"Loaded {len(docs)} doc entries")
        return docs

    else: 
        raise NotImplementedError


def main(args, env_vars, logger):

    # if args.dataset != "tweets_immigration":
    #     raise NotImplementedError

    # SET UP CONFIG
    config = ExperimentConfig(
        task_type="cluster_lr",
        task_name="",
        dataset=args.dataset,
        labeled_only=args.labeled_only,
        logger=logger,
        env_vars=env_vars
    )
    data_loader = DatasetLoader(config)

    # SET UP LABEL TO PREDICT - this is hard coded for the tweets dataset
    if args.label: # if label specified, go with that
        label_list = [args.label]
        if args.label == "concept":
            args.predict_over_met_docs = True
    elif args.labeled_only: # default to 'concept' for labeled dataset
        label_list = ["concept"]
        args.predict_over_met_docs = True
    else: # or list of LLM-induced labels for unlabeled dataset
        label_list = ["Thematic", "Episodic", "Hero", "Threat", "Victim", "Health and Safety", "Legality, Constitutionality, Jurisdiction", "Morality and Ethics", "Policy Prescription and Evaluation",	"Political Factors and Implications", "Public Sentiment", "Quality of Life", "Security and Defense", "Capacity and Resources", "Crime and Punishment",	"Cultural Identity", "External Regulation and Reputation", "Fairness and Equality"]

    
    # IF TWEET AND MET EMBEDDINGS SPECIFIED, COMBINE RESULTs
    if args.tweet_embedding_name and args.met_embedding_name: 

        # if the best k arguments aren't set, try everything
        ml_k_list = [args.best_ml_k] if args.best_ml_k else DEFAULT_K_LIST
        tl_k_list = [args.best_tl_k] if args.best_tl_k else DEFAULT_K_LIST


        # LOAD EMBEDDING FILE TO GET SDP IDS 
        met_embeddings_path = f"{env_vars["RESULTS_DIR"]}/gen_embeddings/{args.embedding_model}/{config.dataset_out_name}_{args.met_embedding_name}_embeddings.pkl"
        with open(met_embeddings_path, 'rb') as f:
            metaphor_embeddings = pickle.load(f)
            logger.info(f"Loaded {len(metaphor_embeddings)} metaphor embeddings")

        tweet_embeddings_path = f"{env_vars["RESULTS_DIR"]}/gen_embeddings/{args.embedding_model}/{config.dataset_out_name}_{args.tweet_embedding_name}_embeddings.pkl"
        with open(tweet_embeddings_path, 'rb') as f:
            tweet_embeddings = pickle.load(f)
            logger.info(f"Loaded {len(tweet_embeddings)} tweet embeddings")

        # load LLM metaphor scores
        llm_res_df = data_loader.load_metaphor_classification_results(metaphors_only=args.metaphors_only, metaphor_thresh=args.metaphor_thresh)
        llm_res_df = llm_res_df.set_index("id")

        # ADD INDUCED TARGET DOMAIN INFO TO MET EMBEDDINGS
        if args.add_target_info: # TODO: check this
            metaphor_embeddings = add_target_info_to_embs(config.dataset_out_name, metaphor_embeddings, env_vars, logger)

        metaphor_ids = [v["id"] for v in metaphor_embeddings.values()]
        if args.testing:
            metaphor_ids = metaphor_ids[:100]

        sdp_doc_ids = [m_id.split("_")[0] for m_id in metaphor_ids]
        logger.info(f"sdps match to {len(list(set(sdp_doc_ids)))} unique documents")

        docs = load_filtered_raw_data(
            config.dataset, labeled_only=config.labeled_only, label_list=label_list, 
            filter_to_met_docs=args.predict_over_met_docs, logger=logger, data_loader=data_loader
        )

        logger.info(f"Running combined setting with metaphor k: [{ml_k_list}] tweet k: [{tl_k_list}]")
        for ml_k in ml_k_list:
            met_cluster_path = f"{env_vars["RESULTS_DIR"]}/cluster_embeddings/kmeans/{args.dataset_out_name}/{args.met_embedding_name}_clusters_{ml_k}.pkl"
            with open(met_cluster_path, 'rb') as f:
                met_clusters = pickle.load(f)

            for tl_k in tl_k_list:
                # load clusters
                tweet_cluster_path = f"{env_vars["RESULTS_DIR"]}/cluster_embeddings/kmeans/{args.dataset_out_name}/{args.tweet_embedding_name}_clusters_{tl_k}.pkl"
                with open(tweet_cluster_path, 'rb') as f:
                    tweet_clusters = pickle.load(f)

                # get path to save classification reports for each lr model
                results_dir = f"{env_vars["RESULTS_DIR"]}/cluster_lr/{config.dataset_out_name}/LABEL/combined_clusters"
                if args.metaphors_only:
                    results_dir = results_dir.replace("combined_clusters", "combined_clusters_mets_only")

                # run logistic regression on the document embeddings to predict the labels and save report
                model = RegressionModel(env_vars["RANDOM_SEED"])
                model.run_combined_features(met_clusters, tweet_clusters, metaphor_embeddings, tweet_embeddings, docs, ml_k, tl_k, \
                                label_list, results_dir, add_target_info=args.add_target_info)

    else:
        embedding_name = args.tweet_embedding_name
        if not embedding_name: # if tweet embedding name is none, switch to mets
            embedding_name = args.met_embedding_name
        
        # SET VARIABLES FOR LOOPING THROUGH LR EXPERIMENT
        if args.k:
            k_list = [args.k]
        elif args.labeled_only:
            k_list = DEFAULT_K_LIST
        else:
            k_list = DEFAULT_K_LIST


        # LOAD EMBEDDING FILE TO GET SDP IDS 
        embeddings_path = f"{env_vars["RESULTS_DIR"]}/gen_embeddings/{args.embedding_model}/{config.dataset_out_name}_{embedding_name}_embeddings.pkl"
        with open(embeddings_path, 'rb') as f:
            metaphor_embeddings = pickle.load(f)
            logger.info(f"Loaded {len(metaphor_embeddings)} embeddings")

        # LOAD MET CLASSIFICATION RESULTS
        if args.met_embedding_name:
            if args.old_results:
                llm_res_df = data_loader.load_old_metaphor_classification_results(metaphors_only=args.metaphors_only)
            else: 
                llm_res_df = data_loader.load_metaphor_classification_results(metaphors_only=args.metaphors_only)
            llm_res_df = llm_res_df.set_index('id')
        
        # ADD OUR INDUCED TARGET DOMAIN INFO TO MET EMBEDDINGS (not used in workshop paper)
        if args.add_target_info:
            metaphor_embeddings = add_target_info_to_embs(config.dataset_out_name, metaphor_embeddings, env_vars, logger)


        # CREATE DOC DATA DICT where key is doc_id and value has the label we want to predict
        metaphor_ids = [v["id"] for v in metaphor_embeddings.values()]
        if args.testing:
            metaphor_ids = metaphor_ids[:100]

        sdp_doc_ids = [m_id.split("_")[0] for m_id in metaphor_ids]
        logger.info(f"sdps match to {len(list(set(sdp_doc_ids)))} unique documents")
        docs = load_filtered_raw_data(
            config.dataset, config.labeled_only, 
            label_list, args.predict_over_met_docs, logger, data_loader
        )

        # RUN THE EXPERIMENT
        for k in k_list:

            # get path to save classification reports for each lr model
            results_dir = f"{env_vars["RESULTS_DIR"]}/cluster_lr/{config.dataset_out_name}/LABEL/kmeans/{embedding_name}"

            # LOAD CLUSTERS
            cluster_path = f"{env_vars["RESULTS_DIR"]}/cluster_embeddings/kmeans/{config.dataset_out_name}/{embedding_name}_clusters_{k}.pkl"
            if args.pckmeans: # specify structured clustering, change path accordingly
                cluster_path = cluster_path.replace("/kmeans/", "/pckmeans/")
                cluster_path = cluster_path.replace(".pkl", f"_w{args.w_cl}.pkl")

                # also change results path
                results_dir = results_dir.replace("/kmeans/", "/pckmeans/")

            with open(cluster_path, 'rb') as f:
                clusters = pickle.load(f)

            # run logistic regression on the document embeddings to predict the labels and save report
            model = RegressionModel(env_vars["RANDOM_SEED"])
            model.run_regression(
                clusters, metaphor_embeddings, docs,
                label_list, k, results_dir, 
                w_cl = args.w_cl,
                filter_labels=False, add_target_info=args.add_target_info
            )

        



if __name__ == "__main__":
    # set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate KMeans Clustering with Logistic Regression')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to evaluate on. Options: "subframes_guns", "tweets_immigration')
    parser.add_argument('--labeled_only', action="store_true")
    
    parser.add_argument('--label', type=str, required=False, help='Label to predict. If not set, will loop through labels.')
    parser.add_argument('--k', type=int, required=False, help='Number of clusters used for kmeans')
    parser.add_argument('--add_target_info', action="store_true")

    parser.add_argument('--pckmeans', action='store_true')
    parser.add_argument("--w_cl", type=float, help="weight for cannot-link constraints (pckmeans only)")

    parser.add_argument("--embedding_model", default="sbert", help="model used to generate embeddings, only sbert implemented for now")
    parser.add_argument('--tweet_embedding_name', type=str, default=None)
    parser.add_argument('--met_embedding_name', type=str, default=None)
    parser.add_argument('--old_results', action="store_true")

    parser.add_argument('--metaphors_only', action="store_true", help="use our predicted metaphor info")
    parser.add_argument("--metaphor_thresh", type=float, default=0.3)
    parser.add_argument('--predict_over_met_docs', action="store_true", help="use only samples from the raw dataset which are metaphorical")

    parser.add_argument('--best_tl_k', type=int)
    parser.add_argument('--best_ml_k', type=int)

    parser.add_argument('--concat', action="store_true")
    parser.add_argument('--mean', action="store_true")

    parser.add_argument('--testing', action="store_true")


    logger = get_logger(f"llm_annotate_metaphor", level=logging.INFO)
    env_vars = load_env(logger=logger)

    args = parser.parse_args()
    main(args, env_vars, logger)