import os
import argparse
import pickle
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.utils import load_env, get_logger, load_json
from src.lr import RegressionModel
from analysis.utils import get_llm_metaphor_class_df


def main(args):
    env_vars = load_env()
    logger = get_logger("combine embeddings")

    dataset_name = args.dataset
    if args.labeled_only:
        dataset_name += "_labelled"

    met_embeddings_path = f"{env_vars["RESULTS_DIR"]}/gen_embeddings/get_met_embeddings/{dataset_name}_sbert_qwen_expl.pkl"
    tweet_embeddings_path = met_embeddings_path.replace("qwen_expl.pkl", "full_tweets.pkl")
    
    with open(met_embeddings_path, 'rb') as f:
        metaphor_embeddings = pickle.load(f)
        logger.info(f"Loaded {len(metaphor_embeddings)} metaphor embeddings")

    with open(tweet_embeddings_path, 'rb') as f:
        tweet_embeddings = pickle.load(f)
        logger.info(f"Loaded {len(tweet_embeddings)} tweet embeddings")
        # '1486': {'id': '1115077237907316736', 'emb': array

    for t_idx, t_emb_info in tqdm(tweet_embeddings.items()):
        t_emb = t_emb_info['emb']
        t_id = t_emb_info["id"]

        # find the corresponding
        filter_id = f"{t_id}_"
        corr_mets = []
        for m_idx, m_emb_info in metaphor_embeddings.items():
            m_id = m_emb_info["id"]
            if m_id.startswith(filter_id):
                m_emb = m_emb_info['emb']
                corr_mets.append(m_emb)

        # average mets
        if len(corr_mets) == 0:
            av_mets = np.zeros(t_emb.shape[0])
            print(av_mets.shape)
        else:
            av_mets = np.mean(corr_mets, axis=0)
        # print(av_mets.shape)

        # concat to tweets emb
        if args.concat:
            final_emb = np.concat([t_emb, av_mets])
        elif args.mean:
            final_emb = np.mean([t_emb, av_mets], axis=0)
            print(final_emb.shape)
        else:
            raise NotImplementedError

        t_emb_info['emb'] = final_emb
    
    out_path = f"{env_vars["RESULTS_DIR"]}/gen_embeddings/get_met_embeddings/tweets_immigration_labelled_sbert_embeddings.pkl"
    if args.concat:
        out_path = out_path.replace("sbert_embeddings.pkl", "concat_sbert_embeddings.pkl")
    elif args.mean:
        out_path = out_path.replace("sbert_embeddings.pkl", "mean_sbert_embeddings.pkl")
    
    logger.info(f"Saving {len(tweet_embeddings)} embeddings to {out_path}")
    with open(out_path, "wb") as fOut:
        pickle.dump(tweet_embeddings, fOut)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate KMeans Clustering with Logistic Regression')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset to evaluate on. Options: "subframes_guns", "tweets_immigration')
    parser.add_argument('--labeled_only', action="store_true")
    parser.add_argument("--concat", action="store_true")
    parser.add_argument("--mean", action="store_true")


    args = parser.parse_args()
    main(args)
