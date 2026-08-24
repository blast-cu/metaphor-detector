import argparse
import logging
import os
from tqdm import tqdm
import random
import pandas as pd
import numpy as np

from src.utils import get_logger, load_env, load_json, save_to_json, get_token
from analysis.utils import get_path_info

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(
        description="Sample data to annotate for metaphoricity."
    )
    parser.add_argument(
        "--sample_size", type=int,
        required=True
    )
    parser.add_argument(
        "--dataset", type=str,
        required=True
    )
    parser.add_argument(
        "--file_path", type=str,
        required=True,
        help="Path to metaphors with features.",
    )
    parser.add_argument(
        "--threshold", type=float,
        default=0.4,
        help="Metaphor salience score cutoff (inclusive)"
    )

    args = parser.parse_args()        

    ### SET UP ENVIRONMENT VARIABLES AND LOGGER
    LOGGER = get_logger(f"export_mets", level=logging.INFO)
    env_vars = load_env(logger=LOGGER)
    random.seed(env_vars["RANDOM_SEED"])

    met_sample_size = int(0.9*args.sample_size)
    non_met_sample_size = args.sample_size - met_sample_size

    # load prior annotations from output directory
    out_dir = "results/to_annotate"
    file_index = 0
    os.makedirs(out_dir, exist_ok=True)
    
    # load files in the directory to check if we've already exported annotations for this dataset
    existing_files = os.listdir(out_dir)
    existing_file_indices, existing_path_anns = [], []
    for f in existing_files:
        if args.dataset in f and f.endswith(".csv"):
            index = f.split("_")[-1].replace(".csv", "")
            if index.isdigit():
                existing_file_indices.append(int(index))
                # load file and add ids to existing_path_anns
                df = pd.read_csv(os.path.join(out_dir, f))
                existing_path_anns.extend(df["id"].tolist())
            else:
                LOGGER.warning(f"Unexpected file format: {f}, skipping...")
    if existing_file_indices:
        file_index = max(existing_file_indices) + 1
        LOGGER.info(f"Found existing annotation files for {args.dataset}, with existing {len(existing_path_anns)} path annotations, incrementing file index to {file_index}...")            

    processed_metaphors = load_json(args.file_path)
    
    # 10% of our annotations need to be non-metaphorical, gather them into dict
    unfiltered_metaphor_data = load_json(f"results/llm_ann/llm_metaphor_classifier/{args.dataset}_source_verb_score_qwen.json")["data"]

    path_info = get_path_info(args.dataset)
    non_metaphors = {}
    for k, v in tqdm(unfiltered_metaphor_data.items(), desc="gathering metaphorical data"):
        if v["annotation"]["metaphor_salience_score"] < args.threshold:

            met_path = int(k.split("_")[-1])
            sdp = path_info[k]["sdp"]
            clean_text = v["text"].split('Sentence: ')[1].strip()
              
            non_metaphors[k] = {
                "met_score": v["annotation"]["metaphor_salience_score"],
                "source": get_token(met_path, sdp, LOGGER)["text"],
                "target": get_token(met_path, sdp, LOGGER, target_word=True)["text"],
                "sentence": clean_text
            }
    def stratified_sample(x, n):
        # Take all samples if group is smaller than n
        actual_n = min(len(x), n)
        return x.sample(actual_n)

    met_df = pd.DataFrame.from_dict(processed_metaphors, orient="index")
    met_df = met_df[met_df["met_score"] >= args.threshold]
    met_df = met_df[~met_df.index.isin(existing_path_anns)] # remove metaphors we've already annotated in prior files
    LOGGER.info(f"After filtering metaphors with score >= {args.threshold} and removing previously annotated paths, {len(met_df)} metaphors remain for sampling...")
    met_cols = met_df.columns
    
    met_df["met_score_rounded"] = met_df["met_score"].round(1)
    met_unique_scores = len(list(set(met_df["met_score_rounded"].to_list())))
    m_score_samples = int(met_sample_size / met_unique_scores)
    
    sampled_met_df = met_df.groupby('met_score_rounded', group_keys=False).apply(lambda x: stratified_sample(x, m_score_samples))
    sampled_met_df = sampled_met_df[met_cols] # remove rounded score column
    
    
    # stratified sampling over metaphor scores
    non_met_cols = ["met_score", "source", "target", "sentence"]
    non_met_df = pd.DataFrame.from_dict(non_metaphors, orient="index")
    non_met_df = non_met_df[~non_met_df.index.isin(existing_path_anns)] # remove non-metaphors we've already annotated in prior files
    LOGGER.info(f"After filtering non-metaphors with score < {args.threshold} and removing previously annotated paths, {len(non_met_df)} non-metaphors remain for sampling...")

    non_met_df["met_score_rounded"] = non_met_df["met_score"].round(1)
    non_met_unique_scores = len(list(set(non_met_df["met_score_rounded"].to_list())))
    nm_score_samples = int(non_met_sample_size / non_met_unique_scores)
    
    sampled_non_met_df = non_met_df.groupby('met_score_rounded', group_keys=False).apply(lambda x: x.sample(n=nm_score_samples, random_state=42))
    sampled_non_met_df = sampled_non_met_df[non_met_cols] # remove rounded score column
    # LOGGER.info(f"Sampled {len(sampled_met_df)} metaphors and {len(sampled_non_met_df)} non-metaphors...")

    ann_df = pd.concat([sampled_met_df, sampled_non_met_df])
    
    if len(ann_df) < args.sample_size:
        LOGGER.warning(f"Final sample too small: {len(ann_df)} < {args.sample_size}, sampling more metaphors to fill in")
        n_fill = args.sample_size - len(ann_df)

        leftover_metaphors = met_df[~met_df.index.isin(ann_df.index.tolist())]

        fill_metaphors = leftover_metaphors.sample(n_fill)
        fill_metaphors = fill_metaphors[met_cols]

        ann_df = pd.concat([ann_df, fill_metaphors])

    ann_df["id"] = ann_df.index
    idx = ann_df.index.to_list()
    np.random.shuffle(idx)
    ann_df = ann_df.loc[idx].set_index("id")

    LOGGER.info(f"Exporting {len(ann_df)} annotation samples...")
    print(ann_df.groupby("met_score").agg(count = ("source", "count")))
    ann_df.to_csv(f"{out_dir}/{args.dataset}_to_annotate_{file_index}.csv")


