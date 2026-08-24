import json
import os
import argparse
import logging
import random
import pandas as pd
from tqdm import tqdm

from src.utils import get_logger, load_env, load_json, save_to_json, get_token, VERB_PATHS, METAPHOR_PATHS
from src.experiment_config import ExperimentConfig
from analysis.utils import get_target_coref, get_path_info
from src.data_loading import DatasetLoader


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(
        description="Analyze a dataset using Kmeans clustering."
    )
    parser.add_argument(
        "--dataset", type=str,
        required=True,
        help="Dataset we're exporting metaphors for.",
    )
    parser.add_argument(
        "--labeled", 
        action="store_true"
    )
    parser.add_argument(
        "--file_path", type=str,
        required=True,
        help="Path to the file of llm classification results to export.",
    )
    parser.add_argument(
        "--threshold", type=float,
        default=0.3,
        help="Metaphor salience score cutoff (inclusive)"
    )
    parser.add_argument(
        "--sample", type=int,
        default=None,
        help="If set, sets number of metaphors to randomly select for export."
    )
    parser.add_argument(
        "--get_coref", action="store_true"
    )
    args = parser.parse_args()        

    ### SET UP ENVIRONMENT VARIABLES AND LOGGER
    LOGGER = get_logger(f"export_mets", level=logging.INFO)
    env_vars = load_env(logger=LOGGER)

    path_info = get_path_info(args.dataset)

    data = load_json(args.file_path, LOGGER)
    config_data = data["config"]
    config = ExperimentConfig.from_dict(config_data, logger=LOGGER)
    config.repo_dir = env_vars["REPO_PATH"]  # reset repo dir
    config.raw_data_dir = env_vars["RAW_DATA_DIR"]  # reset raw data dir
    config.db_dir = env_vars["DB_DIR"]  # reset db dir
    config.skip_load = True

    out_dir = "results/metaphors/"
    os.makedirs(out_dir, exist_ok=True)

    expected_cols = ["score", "source", "target", "sentence", "met_path"]
    
    data = data["data"]
    metaphor_data = {} # collect score, target
    for k, v in tqdm(data.items(), desc="gathering metaphorical data"):
        if v["annotation"]["metaphor_salience_score"] >= args.threshold:

            met_path = int(k.split("_")[-1])
            sdp = path_info[k]["sdp"]
            clean_text = v["text"].split('Sentence: ')[1].strip()
              
            metaphor_data[k] = {
                "score": v["annotation"]["metaphor_salience_score"],
                "source": v["source_word"],
                "target": v["target_word"],
                "sentence": clean_text,
                "met_path": METAPHOR_PATHS[met_path]
            }
    
    # sample before coref bc it takes forever
    if args.sample:
        LOGGER.info(f"Selecting {args.sample}/{len(metaphor_data)} to process...")
        random.seed(env_vars["RANDOM_SEED"])
        key_sample = random.sample(list(metaphor_data.keys()), args.sample)
        metaphor_data = {k: metaphor_data[k] for k in key_sample}

    data_loader = DatasetLoader(config)
    session = data_loader.load_data()
    if args.get_coref:
        expected_cols.append("target_rep_text")
        _, metaphor_data = get_target_coref(metaphor_data, session)

    # save to json before formatting data for csv export
    save_to_json(metaphor_data, out_dir, f"{config.dataset}.json", logger=LOGGER)

    # # convert data to list for export to csv
    # for k, v in tqdm(metaphor_data.items(), desc="convert data to list for export to csv"):
    #     v["id"] = k
    #     v["target"] = v["target"]["text"]
    #     v["source"] = v["source"]["text"]
    # data_list = list(metaphor_data.values())
    # df = pd.DataFrame.from_dict(data_list).sort_values('score', ascending=False)
    # df = df[expected_cols]

    # # export csv
    # df.to_csv(f"{out_dir}/{config.dataset}.csv")
