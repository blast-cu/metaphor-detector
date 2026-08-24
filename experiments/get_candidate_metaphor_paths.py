
import logging
import argparse
from tqdm import tqdm
import concurrent # for parallelization

from src.utils import load_env, set_seed, get_logger, sample_documents, save_to_json, load_json
from src.experiment_config import ExperimentConfig
from src.data_loading import DatasetLoader
from src.rb_filter_utils import MetaphorFilter


def main(args, env_vars, LOGGER):

    LOGGER.info("Starting script to extract candidate metaphor paths...")

    # set seed
    seed = env_vars["RANDOM_SEED"]
    set_seed(seed)

    # experiment config
    # load data from xml files
    config = ExperimentConfig(
        task_type="get_candidate_metaphors",
        task_name="rule_based",
        dataset=args.dataset,
        env_vars=env_vars,
        logger=LOGGER,
        skip_load=args.skip_load,
        labeled_only=args.labeled_only,
        device=env_vars["DEVICE"],
        random_seed=seed,
        num_workers=args.num_workers
    )

    # LOAD DATASET
    data_loader = DatasetLoader(config)
    session = data_loader.load_preprocessed_data()
    documents = sample_documents(session, config)

    ms_model_path = args.model_path
    metaphor_filter = MetaphorFilter(config=config, verbs_only=True, ms_model_path=ms_model_path, logger=LOGGER)

    # if we already have the sdps, load from json
    if args.sdp_file is not None:
        LOGGER.info(f"Loading pre-existing metaphor paths from {args.sdp_file}")
        valid_paths = load_json(args.sdp_file)["data"]

    else: # need to compute them from scratch
        if args.testing:
            documents = documents[:5]

        # GET ALL SENTENCES WTIH 2 OR MORE POS OF INTEREST
        candidate_sentences = metaphor_filter.get_candidate_sentences(documents)
        LOGGER.info(f"Found {len(candidate_sentences)} candidate sentences with 2+ pos of interest")

        # GET SDPS BETWEEN NOUNS, VERBS IN ALL DEPENDENCY PATHS
        sdps = metaphor_filter.get_sdps(candidate_sentences)
        clean_sdps = metaphor_filter.clean_sdps(sdps)
        LOGGER.info(f"{len(clean_sdps)} SDPs found in dataset")

        # CHECK FOR SDP METAPHOR MATCHES
        valid_paths = metaphor_filter.check_metaphor_paths(clean_sdps)
        LOGGER.info(f"{len(valid_paths)}/{len(clean_sdps)} are valid metaphor paths")

        # CONSOLIDATE TO TARGET, SOURCE PAIRS
        target_source_pairs = metaphor_filter.get_target_source_pairs(valid_paths, session)
        LOGGER.info(f"Consolidated to {len(target_source_pairs)} (target, source) pairs")

        if config.dataset == "lcc-large" and config.labeled_only:
            target_source_pairs = metaphor_filter.filter_lcc_data(session, target_source_pairs)
            LOGGER.info(f"Filtered down to {len(target_source_pairs)} for lcc dataset")

        # if lcc, need to filter down to matches

    out_file_name = f"{config.dataset_out_name}_metaphor_paths.json"
    
    # ADD METAPHOR SCORE FOR EACH PATH
    if args.model_path is None:
        LOGGER.info("No model path specified, exporting without metaphor scores...")
    else:
        target_source_pairs = metaphor_filter.get_scores(session, target_source_pairs)
        out_file_name = out_file_name.replace(".json", "_scores.json")
    
    # EXPORT MATCHES (must include token ids)
    final_json = {"config": config.to_dict(), "data": target_source_pairs}
    save_to_json(final_json, config.task_dir, out_file_name, logger=LOGGER)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use a rb approach to extract all sentences which could invoke a metaphor.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['tweets_immigration', 'subframes_immigration', 'subframes_guns', 'subframes_abortion', 'lcc-large', 'podcasts'],
        help="The dataset to measure metaphoricity of. Currently only 'wildchat' is supported."
    )
    parser.add_argument(
        "--sdp_file",
        type=str,
        default=None,
        help="Path to pre-computed sdps."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned roberta metaphor score classifier."
    )
    parser.add_argument(
        "--labeled_only",
        action='store_true',
        help="If set, only use documents that have human metaphor annotations (only for tweets_immigration dataset)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for parallel processing."
    )
    parser.add_argument(
        "--skip_load",
        action='store_true',
        help="Do not load data from raw files, just use the db we have already."
    )
    parser.add_argument(
        "--testing",
        action='store_true',
        help="Run in testing mode with reduced candidate sentences."
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Run in debug mode with more print statements."
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = get_logger(f"rule_based_met_filter", level=log_level)
    env_vars = load_env(logger=logger)

    main(args, env_vars, logger)