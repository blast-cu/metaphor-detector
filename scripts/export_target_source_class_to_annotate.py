import argparse
import logging
from tqdm import tqdm
import random
import pandas as pd

from src.utils import load_env, set_seed, get_logger, load_json, save_to_json
from src.experiment_config import ExperimentConfig

def main(args):
    env_vars = load_env()
    seed = env_vars["RANDOM_SEED"]
    set_seed(seed)

    LOGGER = get_logger("train_src_class", level=logging.DEBUG)
    LOGGER.info("Starting script to set agent/patient structure...")

    config = ExperimentConfig(
        task_type="annotation",
        task_name="lm_representation",
        dataset=args.dataset,
        logger=LOGGER,
        env_vars=env_vars,
        labeled_only=args.labeled_only,
        device=env_vars["DEVICE"],
        random_seed=seed,
        skip_load=True # SKIPPING LOAD FOR SPEEEED
    )

    # sample from ppto classifications, balanced over these
    ppto_file_path = f"{env_vars["RESULTS_DIR"]}/feature_extractor/llm/{config.dataset_out_name}_target_ppto_class_qwen.json"
    ppto_class = load_json(ppto_file_path, LOGGER)["data"]

    group_class = {}
    for ppto in ["person", "place", "thing", "organization"]:
        group_file_path = f"{env_vars["RESULTS_DIR"]}/feature_extractor/llm/{config.dataset_out_name}_target_{ppto}_class_qwen.json"
        group_class[ppto] = load_json(group_file_path)["data"]

    # load metaphor classifications
    met_class_path = f"{env_vars["RESULTS_DIR"]}/metaphor_classification/llm/{config.dataset_out_name}_source_verb_target_noun_binary_met_class_qwen.json"
    met_class_data = load_json(met_class_path)["data"]

    # load source verb classifications
    source_verb_path = f"{env_vars["RESULTS_DIR"]}/classify_source_verbs/llm/verbs_classify_qwen.json"
    source_verb_class = load_json(source_verb_path)["data"]

    # load llm generated frame entailments
    entailments_path = f"{env_vars["RESULTS_DIR"]}/gen_entailments/llm/{config.dataset_out_name}_metaphor_framing_entailments_qwen.json"
    entailments = load_json(entailments_path)["data"]

    # sample n from these
    sampled_items = random.sample(list(ppto_class.items()), args.n)

    # convert the sampled pairs back into a dictionary
    sampled_mets = dict(sampled_items)
    LOGGER.info(f"Loaded {args.n} metaphors to annotate...")

    # add all the data we need to the dict, then convert to df
    for k, v in sampled_mets.items():
        noun_group = sampled_mets[k]["annotation"]["noun_classification"]
        sampled_mets[k]["noun_group"] = noun_group
        sampled_mets[k]["target_group"] = group_class[noun_group][k]["annotation"]["target_group"]

        source_word = met_class_data[k]["source_word"] 
        sampled_mets[k]["source_word"] = source_word
        sampled_mets[k]["source_schema"] = source_verb_class[source_word.lower().strip()]["annotation"]["image_schema_group"]
        sampled_mets[k]["frame_entailments"] = " ".join(list(entailments[k]["annotation"].values()))

        # remove the unused stuff
        del sampled_mets[k]["text"]
        del sampled_mets[k]["annotation"]

    # save_to_json(sampled_mets, config.task_dir, f"{config.dataset_out_name}_{args.n}_to_annotate.json")
    df = pd.DataFrame.from_dict(sampled_mets, orient="index")
    df.to_csv(f"{config.task_dir}/{config.dataset_out_name}_{args.n}_to_annotate.csv")




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
        "--n",
        type=int
    )
    
    args = parser.parse_args()
    main(args)