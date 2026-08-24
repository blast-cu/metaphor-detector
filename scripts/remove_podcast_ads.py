import pickle 
import argparse
from tqdm import tqdm

from src.utils import get_logger, load_env, load_json, save_to_json
from src.experiment_config import ExperimentConfig

"""
Add LLM-gen features to embeddings file
"""

def main():
    env_vars = load_env()
    LOGGER = get_logger("consolidate_features")
    seed = env_vars["RANDOM_SEED"]

    config = ExperimentConfig(
        task_type="metaphor_classification", # output directory - RESULTS_DIR/task_type/task_name
        task_name="llm",
        dataset="podcasts",
        logger=LOGGER,
        env_vars=env_vars,
        labeled_only=False,
        device=env_vars["DEVICE"],
        random_seed=seed,
        skip_load=True # do not check preprocessing on db, use as is (saves time)
    )

    # load the original metaphor classification results
    met_class_path = f"{config.task_dir}/podcasts_source_verb_target_noun_binary_met_class_qwen_with_ads.json"
    met_class_w_config = load_json(met_class_path)
    met_class_res = met_class_w_config["data"]

    # # load the frame entailments
    # fe_path = f"{env_vars["RESULTS_DIR"]}/gen_entailments/llm/podcasts_metaphor_framing_entailments_qwen_with_ads.json"
    # fe_w_config = load_json(fe_path)
    # fe_res = fe_w_config["data"]


    # read in ad classification results
    ad_class_path = f"{env_vars['RESULTS_DIR']}/remove_ads/llm/podcasts_ad_classification_qwen.json"
    ad_class_res = load_json(ad_class_path)["data"]

    # verify that all llm generations correctly formatted
    for v in ad_class_res.values():
        ann = v["annotation"]

        if ann["classification"] not in ["advertisement", "podcast"]:
            raise ValueError(f"invalid annotation: {ann["classification"]}")

    LOGGER.info("All LLM annotations for ad removal are valid, continuing...")

    # remove ads from metaphor_classification results
    og_len = len(met_class_res)
    met_class_res = {k: v for k, v in met_class_res.items()
                     if (k not in ad_class_res) # keep non-metaphorical (not annotated for ad content)
                        or (ad_class_res[k]["annotation"]["classification"] == "podcast")} # or those not deemed ads

    LOGGER.info(f"Removed {og_len - len(met_class_res)} sentences from ads, {len(met_class_res)} remaining")

    # save results 
    met_class_w_config["data"] = met_class_res
    mc_out_filename = "podcasts_source_verb_target_noun_binary_met_class_qwen.json"
    save_to_json(met_class_w_config, config.task_dir, mc_out_filename, logger=LOGGER)

    # # remove ads from frame entailments
    # og_len = len(fe_res)
    # fe_res = {k: v for k, v in fe_res.items()
    #             if (ad_class_res[k]["annotation"]["classification"] == "podcast")} # or those not deemed ads

    # LOGGER.info(f"Removed {og_len - len(fe_res)} sentences from ads, {len(fe_res)} remaining")

    # # save results
    # fe_w_config["data"] = fe_res
    # fe_out_path = f"{env_vars["RESULTS_DIR"]}/gen_entailments/llm/"
    # fe_out_filename = "podcasts_metaphor_framing_entailments_qwen.json"
    # save_to_json(fe_w_config, fe_out_path, fe_out_filename, logger=LOGGER)
    



if __name__ == "__main__":
    main()