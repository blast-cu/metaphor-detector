import pickle 
import argparse
from tqdm import tqdm

from src.utils import get_logger, load_env, load_json
from src.experiment_config import ExperimentConfig

"""
Add LLM-gen features to embeddings file
"""

def validate_group_label(og_group_ann, valid_group_labels, logger):
    group_ann = og_group_ann.split(":")[0] # in case llm included definition
    group_ann = group_ann.split(" (e.g.,")[0] # in case llm included example

    if group_ann == "Unclear" or group_ann in ["Not a thing", "Not a person", "Not a place", "Not an organization"]:
        return "Other"
    
    if group_ann not in valid_group_labels:
        logger.error(f"Invalid group label: {og_group_ann}, setting to 'Other'")
        return "Other"
    else: 
        return group_ann

def main(args):
    env_vars = load_env()
    LOGGER = get_logger("consolidate_features")
    seed = env_vars["RANDOM_SEED"]

    config = ExperimentConfig(
        task_type="cluster_embeddings", # output directory - RESULTS_DIR/task_type/task_name
        task_name="kmeans",
        dataset=args.dataset,
        logger=LOGGER,
        env_vars=env_vars,
        labeled_only=args.labeled_only,
        device=env_vars["DEVICE"],
        random_seed=seed,
        skip_load=True # do not check preprocessing on db, use as is (saves time)
    )

    # load the embeddings for the dataset
    input_path = f"{env_vars["RESULTS_DIR"]}/gen_embeddings/{args.embedding_model}/{config.dataset_out_name}_{args.embedding_name}_embeddings.pkl"
    LOGGER.info(f"Loading embeddings from {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f) # {'metaphor_idx': {"id", "emb", "text"}}

    # load the metaphor classification results to create id: source verb dict
    met_class_path = f"{env_vars["RESULTS_DIR"]}/metaphor_classification/llm/{config.dataset_out_name}_source_verb_target_noun_binary_met_class_qwen.json"
    met_class_res = load_json(met_class_path)["data"]
    id_source_map = {k: v["source_word"].lower().strip() for k, v in met_class_res.items()}

    # load verb classification results, validate results
    verb_class_path = "results/classify_source_verbs/llm/verbs_classify_qwen.json"
    verb_class_res = load_json(verb_class_path)["data"]
    id_image_schema_map = {k: v["annotation"]["image_schema_group"].strip().lower() for k, v in verb_class_res.items()}
    for _, v in id_image_schema_map.items():
        if v not in ['spatial motion', 'force', 'balance', 'other']:
            raise ValueError(f"invalid image schema result: {v}")
        
    # read in the possible group labels
    group_label_path = f"{env_vars['RESULTS_DIR']}/clean_target_groups/llm/{config.dataset_out_name}_target_group_cleanup.json"
    group_label_results = load_json(group_label_path)["data"]
    valid_group_labels = [list(v["annotation"]["clean_groups"].keys()) for v in group_label_results.values()]
    valid_group_labels = [item for sublist in valid_group_labels for item in sublist]
    valid_group_labels = [item.split(" (e.g.,")[0] for item in valid_group_labels]
    valid_group_labels.append("Other")

    if args.dataset == "podcasts":
        valid_group_labels.extend(["Audience", "Conversation Partner"])

    # for each metaphor, add a "features" dictionary where keys are feature name and value is feature value
    # load the ppt (noun_type) classification features
    ppt_res_path = f"{env_vars["RESULTS_DIR"]}/feature_extractor/llm/{config.dataset_out_name}_target_ppto_class_qwen.json"
    ppt_results = load_json(ppt_res_path)["data"]

    # get target group classification results
    group_results = {}
    for ppt_type in ["person", "place", "thing", "organization"]:
        cur_path = f"{env_vars["RESULTS_DIR"]}/feature_extractor/llm/{config.dataset_out_name}_target_{ppt_type}_class_qwen.json"
        cur_res = load_json(cur_path)["data"]
        group_results.update(cur_res)
    
    # check the noun classification results are valid, if they are get target group
    skipped_emb_count = 0
    target_groups = []
    for met_idx, met_data in tqdm(data.items(), desc="matching metaphors to llm-annotated features"):
        # use metaphor id to get llm annotation
        met_id = met_data["id"]
        ppt_ann = ppt_results[met_id]["annotation"]["noun_classification"].lower()
        if ppt_ann not in ["person", "place", "thing", "organization"]:
            LOGGER.warning(f"invalid ppt: '{ppt_ann}'")
            skipped_emb_count += 1
            continue
        
        try:
            group_ann = group_results[met_id]["annotation"]["target_group"]
        except:
            LOGGER.warning(f"{met_id} not in group_results")
            skipped_emb_count += 1
            continue

        # check that target group is valid
        group_ann = validate_group_label(group_ann, valid_group_labels, LOGGER)

        source_verb = id_source_map[met_id]
        source_verb_is_group = id_image_schema_map[source_verb]
        
        data[met_idx]["features"] = {
            "noun_classification": ppt_ann,
            "target_group": group_ann,
            "source_image_schema_group": source_verb_is_group
        }
        target_groups.append(group_ann)

    if skipped_emb_count > 0:
        LOGGER.warning(f"Skipped {skipped_emb_count} embeddings because of invalid info")


    out_path = f"{env_vars["RESULTS_DIR"]}/gen_embeddings/{args.embedding_model}/{config.dataset_out_name}_{args.embedding_name}_embeddings_features.pkl"
    LOGGER.info(f"Saving {len(data)} embeddings as pkl to {out_path}...")


    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    target_groups = list(set(target_groups))
    with open("temp.txt", "w") as t:
        t.write("\n".join(target_groups))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument("--dataset", required=True, help="dataset to cluster embeddings for", choices=['tweets_immigration', 'subframes_immigration', 'subframes_guns', 'subframes_abortion', 'lcc-large', 'podcasts'],)
    parser.add_argument("--labeled_only", action="store_true")
    parser.add_argument("--embedding_model", default="sbert", help="model used to generate embeddings, only sbert implemented for now")
    parser.add_argument("--embedding_name", required=True, help="name to distinguish the embeddings we're working with ([dataset_name]_[embedding_name]_embeddings.pkl")

    args = parser.parse_args()

    main(args)