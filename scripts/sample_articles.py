import argparse
from src.utils import load_env, load_json, get_logger, save_to_json
from src.experiment_config import LLMExperimentConfig
from src.data_loading import DatasetLoader

def main(args):
    env_vars = load_env()
    logger = get_logger("check_datasets")

    CANDIDATE_PATH = f"{env_vars["RESULTS_DIR"]}/get_candidate_metaphors/rule_based"
    MET_CLASS_PATH = f"{env_vars["RESULTS_DIR"]}/metaphor_classification/llm"

    # read in the article data 
    met_classifications_filename = f"{args.dataset}_source_verb_target_noun_binary_met_class_qwen.json"
    met_classifications = load_json(f"{MET_CLASS_PATH}/{met_classifications_filename}")
    met_class_config = met_classifications["config"]
    met_classifications = met_classifications["data"]

    # mets = [c_id for c_id, v in met_classifications.items() if v["annotation"]["classification"] == "Metaphorical"]
    # print(f"{len(mets)} metaphors")

    # class_doc_ids = [c_id.split("_")[0] for c_id in met_classifications.keys()]
    # class_doc_ids = list(set(class_doc_ids))

    # get a dict where the keys are article ids and items are the metaphor dict
    article_dict = {}
    for met_id, met_info in met_classifications.items():
        art_id = met_id.split("_")[0]
        if art_id not in article_dict:
            article_dict[art_id] = {}
        
        if met_info["annotation"]["classification"].lower() == "metaphorical":
            article_dict[art_id][met_id] = met_info


    config = LLMExperimentConfig.from_dict(met_class_config, logger, env_vars)
    data_loader = DatasetLoader(config)

    data = data_loader.load_raw_data(processed_only=True)
    assert len(data) == len(article_dict)

    # check that the dataset is balanced
    if args.dataset in ["subframes_guns", "subframes_abortion", "tweets_immigration"]:
        sampled_data = data.groupby(by="polarity").head(args.num_docs/2)
        print(sampled_data.groupby(by="polarity").count())
        sampled_article_ids = sampled_data["id"].tolist()
    
    else:
        raise ValueError(f"Sampling for {args.dataset} not implemented...")

    sampled_metaphor_dict = {}
    met_count = 0
    for article_id in sampled_article_ids:
        sampled_metaphor_dict[article_id] = article_dict[article_id]
        met_count += len(article_dict[article_id])

    logger.info(f"Number of sampled metaphors: {met_count}")

    if args.save:
        # filter the metaphor classification results, format dict to make output json
        sampled_met_class_path =  f"art_sampled_results_{args.num_docs}_articles/metaphor_classification/llm"
        filtered_met_classifications = {}
        for met_id, met_info in met_classifications.items():
            cur_art_id = met_id.split("_")[0]
            if cur_art_id in sampled_article_ids:
                filtered_met_classifications[met_id] = met_info

        met_classifications_out = {
            "config": met_class_config,
            "data": filtered_met_classifications
        }

        # write to json
        save_to_json(met_classifications_out, sampled_met_class_path, met_classifications_filename)

        # FILTER FEATURE DATA
        feature_data_path = f"{env_vars["RESULTS_DIR"]}/feature_extractor/llm"
        sampled_feature_data_path = f"art_sampled_results/feature_extractor/llm"

        ppt_res_filename = f"{config.dataset_out_name}_target_ppt_class_qwen.json"
        ppt_results = load_json(f"{feature_data_path}/{ppt_res_filename}")
        ppt_results_config = ppt_results["config"]
        ppt_results = ppt_results["data"]

        filtered_ppt_results = {}
        for met_id, met_info in ppt_results.items():
            cur_art_id = met_id.split("_")[0]
            if cur_art_id in sampled_article_ids:
                filtered_ppt_results[met_id] = met_info
        
        ppt_res_out = {
            "config": ppt_results_config,
            "data": filtered_ppt_results
        }

        # write to json
        save_to_json(ppt_res_out, sampled_feature_data_path, ppt_res_filename)

        for noun_class in ["person", "place", "thing"]:
            # filter
            nc_filename = f"{config.dataset_out_name}_target_{noun_class}_class_qwen.json"
            nc_res = load_json(f"{feature_data_path}/{nc_filename}")
            nc_config = nc_res["config"]
            nc_res = nc_res["data"]
            filtered_nc_results = {}
            for met_id, met_info in nc_res.items():
                cur_art_id = met_id.split("_")[0]
                if cur_art_id in sampled_article_ids:
                    filtered_nc_results[met_id] = met_info 

            nc_res_out = {
                "config": nc_config,
                "data": filtered_nc_results
            }
            save_to_json(nc_res_out, sampled_feature_data_path, nc_filename)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a dataset using Kmeans clustering."
    )
    parser.add_argument(
        "--dataset", type=str,
        required=True,
        help="Dataset we're exporting metaphors for.",
    )
    parser.add_argument(
        "--num_docs", type=int,
        required=True,
        help="number of documents to sample"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="whether or not to overwrite the old files with the sampled data"
    )
    args = parser.parse_args()
    main(args)