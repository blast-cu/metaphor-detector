import argparse
from pydantic import BaseModel
from typing import Union

from src.utils import load_env, get_logger, load_json, set_seed
from src.experiment_config import ExperimentConfig, LLMExperimentConfig
from src.llm_ann.ollama_utils import Annotator

class TargetGroupCleanup(BaseModel):
    clean_groups: dict[str, str]


def main(args):

    logger = get_logger("clean_target_groups")
    env_vars = load_env()
    set_seed(env_vars["RANDOM_SEED"])

    config = ExperimentConfig(
        task_type="create_target_groups",
        task_name="",
        dataset=args.dataset,
        logger=logger,
        env_vars=env_vars,
        labeled_only=args.labeled_only,
        device=env_vars["DEVICE"],
        random_seed=env_vars["RANDOM_SEED"],
        skip_load=True # do not check preprocessing on db, use as is (saves time)
    )

    # load the llm label results
    llm_label_res_path = f"{env_vars["RESULTS_DIR"]}/create_target_groups/llm/{config.dataset_out_name}_target_group_creation.json"
    llm_label_res = load_json(llm_label_res_path)["data"]


    # divide metaphors into person/place/thing based on llm classification results
    data_formatted = {}
    for ppt_class in ["person", "place", "thing", "organization"]:

        # gather all the labels for the ppt class
        group_labels = []
        for cluster_key, res in llm_label_res.items():
            if ppt_class in cluster_key:
                group_labels.extend(res["annotation"]["groups"])
        
        # construct the document
        data_formatted[ppt_class] = {
            "text": f"group labels: {"\n".join(group_labels)}"
        }

    # set up annotator
    prompt_file = "clean_target_groups.json"

    # set up llm config
    config = LLMExperimentConfig(
        dataset=args.dataset,
        labeled_only=args.labeled_only,
        task_type="clean_target_groups",
        logger=logger,
        env_vars=env_vars,
        device=env_vars["DEVICE"],
        random_seed=env_vars["RANDOM_SEED"],
        prompt_file=prompt_file,
        out_file=args.out_file,
        model_config_file=args.annotation_config,
        host=args.host,
        port=args.port,
        testing=args.testing,
        sample_size=args.sample_size,
        shots=args.shots,
        skip_load=args.skip_load,
    )
    config.answer_format = TargetGroupCleanup

    ### SET UP LLM ANNOTATOR
    annotator = Annotator(
        config=config,
        data_entries=data_formatted
    )
    annotator.process_docs()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create target groups')
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['tweets_immigration', 'subframes_immigration', 'subframes_guns', 'subframes_abortion', 'lcc-large', 'podcasts'],
        help="The dataset to measure metaphoricity of. Currently only 'wildchat' is supported."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="The number of documents from the dataset to annotate."
    )
    parser.add_argument(
        "--labeled_only",
        action='store_true',
        help="If set, only use documents that have human metaphor annotations (only for tweets_immigration dataset)."
    )
    parser.add_argument(
        "--annotation_config", type=str, default="default.yaml",
        help="Name of config file in src/llm_ann/configs directory"
    )
    parser.add_argument(
        "--out_file", type=str, default=None,
        help="The desired name of the output file."
    )
    parser.add_argument(
        '--host', metavar='HOST', required=True,
        help="The host for the Ollama API."
    )
    parser.add_argument(
        '--port', metavar='PORT',
        help="The port for the Ollama API."
    )
    parser.add_argument(
        "--testing",
        action='store_true',
        help="If set, the script will run in testing mode with a small sample of the data."
    )
    parser.add_argument(
        "--skip_load",
        action='store_true',
        help="If set, won't check dataset before loading."
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="The number of examples to add to the prompt. By default, just uses all examples included in the prompt json. "
    )
    parser.add_argument(
        "--shuffle_docs",
        action='store_true',
        help="If set, shuffles docs before processing."
    )

    args = parser.parse_args()
    main(args)
