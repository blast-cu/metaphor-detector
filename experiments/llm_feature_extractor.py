import argparse
import random
import logging
import torch # testing gpu availability
from tqdm import tqdm

from pydantic import BaseModel

from src.utils import load_env, get_logger, load_json
from src.experiment_config import LLMExperimentConfig
from src.llm_ann.ollama_utils import Annotator
from src.data_loading import DatasetLoader

"""
Prompting to determine metaphor score or binary classification for sdp 
candidates extracted via 'experiments.get_candidate_metaphor_paths'.
"""

class PPTClassAF(BaseModel):
    # {'noun_classification': '[person/place/thing]'}
    noun_classification: str

class TargetGroupFeaturesAF(BaseModel):
    target_group: str

class FEPromptEngineer:

    def __init__(self, config, data_loader: DatasetLoader, logger: logging.Logger, num_workers: int = 8, shuffle_docs: bool = False):

        self.config = config
        self.data_loader = data_loader
        self.prompt_file = config.prompt_file
        self.logger = logger
        self.num_workers = num_workers
        self.shuffle_docs = shuffle_docs

        self.metaphor_docs = None # load later

    def get_af(self):
        if self.prompt_file in ["person_place_thing_class.json", "person_place_thing_org_class.json"]:
            answer_format = PPTClassAF
        elif self.prompt_file in [
            "tweets_immigration_target_person_prompt.json", "subframes_guns_target_person_prompt.json", "subframes_abortion_target_person_prompt.json", "podcasts_target_person_prompt.json",
            "tweets_immigration_target_place_prompt.json", "subframes_guns_target_place_prompt.json", "subframes_abortion_target_place_prompt.json", "podcasts_target_place_prompt.json",
            "tweets_immigration_target_thing_prompt.json", "subframes_guns_target_thing_prompt.json", "subframes_abortion_target_thing_prompt.json", "podcasts_target_thing_prompt.json",
            "tweets_immigration_target_organization_prompt.json", "subframes_guns_target_organization_prompt.json", "subframes_abortion_target_organization_prompt.json", "podcasts_target_organization_prompt.json"
        ]:
            answer_format = TargetGroupFeaturesAF
        else:
            self.logger.error(f"No answer format specified for prompt file: {self.prompt_file}")
            raise ValueError

        return answer_format
    
    def format_ppt_docs(self) -> dict:
        # load raw metaphors
        self.metaphor_docs = {}
        metaphor_df = self.data_loader.load_metaphor_classification_results(metaphors_only=True)
        for _, row in tqdm(metaphor_df.iterrows(), desc="Formatting metaphorical docs for PPT classification"):
            self.metaphor_docs[row["id"]] = {
                "text": f"TARGET NOUN: {row["target_word"]} \nCONTEXT SENTENCE: {row["sentence"]}",
                "sentence": row["sentence"],
                "target_word": row["target_word"]
            }

        return self.metaphor_docs
    
    def format_final_feature_docs(self, noun_type) -> dict:
        # load results of ppt extraction
        ppt_res_path = f"{self.config.task_dir}/{self.config.dataset_out_name}_target_ppto_class_qwen.json"
        metaphor_docs = load_json(ppt_res_path)["data"]

        # first get the list of character groups
        target_group_path = f"results/clean_target_groups/llm/{self.config.dataset_out_name}_target_group_cleanup.json"
        target_groups = load_json(target_group_path)["data"][noun_type]["annotation"]["clean_groups"] # dictionary where keys are groups, value is definition
        target_groups_clean = [f"{k}: {v}" for k, v in target_groups.items()]
        
        # first filter to person noun classifications, add data needed for llm annotation
        filtered_metaphors = {}
        for k, v in tqdm(metaphor_docs.items(), desc="Formatting metaphorical docs for person feature extraction"):
            if v["annotation"]["noun_classification"].lower() == noun_type:
                del v["annotation"]
                v["text"] = f"TARGET GROUPS: {'\n'.join(target_groups_clean)} \nTARGET NOUN: {v["target_word"]} \nCONTEXT SENTENCE: {v["sentence"]}"
                
                filtered_metaphors[k] = v

        self.metaphor_docs = filtered_metaphors
        return self.metaphor_docs
    
    
    def get_data(self) -> dict:

        if self.prompt_file in ["person_place_thing_class.json", "person_place_thing_org_class.json"]:
            return self.format_ppt_docs()
        
        else: # already checked for validity when getting answer format
            noun_type = self.prompt_file.split("_")[-2]
            return self.format_final_feature_docs(noun_type)


def main(args, env_vars, LOGGER):
    LOGGER.info("Starting script to run LLM annotator...")

    # SET PATHS AND VARS
    RANDOM_SEED = env_vars["RANDOM_SEED"]
    DEVICE = env_vars["DEVICE"]
    random.seed(RANDOM_SEED)

    # CHECK IF GPU IS AVAILABLE
    if torch.cuda.is_available():
        LOGGER.info(f"GPU is available. Using device: {DEVICE}")
    else:
        LOGGER.info("GPU is NOT available. Using CPU.")

    ## SET UP EXPERIMENT CONFIGURATION
    config = LLMExperimentConfig(
        dataset=args.dataset,
        labeled_only=args.labeled_only,
        task_type="feature_extractor",
        logger=LOGGER,
        env_vars=env_vars,
        device=DEVICE,
        random_seed=RANDOM_SEED,
        prompt_file=args.prompt_file,
        out_file=args.out_file,
        model_config_file=args.annotation_config,
        host=args.host,
        port=args.port,
        testing=args.testing,
        sample_size=args.sample_size,
        shots=args.shots,
        skip_load=args.skip_load
    )
    
    data_loader = DatasetLoader(config)


    ### PREPROCESS TEXT FOR ANNOTATION
    prompt_engineer = FEPromptEngineer(config, data_loader, LOGGER, shuffle_docs=args.shuffle_docs)
    config.answer_format = prompt_engineer.get_af()
    data_formatted = prompt_engineer.get_data()

    config.logger.info(f"Prepared {len(data_formatted)} sentences for annotation")

    ### SET UP LLM ANNOTATOR
    annotator = Annotator(
        config=config,
        data_entries=data_formatted
    )
    annotator.process_docs()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Use a trained metaphor model to measure the metaphoricity of a dataset.")
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
        "--prompt_file", type=str,
        help="The name of the json file in src/llm_ann/prompts/ to be used.",
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

    ### SET UP ENVIRONMENT VARIABLES AND LOGGER
    logger = get_logger(f"llm_get_features", level=logging.INFO)
    env_vars = load_env(logger=logger)

    args = parser.parse_args()
    main(args, env_vars, logger)