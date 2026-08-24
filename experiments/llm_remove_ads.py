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

class AdClassificationAF(BaseModel):
    classification: str


class RemoveAdsPromptEngineer:

    def __init__(self, config, data_loader: DatasetLoader, logger: logging.Logger, num_workers: int = 8, shuffle_docs: bool = False):

        self.config = config
        self.data_loader = data_loader
        self.prompt_file = config.prompt_file
        self.logger = logger
        self.num_workers = num_workers
        self.shuffle_docs = shuffle_docs

        self.metaphor_docs = None # load later

    def get_af(self):

        if self.prompt_file == "podcasts_ad_classification.json":
            return AdClassificationAF
        else: 
            raise ValueError(f"Invalid prompt file: {self.prompt_file}")
    
        
    def get_data(self) -> dict:
        # load results of ppt extraction
        metaphor_df = self.data_loader.load_metaphor_classification_results(metaphors_only=True, with_ads=True).set_index("id")
        metaphor_docs = metaphor_df.to_dict(orient='index')

        self.metaphor_docs = {
            k: {"text": f"Sentence: {v['sentence']}"} 
            for k, v in metaphor_docs.items()
        }

        if self.config.sample_size is not None:
            self.metaphor_docs = {k: v for k, v in list(self.metaphor_docs.items())[:self.config.sample_size]}

        return self.metaphor_docs
    

def main(args, env_vars, LOGGER):
    LOGGER.info("Starting script to run LLM annotator for classifying podcast ads...")

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
        task_type="remove_ads",
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
    prompt_engineer = RemoveAdsPromptEngineer(config, data_loader, LOGGER, shuffle_docs=args.shuffle_docs)
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
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     choices=['tweets_immigration', 'subframes_immigration', 'subframes_guns', 'subframes_abortion', 'lcc-large', 'podcasts'],
    #     help="The dataset to measure metaphoricity of. Currently only 'wildchat' is supported."
    # )
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


    ### THIS IS ONLY APPLICABLE FOR PODCAST DATASET
    args = parser.parse_args()
    args.dataset = "podcasts"

    main(args, env_vars, logger)
