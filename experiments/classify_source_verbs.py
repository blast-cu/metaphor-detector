import argparse
import os
from tqdm import tqdm
import logging
from pydantic import BaseModel

from src.utils import load_env, get_logger, load_json
from src.data_loading import get_llm_metaphor_binary_df
from src.experiment_config import LLMExperimentConfig
from src.llm_ann.ollama_utils import Annotator


class ImageSchemaGroupAF(BaseModel):
    image_schema_group: str

def format_docs(docs, testing):
    formatted_docs = {}
    for verb, examples in tqdm(docs.items(), desc="formatting docs"):
        examples = examples[:3] # pick 3
        examples_formatted = [f"target noun: '{tn}'\nsentence: '{s}'" for tn, s in examples]
        formatted_docs[verb] = {
            "text": f"SOURCE VERB: {verb}\nEXAMPLES:\n {("\n").join(examples_formatted)}"
        }
        if testing and len(formatted_docs) == 10: # test on 10 docs
            break

    return formatted_docs


def load_source_verbs(input_path, testing):
    
    all_source_verbs = {}
    with os.scandir(input_path) as entries:
        for entry in tqdm(entries, desc="reading files to get all verbs"):
            if entry.is_file():
                
                # load the results into a pandas df
                df = get_llm_metaphor_binary_df(entry.path, logger)

                # filter df to just metaphorical instances
                df = df[df["llm_met_class"] == True]

                df["source_word"] = df["source_word"].apply(lambda x: x.lower().strip())
                source_verbs = df["source_word"].unique().tolist()
                for v in source_verbs:
                    # get max 3 examples for the source verb
                    sv_df = df[df["source_word"] == v]
                    example_sentence = sv_df["sentence"].tolist()[:3]
                    example_target_noun = sv_df["target_word"].tolist()[:3]
                    examples = list(zip(example_sentence, example_target_noun)) # list of tuples

                    if v not in all_source_verbs:
                        all_source_verbs[v] = examples
                    else:
                        all_source_verbs[v].extend(examples)

                if testing: # if testing, only process the first file
                    break

    return all_source_verbs


def main(args, env_vars, logger):
    
    # for all the json files in results/metaphor_classification/llm
    # read in the file and get the verb
    input_path = f"{env_vars["RESULTS_DIR"]}/metaphor_classification/llm"

    # create dict where the verb is the key
    all_source_verbs = load_source_verbs(input_path, args.testing)

    ## FORMAT TEXT FOR PROMPTING
    data_formatted = format_docs(all_source_verbs, args.testing)

    ## SET UP EXPERIMENT CONFIGURATION
    config = LLMExperimentConfig(
        dataset="all",
        labeled_only=False,
        task_type="classify_source_verbs",
        logger=logger,
        env_vars=env_vars,
        device=env_vars["DEVICE"],
        random_seed=env_vars["RANDOM_SEED"],
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
    config.answer_format = ImageSchemaGroupAF

    ### SET UP LLM ANNOTATOR
    annotator = Annotator(
        config=config,
        data_entries=data_formatted
    )
    annotator.process_docs()       


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use a trained metaphor model to measure the metaphoricity of a dataset.")
    parser.add_argument(
        "--sample_size",
        type=int,
        help="The number of documents from the dataset to annotate."
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
    logger = get_logger(f"process_source_verbs", level=logging.INFO)
    env_vars = load_env(logger=logger)

    args = parser.parse_args()
    main(args, env_vars, logger)