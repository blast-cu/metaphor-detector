from tqdm import tqdm
import time
import os
import random
from typing import Iterable

import concurrent
import ollama
from pydantic import BaseModel
from src.experiment_config import LLMExperimentConfig

from src.utils import get_logger, save_to_json, load_json

"""
Utility functions for annotating data with an LLM.
Includes functions for loading models, generating text, and formatting data.
"""

class OllamaClient:
    """
    Class to handle the Ollama client.
    """
    class DefaultAnswer(BaseModel):
        """
        Class to format the response from the LLM.
        """
        response: str
    
    class Messages():
        """
        Class to handle the messages for the LLM.
        """
        def __init__(self, system_prompt: str, user_head_prompt: str):
            self.system_prompt = system_prompt
            self.head_user_prompt = user_head_prompt
        
        def add_doc_prompt(self, doc_prompt: str):
            to_process = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.head_user_prompt + "\n\n" + doc_prompt}
            ]
            return to_process
    

    def __init__(self, host, port, model, seed, temperature, system_prompt, user_head_prompt, answer_format_class, logger):
        """
        Initialize the Ollama client.
        """
        self.logger = logger
        server_host = f"{host}:{port}"
        self.client = ollama.Client(server_host)

        self.model = model
        self.seed = seed
        self.temperature = temperature

        ollama.pull(model)

        self.options: ollama.Options = {
            "seed": seed,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 3
        }

        # set the messages for client.
        self.messages = self.Messages(system_prompt, user_head_prompt)

        # customize the Answer class here if needed.
        if answer_format_class is None:
            self.answer_format_class = self.DefaultAnswer
        else:
            self.answer_format_class = answer_format_class

    def print_logprobs(self, logprobs: Iterable[dict], label: str) -> None:
        self.logger.info(f'\n{label}:')
        for entry in logprobs:
            token = entry.get('token', '')
            logprob = entry.get('logprob')
            self.logger.info(f'  token={token!r:<12} logprob={logprob:.3f}')
            for alt in entry.get('top_logprobs', []):
                if alt['token'] != token:
                    self.logger.info(f'    alt -> {alt["token"]!r:<12} ({alt["logprob"]:.3f})')

    def chat(self, doc_prompt: str):
        """
        Chat with the LLM and check the response.
        """
        to_process = self.messages.add_doc_prompt(doc_prompt)
        
        response = self.client.chat(
            self.model,
            messages=to_process,
            options=self.options,
            format=self.answer_format_class.model_json_schema()
        )
        
        try:

            # self.print_logprobs(response.get('logprobs', []), 'chat logprobs')
            # self.logger.info(response.get('logprobs', []))
            # print('Chat response:', response['message']['content'])
            # print(response['logprobs'])
            
            response = self.answer_format_class.model_validate_json(
                response.message.content
            )

            if response is not None:  # make the output serializable.
                response = response.model_dump()
            return response

        except Exception as e:
            self.logger.exception("Exception: " + str(e))
            self.logger.exception("Invalid response. Please try again.")
            return None


class Annotator:
    """
    Class for annotating text with LLM output.
    """

    def __init__(
            self, config: LLMExperimentConfig, data_entries: dict
    ):
        self.config = config
        if config.logger is None:
            self.logger = get_logger()
        else:
            self.logger = config.logger

        self.logger.info("Setting Annotator variables and initializing Ollama client.")
        self.docs = data_entries

        # load the prompt data.
        self.prompt_data = self.load_prompt(config.repo_dir)

        # check for existing results.
        self.already_processed, self.docs = \
            self.handle_processed()
        
        # set the head messages for the conversations.
        system_prompt = self.prompt_data["system_prompt"]
        user_head_prompt = self.get_user_head_prompt()
        self.logger.info(user_head_prompt)
        
        # initialize the ollama client.
        self.ollama_client = OllamaClient(
            config.host,
            config.port,
            config.model,
            config.random_seed,
            config.temperature,
            system_prompt,
            user_head_prompt,
            config.answer_format,
            self.logger,
        )

    def load_prompt(self, repo_path: str) -> dict:
        """
        Load the prompt data and dataset from json.
        """
        prompt_path = os.path.join(repo_path, 'src/llm_ann/prompts', self.config.prompt_file)
        prompt_data = load_json(
            prompt_path, logger=self.logger
        )
        return prompt_data

    def handle_processed(self) -> tuple[dict, dict]:
        """
        Load docs that have already been processed and 
        remove them from the list of docs to process.
        """
        already_processed = {}
        to_process = self.docs

        out_dir = self.config.task_dir
        if self.config.out_file:  # if custom output file is set use that, else use default
            filename = self.config.out_file
        else:
            filename = self.config.out_path.split("/")[-1]
        self.out_path = os.path.join(out_dir, filename)

        if os.path.exists(self.out_path):
            existing_data = load_json(self.out_path, logger=self.logger)
            already_processed = existing_data["data"] if "data" in existing_data else {}
            self.logger.info(f"Found {len(already_processed)} existing results.")

            # remove already processed docs from the list
            to_process = {doc_id: doc for doc_id, doc in self.docs.items()
                          if doc_id not in already_processed.keys()}
        else:
            self.logger.info(f"No previously processed data found at {self.out_path}, starting from scratch...")

        return already_processed, to_process

    def get_user_head_prompt(self) -> str:
        """
        Set the head messages for the conversations, implement n shot prompting.
        """
        shots = []
        head_user_prompt = ""
        if "demos" in self.prompt_data.keys() and len(self.prompt_data["demos"]) > 0:
            demos = self.prompt_data["demos"]
            
            if self.config.shots == 0: # force 0 shot prompting
                return head_user_prompt
            
            elif self.config.shots < len(demos):
                self.logger.info(f"Specified fewer shots than number of demos in prompt file. Randomly selecting {self.config.shot} from {len(demos)}")
                demos = random.sample(demos, self.config.shots)

            elif self.config.shots > len(demos):
                self.logger.warning(f"Specified {self.config.shots} shots, but only {len(demos)} examples in prompt file. Using everything we've got")
                self.config.shots = len(self.prompt_data["demos"])

            for demo_item in demos:
                user_turn = "user: " + self.prompt_data["question"] + "\n" + demo_item["text"]
                assissant_turn = "assistant: " + str(demo_item["answer"])
                shots.append(user_turn + "\n" + assissant_turn)

            head_user_prompt = "\n\n".join(shots)

        return head_user_prompt

    def annotate(self, doc_prompt: str) -> dict:
        """
        Annotate the messages with the LLM. Try multiple times if the response is invalid.
        """
        max_retries = self.config.max_retries
        retry_count = 0
        while retry_count < max_retries:
            try:
                annotation = self.ollama_client.chat(
                    doc_prompt
                )
                if annotation is not None:
                    return annotation
                else:  # retry if the response did not match the schema.
                    retry_count += 1

            except Exception as e:
                self.logger.exception("Exception: " + str(e))
                self.logger.exception("Ollama Error. Please try again.")
                retry_count += 1
        return None
    
    def process_doc(self, doc: dict) -> dict:
        """
        Process the doc by prompting the LLM.
        """
        doc_prompt = self.prompt_data["question"] + '\n' + doc["text"]
        annotation = self.annotate(doc_prompt)
        doc["annotation"] = annotation
        return doc
    
    def save_results(self, processed_data: dict):
        """
        Save the results and config details to a json file.
        """
        final_output = {}
        final_output["config"] = self.config.to_dict()
        final_output["time_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        final_output["prompt_data"] = self.prompt_data
        final_output["data"] = processed_data

        out_dir = self.out_path.rsplit("/", 1)[0]
        filename = self.out_path.rsplit("/", 1)[1]

        save_to_json(
            data=final_output, json_path=out_dir,
            json_name=filename, logger=None
        )

    def process_docs(self):
        """
        Process the docs in parallel.
        """
        num_workers = self.config.workers
        save_interval = self.config.save_interval

        data = self.docs  # use unformatted docs for now.

        annotated_docs = self.already_processed
        total_docs = len(data)

        # Use a ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process_doc, doc): doc_id for doc_id, doc in data.items()}

            # Initialize tqdm progress bar to track doc processing
            with tqdm(total=total_docs, desc=f"Prompting LLM with {num_workers} workers") as pbar:
                processed_count = 0
                # Process docs and save at regular intervals
                for future in concurrent.futures.as_completed(futures):
                    doc_idx = futures[future]
                    try:
                        # Get the results of process_doc() for each doc
                        doc = future.result()
                        annotated_docs[doc_idx] = doc
                        processed_count += 1

                        # Update the progress bar
                        pbar.update(1)

                        # Save annotated_docs at regular intervals
                        if processed_count % save_interval == 0:
                            self.save_results(annotated_docs)

                    except Exception as e:
                        self.logger.exception(f"Error processing doc {doc_idx}: {e}")

        # Save the final results.
        self.save_results(annotated_docs)