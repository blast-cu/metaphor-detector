from dotenv import load_dotenv
import os
import logging
import json
import sys

import random
import numpy as np
import torch
from sqlalchemy import func, select
from src.db_config import Document
from typing import List
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # If using CUDA

def get_logger(name: str, level: int=logging.INFO) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to console handler
    ch.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(ch)
    
    return logger


def load_env(logger: logging.Logger=None):
    """
    Load the environment variables.
    """
    load_dotenv()
    vars = {
        "REPO_PATH": os.environ.get("REPO_PATH"),
        "RAW_DATA_DIR": os.environ.get("RAW_DATA_DIR"),
        "DB_DIR": os.environ.get("DB_DIR"),
        "INTERIM_DIR": os.environ.get("INTERIM_DIR"),
        "RESULTS_DIR": os.environ.get("RESULTS_DIR"),
        "RANDOM_SEED": int(os.environ.get("RANDOM_SEED", 42)),
        "DEVICE": os.environ.get("DEVICE", "cuda"),  # Default to 'cuda' if not set
    }

    if logger:
        logger.info("Loaded environment variables")

    return vars

def save_to_json(data: dict, json_path: str, json_name: str, indent: int=4, logger=None):
    # save the data to the json file.
    os.makedirs(json_path, exist_ok=True)
    out_path = os.path.join(json_path, json_name)
    try:
        with open(out_path, "w") as f:
            if indent is None:
                json.dump(data, f)
            else:
                json.dump(data, f, indent=indent)

        if logger is not None:
            logger.info(f"Saved data to: {out_path}")

    except Exception as e:
        raise IOError(f"Error saving data to {out_path}: {e}")

def load_json(file_path, logger=None):
    """
    Load the data from a file.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if logger: logger.info(f"Loaded data from: {file_path}")

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    return data

def sample_documents(session, config):
    if config.labeled_only:  # get list of only hand-labeled documents
        documents = session.query(Document).join(Document.doc_annotations).filter_by(annotation_type='human_metaphor_score').all()
        config.logger.info(f"Loaded {len(documents)} hand-labelled documents")

    else:
        if config.sample_size is not None:
            row_count = session.scalar(select(func.count()).select_from(Document))
            if config.sample_size > row_count:  # check that the desired sample size isn't too big
                config.logger.warning(f"Sample size {config.sample_size} is greater than the number of documents {row_count}. Using all documents.")
                config.sample_size = row_count
            documents = session.query(Document).order_by(func.random()).limit(config.sample_size).all()

        else:  # use complete corpus
            documents = session.query(Document).all()

        config.logger.info(f"Loaded {len(documents)} documents for processing")

    return documents

# spacy, shortest dependency path between possible nouns
METAPHOR_PATHS = {
    0: [("S_NOUN", "prep"), ("ADP", "pobj"), ("T_NOUN", "")],
    1: [("S_VERB", "dobj"), ("T_NOUN", "")],
    2: [("T_NOUN", "nsubj"), ("S_VERB", "")],
    3: [("T_NOUN", "compound"), ("S_NOUN", "")],
    4: [("S_ADJ", "amod"), ("T_NOUN", "")],
    5: [("T_NOUN", "nsubj"), ("AUX", "attr"), ("S_NOUN", "")],
    6: [("S_VERB", "agent"), ("ADP", "pobj"), ("T_NOUN", "")],
    7: [("S_VERB", "amod"), ("T_NOUN", "")],
    8: [("S_NOUN", "compound"), ("T_NOUN", "")],
    9: [("T_NOUN", "nsubjpass"), ("S_VERB", "")],
    10: [("T_NOUN", "nsubj"), ("AUX", "acomp"), ("S_ADJ", "")],
    11: [("T_NOUN", "poss"), ("S_NOUN", "")],
}
NOUNS = ["NOUN", "PROPN", "PRON"]
VERB_PATHS = [1, 2, 6, 7, 9]


def get_token(valid_met_path: int, sdp: List, logger:logging.Logger, target_word:bool=False):
    if target_word: sw = "T_"
    else: sw = "S_"
    
    for idx, met_token in enumerate(METAPHOR_PATHS[valid_met_path]):
        if met_token[0].startswith(sw):
            return sdp[idx]
    
    logger.error(f"No source token found in metaphor path {valid_met_path}")
    raise ValueError
