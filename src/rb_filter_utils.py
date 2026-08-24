import logging
from tqdm import tqdm
import os
import numpy as np

import concurrent
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

import stanza # use same tokenization
import spacy
import spacy.tokens as st
import networkx as nx
import itertools

from transformers import RobertaTokenizerFast, \
    TrainingArguments, Trainer
from datasets import Dataset

from src.db_config import Sentence, Token, Document, DocAnnotation
from typing import List
from src.span_classifier import prepare_dataset, \
    RobertaSpanForSequenceClassification, DataCollatorForSpanClassification
from src.utils import METAPHOR_PATHS, VERB_PATHS, NOUNS, save_to_json, load_json


class MetaphorFilter():

    def __init__(self, config, verbs_only: bool, ms_model_path: str | None, logger: logging.Logger):
        self.logger = logger
        self.config = config
        self.nlp = spacy.load("en_core_web_sm")

        self.stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=True)
        self.nlp.tokenizer = self.stanza_tokenize

        if (type(ms_model_path)) is str and (not os.path.isdir(ms_model_path)):
            logger.error(f"Given metaphor score model path doesn't exist: {ms_model_path}")
            raise ValueError

        self.ms_model_path = ms_model_path
        self.verbs_only = verbs_only
        self.valid_met_paths = METAPHOR_PATHS
        if self.verbs_only:
            self.valid_met_paths = {k: v for k, v in METAPHOR_PATHS.items() if k in VERB_PATHS}

    def stanza_tokenize(self, sentence):
        """
        creates spacy Doc with stanza tokenization
        """
        processed_doc = self.stanza_nlp(sentence)
        sentence = processed_doc.sentences[0]
        words = [w.text for w in sentence.words]
        spaces = [True] * (len(words) - 1) + [False]
        
        return st.Doc(self.nlp.vocab, words=words, spaces=spaces)
    

    def get_candidates_from_doc(self, session: Session, doc: Document) -> List[Sentence]:
        """
        Get candidate sentences from a single doc
        """
        doc_id = doc.id
        cur_doc_candidates = []

        # get doc sentences
        sentence_counter = 0
        found_doc_end = False
        while not found_doc_end:
            sent_id = f"{doc_id}_{sentence_counter}"
            sent = session.query(Sentence).filter_by(id=sent_id).one_or_none()
            if sent is None:
                found_doc_end = True
            else:
                # get tokens, we need at least two nouns
                token_counter, found_sent_end = 0, False
                sent.valid_tokens = []
                found_verb, found_noun = False, False
                while not found_sent_end:
                    token_id = f"{doc_id}_{sentence_counter}_{token_counter}"
                    token = session.query(Token).filter_by(id=token_id).one_or_none()
                    # sent.token_text.append(token.text)
                    if token is None:
                        found_sent_end = True
                    elif token.pos in ["PROPN", "NOUN", "PRON", "VERB"]:
                        if token.pos == "VERB":
                            found_verb = True
                        else:
                            found_noun = True
                        sent.valid_tokens.append((token.id, token.text))

                    token_counter += 1

                if found_verb and found_noun: # need both a verb and a noun to make a met
                    cur_doc_candidates.append(sent)

                sentence_counter += 1

        # for s in cur_doc_candidates:
        #     for t in s.valid_tokens:
        #         if t[0] is None or t[1] is None:
        #             print(t)

        return cur_doc_candidates

    def get_candidate_sentences(self, documents: Document) -> list[Sentence]:
        """
        Parallelized: for each document, iterate over sentences, if they have 
        2 or more tokens with pos of interest (verb, noun)
        """
        engine = create_engine(f"sqlite:///{self.config.db_path}", echo=False)
        SessionLocal = sessionmaker(bind=engine)

        def call_doc_processor(doc): # need new session for each call
            with SessionLocal() as session:
                return self.get_candidates_from_doc(session, doc)
        
        candidate_sentences = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [executor.submit(call_doc_processor, doc) for doc in documents]
            
            with tqdm(total=len(documents), desc="Finding sentences with 2 or more tokens with pos of interest") as pbar:
                processed_count = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        doc_candidates = future.result()
                        candidate_sentences.extend(doc_candidates)
                        processed_count += 1
                        pbar.update(1) # Update the progress bar

                    except Exception as e:
                        self.logger.exception(f"Error processing doc: {e}")

        return candidate_sentences


    def get_sentence_sdps(self, sentence: Sentence) -> List[dict]:
        """
        get shortest dependency path between start and end word, returns list of spacy tokens
        """
        sent_id = sentence.id
        def get_sdps(graph: nx.Graph, spacy_parsing: dict[str, st.Token], start_word: tuple, end_word: tuple):
            
            start_id, start_text = start_word
            end_id, end_text = end_word

            if start_text is None or end_text is None:
                self.logger.error(f"Start or end token text is None - Start text: '{start_text}', End text: '{end_text}'")
                return "Error: start or end token text is None"

            # Create a directed graph from the edges
            # find the specific tokens in the doc for the start and end words
            spacy_start_node = next((t for t in spacy_parsing.values() if f"{sent_id}_{t.i}" == start_id and t.lower_ == start_text.lower()), None)
            spacy_end_node = next((t for t in spacy_parsing.values() if f"{sent_id}_{t.i}" == end_id and t.lower_ == end_text.lower()), None)
            
            if not spacy_start_node or not spacy_end_node:
                spacy_tokens = [t for t in spacy_parsing.values()]
                self.logger.debug(f"Start or end token ({start_text}, {end_text}) not found in spacy-processed sentence: \n{spacy_tokens}")
                return "Error: no start or end token found"

            try: # Find the shortest path in the graph
                path = nx.shortest_path(graph, source=start_id, target=end_id)
                clean_path = [] # gather path, include edge dep types
                for idx, p_id in enumerate(path):
                    spacy_tok = spacy_parsing[p_id]
                    token_dict = {"db_id": p_id, "spacy_tok": spacy_tok}
                    
                    dep_to_next = ""
                    if idx < len(path) - 1:
                        next_p_id = path[idx + 1]
                        dep_to_next = graph.get_edge_data(p_id, next_p_id)["dep"]

                    token_dict["dep_to_next"] = dep_to_next
                    
                    clean_path.append(token_dict)

                return clean_path

            except nx.NetworkXNoPath as e:
                spacy_tokens = [t for t in spacy_sentence]
                self.logger.debug(f"No path between ({start_text}, {end_text}) found in spacy-processed sentence: \n{spacy_tokens}\n")
                return "Error: no path found" # this can be due to bad grammer

            except nx.NodeNotFound as e:
                spacy_tokens = [t for t in spacy_sentence]
                # self.logger.error(f"One or both of the specified words were not found ({start_text}, {end_text}) in graph for sentence: \n{spacy_tokens}\n")
                return "Error: tokens not found in graph"
        
        sdps = []
        spacy_sentence = self.nlp(sentence.text)
        # Build a list of edges for the graph
        spacy_parsing = {f"{sent_id}_{token.i}": token for token in spacy_sentence}
        graph = nx.Graph()
        for token_id, token in spacy_parsing.items():
            # check that token ids match
            if token_id != f"{sent_id}_{token.i}":
                self.logger.error(f"Token ID mismatch: {token_id} vs f{sent_id}_{token.i}")
                raise ValueError("Token ID mismatch detected.")

            for child in token.children:
                graph.add_edge(token_id, f"{sent_id}_{child.i}", dep=child.dep_)

        pos_list = sentence.valid_tokens
        combos = list(itertools.combinations(pos_list, 2))
        for combo in combos:
            sdp = get_sdps(graph, spacy_parsing, combo[0], combo[1])
            sdps.append(sdp)

        return sdps
    
    def get_sdps(self, candidate_sentences):
        sdps = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [executor.submit(self.get_sentence_sdps, sentence) for sentence in candidate_sentences]

            # Initialize tqdm progress bar to track doc processing
            with tqdm(total=len(candidate_sentences), desc=f"Processing docs with {self.config.num_workers} workers") as pbar:
                processed_count = 0
                for future in concurrent.futures.as_completed(futures):
                    try: # Get the results of process_doc() for each doc
                        sdp = future.result()
                        sdps.extend(sdp)
                        processed_count += 1
                        pbar.update(1) # Update the progress bar

                    except Exception as e:
                            self.logger.exception(f"Error processing combo: {e}")
        # flatten sdps
        return sdps
    
    def clean_sdps(self, sdps: list):
        clean_sdps = []
        error_counts = {"no start or end token found": 0, "no path found": 0, "tokens not found in graph": 0}
        for sdp in sdps:
            if type(sdp) is str and sdp.startswith("Error: "):
                error_type = sdp.replace("Error: ", "")
                error_counts[error_type] += 1
            
            else:
                clean_sdps.append(sdp)

        if any(v > 0 for v in error_counts.values()):
            self.logger.warning(f"Errors found in sdp discovery: {error_counts}")

        return clean_sdps

    def check_path(self, sdp: list):
        matching_constr = []

        for path_id, path_constr in self.valid_met_paths.items():
            # check sdp against this particular path
            if len(sdp) != len(path_constr):
                continue # not a match

            # otherwise, check each token
            match = True
            for token_idx, (valid_token_type, valid_dep_to_next) in enumerate(path_constr):

                # get from current sdp,
                # figure out if we're checking pos or dep
                sdp_token = sdp[token_idx]

                # what is valid POS based on the valid token type
                clean_token_type = valid_token_type
                if valid_token_type.startswith("S_") or valid_token_type.startswith("T_"):
                    to_remove = clean_token_type.split("_")[0]
                    clean_token_type = clean_token_type.replace(to_remove + "_", "")
                
                if clean_token_type == "NOUN":
                    valid_pos = NOUNS
                else:
                    valid_pos = [clean_token_type]

                # self.logger.debug(f"Checking if '{sdp_attr}' in '{valid_token_type}': {valid_pos}")
                if not sdp_token["spacy_tok"].pos_ in valid_pos or \
                    not sdp_token["dep_to_next"] == valid_dep_to_next:

                    self.logger.debug(f"Token '{sdp_token['spacy_tok'].text}' with POS '{sdp_token['spacy_tok'].pos_}' and dep_to_next '{sdp_token['dep_to_next']}' does not match constraint ({valid_token_type}, {valid_dep_to_next})")
                    match = False
                    break
                # otherwise it's a match, keep trying

            # if after all that it's true, we have a match!
            # save which valid metaphor path was matched
            if match:
                matching_constr.append(path_id)

        return matching_constr # all of the indices of matching paths

    def check_metaphor_paths(self, sdps: list[list]):
        metaphor_paths = []
        for sdp in tqdm(sdps, desc="Checking whether sdps match metaphor paths."): # collect all that match a predefined met path
            path_matches = self.check_path(sdp)
            if len(path_matches) > 0:
                export_sdp = {
                    "sdp": [{
                        "token_id": token_dict["db_id"], 
                        "text": token_dict["spacy_tok"].text, 
                        "pos": token_dict["spacy_tok"].pos_, 
                        "dep": token_dict["spacy_tok"].dep_
                    } for token_dict in sdp],
                    "met_paths": path_matches
                }
                metaphor_paths.append(export_sdp)

        return metaphor_paths
    
    def get_token_indices(self, sentence: Sentence, token: Token):
        start_char = token.start_char - sentence.start_char
        end_char = token.end_char - sentence.start_char
        source_spans = [start_char, end_char]
        
        # ensure the text matches
        assert sentence.text[source_spans[0]:source_spans[1]] == token.text, f"Source span text mismatch: ({source_spans[0]}, {source_spans[1]}) '{sentence.text[source_spans[0]:source_spans[1]]}' != '{token.text}' \n(sentence text: {sentence.text})"
        return source_spans
    
    def get_sdp_tokens(self, valid_met_path: int, sdp: List, return_id: bool=True):
        """
        return source and target tokens for sdp
        """
        source_token, target_token = None, None
        for idx, met_token in enumerate(METAPHOR_PATHS[valid_met_path]):
            if met_token[0].startswith("S_"):
                source_token = sdp[idx]
            
            elif met_token[0].startswith("T_"):
                target_token = sdp[idx]

        if source_token is None or target_token is None:
            self.logger.error(f"No source or target token found in metaphor path {valid_met_path}")
            raise ValueError
        
        if return_id:
            source_token = source_token["token_id"]
            target_token = target_token["token_id"]

        return (source_token, target_token)
        
    def get_target_source_pairs(self, valid_sdps: List, session):

        s_t_pairs = {}
        for valid_sdp in tqdm(valid_sdps, desc="Consolidating to unique (source, target) pairs."):
            for met_path in valid_sdp["met_paths"]:
                source_token, target_token = self.get_sdp_tokens(met_path, valid_sdp["sdp"], return_id=False)

                # format the id
                target_token_idx = target_token["token_id"].split("_")[-1]
                pair_id = f"{source_token['token_id']}_{target_token_idx}"

                if pair_id not in s_t_pairs:
                    sentence_id = (source_token["token_id"].rsplit('_', 1))[0]
                    sentence = session.get(Sentence, sentence_id)
                    s_t_pairs[pair_id] = {
                        "source_token": source_token,
                        "target_token": target_token,
                        "sentence_text": sentence.text,
                        "met_paths": []
                    }
                s_t_pairs[pair_id]["met_paths"].append(met_path)

        return s_t_pairs
    
    def filter_lcc_data(self, session, s_t_pairs: dict):
        filtered_pairs = {}
        def get_source_str(doc_id):
            source_str = session.query(DocAnnotation).filter_by(id=f"{doc_id}_source_string").one_or_none().annotation_value
            return source_str.lower().strip()
        
        def get_target_str(doc_id):
            target_str = session.query(DocAnnotation).filter_by(id=f"{doc_id}_target_string").one_or_none().annotation_value
            return target_str.lower().strip()
        
        for pair_id, pair_info in s_t_pairs.items():
        
            # filter down to sdps with matches
            doc_id = pair_id.split("_")[0]
            doc_source_str = get_source_str(doc_id)
            doc_target_str = get_target_str(doc_id)

            clean_sw = pair_info["source_token"]["text"].lower().strip()
            clean_tw = pair_info["target_token"]["text"].lower().strip()

            if doc_source_str == clean_sw and doc_target_str == clean_tw:
                filtered_pairs[pair_id] = pair_info

        return filtered_pairs
            

    def get_scores(self, session, valid_paths: List):
        """
        loop over each path, get a score using the model for each possible score
        """
        to_predict = [] # prepare data
        for path in tqdm(valid_paths, desc="formatting paths for metaphor score prediction"):
            # get sentence id by removing token index from token id
            sentence_id = (path["sdp"][0]["token_id"].rsplit('_', 1))[0]
            
            f_tok_id = path["sdp"][0]["token_id"]
            l_tok_idx = path["sdp"][-1]["token_id"].split('_')[-1]
            path_id = f"{f_tok_id}_{l_tok_idx}"

            sentence = session.get(Sentence, sentence_id)
            sentence_text = sentence.text
            for met_path in path["met_paths"]:

                source_token_id, _ = self.get_sdp_tokens(met_path, path["sdp"]) # get this based on the matching path: met_path
                source_token = session.get(Token, source_token_id)
                if source_token.text == '<UNK>': 
                    continue

                source_indices = self.get_token_indices(sentence, source_token)

                sample_id = f"{path_id}_{met_path}"
                to_predict.append({
                    "id": sample_id,
                    "text": sentence_text,
                    "span_token_indices": source_indices,
                    "source_text": source_token.text
                })

            # add sentence text for easier analysis
            path["sentence_text"] = sentence.text
        

        # tokenize texts
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
        predict_ds = Dataset.from_list(to_predict)
        tokenized = prepare_dataset(predict_ds, tokenizer, text_col="text", label_col="label", index_col="span_token_indices", max_length=512)

        # load model
        model = RobertaSpanForSequenceClassification.from_pretrained(self.ms_model_path)
        collator = DataCollatorForSpanClassification(tokenizer)
        predictor = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=collator,
            args=TrainingArguments(remove_unused_columns=False)
        )

        # get predictions
        res = predictor.predict(tokenized)
        preds = np.argmax(res.predictions, axis=-1).tolist()

        # match labels to 'to_predict'
        final_preds = {to_predict[idx]["id"]: pred for idx, pred in enumerate(preds)}
        
        # now add them into the og dataset and return
        for path in valid_paths:
            # get path id
            f_tok_id = path["sdp"][0]["token_id"]
            l_tok_idx = path["sdp"][-1]["token_id"].split('_')[-1]
            path_id = f"{f_tok_id}_{l_tok_idx}"
            
            new_met_paths = []
            for met_path in path["met_paths"]:
                sample_id = f"{path_id}_{met_path}"

                try:
                    met_score = final_preds[sample_id]
                except KeyError as e:
                    self.logger.error(f"Could not retrieve metaphor score for sample '{sample_id}'")

                new_met_paths.append({
                    "path": met_path,
                    "metaphor_score": met_score
                })
            path["met_paths"] = new_met_paths
        
        return valid_paths