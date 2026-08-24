import os
from tqdm import tqdm
import pandas as pd
import pickle
import re
import string
from typing import List
import numpy as np
from pathlib import Path


import stanza  # for NLP processing
import torch  # for clearing GPU cache
import xmltodict

# imports for setting up the database
from sqlalchemy import create_engine 
from sqlalchemy.orm import Session, sessionmaker
from src.db_config import \
    Base, DocAnnotation, Document, Sentence, Token, CoreferenceChain, \
    CoreferenceMention, SentAnnotation

from src.experiment_config import ExperimentConfig
from src.utils import load_json

"""
Class for preprocessing all implemented datasets, loading them into sqlalchemy 
databases and obtaining sqlalchemy Sessions for them.
"""

def load_target_prediction_res(path):
    llm_res = load_json(path)["data"]

    # format into dicts
    llm_res_dicts = [
        {   "id": str(key), "doc_id": str(key).split("_")[0], "sent_idx": str(key).split("_")[1], "path_id": str(key).split("_")[2], \
            f"target_group": entry["annotation"]["target_group"] \
        } for key, entry in llm_res.items()
    ]
    return pd.DataFrame.from_dict(llm_res_dicts)

def get_llm_metaphor_score_df(path):
    llm_res = load_json(path)["data"]

    # format into dicts
    llm_res_dicts = [
        {   "id": str(key), "doc_id": str(key).split("_")[0], "sent_idx": str(key).split("_")[1], "path_id": str(key).split("_")[2], \
            "sentence": entry["text"], "source_word": entry["source_word"], "target_word": entry["source_word"], \
            "llm_met_score": entry["annotation"]["metaphor_salience_score"], "llm_explanation": entry["annotation"]["explanation"] \
        } for key, entry in llm_res.items()
    ]
    return pd.DataFrame.from_dict(llm_res_dicts)

def get_llm_metaphor_binary_df(path, logger):
    llm_res = load_json(path)["data"]

    # format into dicts
    llm_res_dicts = [
        {   "id": str(key), "doc_id": str(key).split("_")[0], "sent_idx": str(key).split("_")[1], "path_id": str(key).split("_")[2], \
            "text": entry["text"], "sentence": entry["text"].split("\n Sentence: '")[1], \
            "source_word": entry["source_word"].lower().strip(), "target_word": entry["target_word"].lower().strip(), \
            "llm_met_class": entry["annotation"]["classification"]\
            # , "llm_explanation": entry["annotation"]["explanation"] \
        } for key, entry in llm_res.items()
    ]
    
    errors = 0
    for idx, d in enumerate(llm_res_dicts):
        if d["llm_met_class"].strip().lower() == "metaphorical":
            llm_res_dicts[idx]["llm_met_class"] = True 
        elif d["llm_met_class"].strip().lower() == "literal":
            llm_res_dicts[idx]["llm_met_class"] = False 
        else:
            errors += 1
            llm_res_dicts[idx]["llm_met_class"] = False

    if errors > 0:
        logger.warning(f"{errors} metaphor classifications with invalid LLM output")
        
    
    return pd.DataFrame.from_dict(llm_res_dicts)
    
class DatasetLoader:
    """
    This class covers loading the evaluation text datasets.
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = config.logger
        self.cpus = 4
        self.accepted_datasets = ["subframes_immigration", "subframes_guns", "subframes_abortion", "tweets_immigration", "lcc-small", "lcc-large", "podcasts"]
        self.texts: List[str] = []

        self.nlp = None  # load later
        self.max_doc_len = 42000  # max character length for stanza processing

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the text by converting to lowercase, removing punctuation, and normalizing whitespace.
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())
        return text

    def filter_texts(self, texts: dict) -> dict:
        """
        Filter texts based on word count criteria.
        """
        return {
            id: text for id, text in texts.items()
            if self.config.occurrences_config.min_words < len(text.split()) < self.config.occurrences_config.max_words
        }
    
    def create_database(self):
        """
        Create the SQLite database and return the engine.
        """
        if not os.path.exists(self.output_path):  # if newly created, log it
            self.logger.info(f"Creating database at {self.output_path}")

        engine = create_engine(f"sqlite:///{self.output_path}", echo=False)
        Base.metadata.create_all(engine)
        return engine
    
    def process_coref(self, session, processed_doc, doc_id: str):
        """
        Process coreference chains and mentions from the processed document and add them to the database.
        """
        for item in processed_doc.coref:
            coref_chain_id = f"{doc_id}_{item.index}"
            coref_chain = CoreferenceChain(
                id=coref_chain_id,
                document_id=doc_id,
                representative_text=item.representative_text
            )
            session.add(coref_chain)
            
            containing_sentences = []
            mention_idx = 0
            for mention in item.mentions:
                # only add unique mentions per sentence
                if mention.sentence not in containing_sentences:
                    # get mention text from sentence words
                    mention_sentence_words = processed_doc.sentences[mention.sentence].words
                    try:
                        mention_text = " ".join([mention_sentence_words[i].text for i in range(mention.start_word, mention.end_word)])
                    except Exception as e:
                        self.logger.error(f"mention.start_word={mention.start_word}, mention.end_word={mention.end_word}")
                        self.logger.error(f"Error getting mention text for mention {mention_idx} in coreference chain {coref_chain_id} in document {doc_id}: {e}")
                        continue

                    coref_mention = CoreferenceMention(
                        id=f"{coref_chain_id}_{mention_idx}",
                        coreference_chain_id=coref_chain_id,
                        sentence_id=f"{doc_id}_{mention.sentence}",
                        start_token_id=f"{doc_id}_{mention.sentence}_{mention.start_word}",
                        end_token_id=f"{doc_id}_{mention.sentence}_{mention.end_word - 1}",
                        text=mention_text
                    )
                    session.add(coref_mention)

                    # track sentences already added, increment mention idx
                    containing_sentences.append(mention.sentence)
                    mention_idx += 1

            session.commit()

        return session
    
    def process_sentences(self, session, processed_doc, doc_id: str):
        """
        Process sentences and tokens from the processed document and add them to the database.
        """
        for sentence in processed_doc.sentences:
            sentence_id = f"{doc_id}_{sentence.sent_id}"
            sent = Sentence(
                id=sentence_id,
                document_id=doc_id,
                text=sentence.text,
                clean_text=self.preprocess_text(sentence.text),
                num_tokens=len(sentence.words),
                start_char = sentence.tokens[0].start_char,
                end_char = sentence.tokens[-1].end_char
            )
            session.add(sent)

            for idx, word in enumerate(sentence.words):
                token = Token(
                    id=f"{sentence_id}_{idx}",
                    sentence_id=sent.id,
                    text=word.text,
                    pos=word.upos,
                    start_char=word.start_char,
                    end_char=word.end_char
                )
                session.add(token)
            session.commit()

        return session

    import re

    def segment_text(self, full_doc_text):
        """
        Break full_doc_text into a list of chunks each <= max_len characters,
        breaking at punctuation boundaries when possible.
        """
        if len(full_doc_text) <= self.max_doc_len:
            return [full_doc_text]

        # Punctuation to prefer breaking on, in order of preference
        sentence_end = re.compile(r'[.!?]\s+')
        soft_break = re.compile(r'[,;:\-]\s+')

        chunks = []
        text = full_doc_text

        while len(text) > self.max_doc_len:
            window = text[:self.max_doc_len]

            # 1. Try to break at the last sentence-ending punctuation in the window
            matches = list(sentence_end.finditer(window))
            if matches:
                split_at = matches[-1].end()
            else:
                # 2. Fall back to last soft punctuation (comma, semicolon, etc.)
                matches = list(soft_break.finditer(window))
                if matches:
                    split_at = matches[-1].end()
                else:
                    # 3. Fall back to last whitespace
                    last_space = window.rfind(' ')
                    split_at = last_space + 1 if last_space != -1 else self.max_doc_len

            chunks.append(text[:split_at].strip())
            text = text[split_at:]

        if text:
            chunks.append(text.strip())

        return chunks

    def preprocess_documents(self, engine, raw_data: dict):
        """
        Preprocess documents using stanza and store in the database.
        raw_data: dict of doc_id -> {'text': str}
        """
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,coref,mwt', use_gpu=True)

        all_doc_chunks = {}
        for doc_id, doc_info in tqdm(raw_data.items(), desc="Chunking documents", total=len(raw_data)):
            # for chunk_idx, chunk_text in enumerate(doc_chunks): # in gun control corpus, skip "4305"
            doc_id = str(doc_id)
            doc_text = doc_info["text"]

            # remove repeated non-breaking spaces that cause stanza issues
            target_string = "&#160;"
            pattern = r'(' + re.escape(target_string) + r')\W*\1+'
            full_doc_text = re.sub(pattern, target_string, doc_text, flags=re.IGNORECASE)
            doc_chunks = self.segment_text(full_doc_text)
            for chunk_idx, chunk_text in enumerate(doc_chunks):
                all_doc_chunks[f"{doc_id}-{chunk_idx}"] = chunk_text
        self.logger.info(f"{len(all_doc_chunks)} doc chunks available for processing")

        # filter out texts that are already in the database
        with Session(engine) as session:
            existing_docs = session.query(Document.id).all()
            existing_doc_ids = {doc_id for (doc_id,) in existing_docs}
            all_doc_chunks = {doc_id: doc_info for doc_id, doc_info in all_doc_chunks.items() if str(doc_id) not in existing_doc_ids}
            self.logger.info(f"Filtered out {len(existing_doc_ids)} documents already in the database. {len(all_doc_chunks)} documents remaining to process.")

            for chunk_id, chunk_text in tqdm(all_doc_chunks.items(), desc="Sentence tokenizing chunks", total=len(all_doc_chunks)):           

                try:  # process the document with stanza, catch any errors
                    processed_doc = self.nlp(chunk_text)

                except Exception as e:  # perhaps handle differently
                    self.logger.error(f"Issue processing document {doc_id} of length {len(chunk_text)} with stanza error: {e}")
                    # skipped_docs.append(doc_id)

                    # clean up and move on
                    session.close()
                    torch.cuda.empty_cache() # then clear the cache
                    raise e

                if len(doc_chunks) > 1:
                    # add index at end of id
                    chunk_id = f"{doc_id}-{chunk_idx}"
                
                doc = Document(
                    id=chunk_id,
                    text=chunk_text,
                    clean_text=self.preprocess_text(chunk_text),
                    num_sentences=len(processed_doc.sentences)
                )
                session.add(doc)
                session.commit()

                # session = self.process_coref(session, processed_doc, chunk_id)
                session = self.process_sentences(session, processed_doc, chunk_id)

                del processed_doc  # free memory
                torch.cuda.empty_cache() # then clear the cache

                # commit changes after each document to avoid large transactions
                session.commit()
            session.close()


    def add_document_annotations(self, engine, doc_ids: list, labels: dict, label_type: str, force_replace: bool = False):
        """
        Add document annotations (labels) to the database.
        """
        
        if force_replace: # delete existing annotations of this type
            self.logger.info(f"Force replacing existing {label_type} annotations.")
            with Session(engine) as session:
                session.query(DocAnnotation).filter(DocAnnotation.annotation_type == label_type).delete()
                session.commit()
                self.logger.info(f"Deleted existing {label_type} annotations, ready to add {len(doc_ids)} new annotations.")
        
        else: # filter annotations that are already in the database
            with Session(engine) as session:
                existing_annotations = session.query(DocAnnotation.document_id, DocAnnotation.annotation_type).all()
                existing_annotation_set = {(doc_id, ann_type) for (doc_id, ann_type) in existing_annotations}
                doc_ids = [doc_id for doc_id in doc_ids if (str(doc_id), label_type) not in existing_annotation_set]
                self.logger.info(f"Filtered out {len(existing_annotation_set)} existing {label_type} annotations. {len(doc_ids)} new annotations to add.")

        missing_labels = 0
        for article_id in tqdm(doc_ids, desc=f"Adding {label_type} labels to articles", total=len(doc_ids)):
            if article_id not in labels:
                self.logger.debug(f"Article id {article_id} not found in provided labels for {label_type}, skipping.")
                missing_labels += 1
                continue

            label = labels[article_id]
            with Session(engine) as session:
                annotation = DocAnnotation(
                    id=f"{article_id}_{label_type}",
                    document_id=str(article_id),
                    annotation_type=label_type,
                    annotation_value=label
                )
                session.add(annotation)
                session.commit()

        if missing_labels > 0:
            self.logger.warning(f"Skipped {missing_labels} document annotations due to missing labels for {label_type}.")

    def add_sentence_annotations(self, engine, sent_ids: list, labels: dict, label_type: str):
        """
        Add document annotations (labels) to the database.
        """
        # filter annotations that are already in the database
        with Session(engine) as session:
            existing_annotations = session.query(SentAnnotation.sentence_id, SentAnnotation.annotation_type).all()
            existing_annotation_set = {(sent_id, ann_type) for (sent_id, ann_type) in existing_annotations}
            sent_ids = [sent_id for sent_id in sent_ids if (str(sent_id), label_type) not in existing_annotation_set]
            self.logger.info(f"Filtered out {len(existing_annotation_set)} existing {label_type} annotations. {len(sent_ids)} new annotations to add.")

        for sent_id in tqdm(sent_ids, desc=f"Adding {label_type} labels to sentences", total=len(sent_ids)):
            label = labels[sent_id]
            with Session(engine) as session:
                annotation = SentAnnotation(
                    sentence_id=str(sent_id),
                    annotation_type=label_type,
                    annotation_value=label
                )
                session.add(annotation)
                session.commit()

    def create_subframes_db(self, raw_data_path: str) -> None:
        """
        Create database of preprocessed documents in subframes dataset
        """
        # load pickle files
        with open(f"{raw_data_path}/article_data.pkl", "rb") as f:
            article_data = pickle.load(f)  # list of dicts [urls, dates, headlines, article_text, labels] (7 dicts total)

        self.logger.info(f"Loaded {len(article_data)} articles from {raw_data_path}/article_data.pkl")
        headlines = article_data[2]
        article_text = article_data[3]
        polarity_labels = article_data[4]  # dict of article_id -> right/left

        # format for processing
        raw_data = {k: {'text': article_text[k]} for k in headlines.keys()}

        # create the database
        engine = self.create_database()

        # process the articles
        self.preprocess_documents(engine, raw_data)

        # add polarity document annotations
        self.add_document_annotations(engine, list(raw_data.keys()), polarity_labels, label_type="polarity")

        return
    
    def load_podcast_data(self, raw_data_path: str):

        transcript_dir = f"{raw_data_path}/transcripts"
        metadata_dir = f"{raw_data_path}/metadata"

        # load raw data
        raw_data = {} # doc_id: {"text"}
        channel_labels, polarity_labels = [], []
        for path in Path(metadata_dir).iterdir(): # 
            if path.is_file():
                metadata = load_json(path)
                channel_label = metadata["author"]
                polarity = str(path).split("/")[-1].split("-")[0]
                
                for ep_id, ep in metadata["episodes"].items():

                    # get path to transcript, load text
                    ep_name = ep["file_path"].split("/")[-1].replace(".mp3", ".json")
                    ep_transcript_path = f"{transcript_dir}/{ep_name}"
                    transcript_data = load_json(ep_transcript_path)

                    transcript_text = transcript_data["text"]
                    raw_data[ep_id] = {"text": transcript_text}

                    channel_labels.append(channel_label)
                    polarity_labels.append(polarity)

        # create the database
        engine = self.create_database()

        # process the articles
        self.preprocess_documents(engine, raw_data)

        # # add annotations
        # self.add_document_annotations(engine, list(raw_data.keys()), channel_labels, label_type="channel")
        # self.add_document_annotations(engine, list(raw_data.keys()), polarity_labels, label_type="polarity")

        return

    def create_tweets_immigration_db(self, raw_data_path: str) -> None:
        """
        Load and preprocess mendhelsohn immigration tweets data.
        """

        with open(f"{raw_data_path}/tweet_ids_with_metaphor_scores.tsv", "rb") as f:
            generated_metaphor_scores = pd.read_csv(f, sep="\t", dtype=str)  # dataframe

        # with open(f"{raw_data_path}/tweet_data_2021-2023.tsv", "rb") as f:
        #     tweet_data = pd.read_csv(f, sep="\t")  # dataframe
        
        with open(f"{raw_data_path}/tweet_data_2018-2019.tsv", "rb") as f:
            # tweet_data = pd.concat([tweet_data, pd.read_csv(f, sep="\t")], ignore_index=True)
            tweet_data = pd.read_csv(f, sep="\t", dtype=str)  # dataframe

        with open(f"{raw_data_path}/annotated_data_with_text.tsv", "rb") as f:
            annotated_data = pd.read_csv(f, sep="\t", dtype=str)  # dataframe

        self.logger.info(f"Loaded {len(generated_metaphor_scores)} rows of generated metaphor scores, {len(tweet_data)} rows of raw tweet data, and {len(annotated_data)} rows of annotated data from {raw_data_path}.")

        # combine data into a dictionary
        # the annotated data has 'id_str' which corresponds to 'id_str' in tweet_data
        combined_data = {}
        for _, row in tqdm(tweet_data.iterrows(), desc="Combining tweet data into dictionary", total=len(tweet_data)):
            combined_data[row['id_str']] = {
                'text': row['text'],
                'human_metaphor_score': None,
                'human_source_domain': None,
                "log_retweets": row['log_retweets']
            }

        # add llm generated metaphor scores to the combined data dictionary
        generated_ann_types = ["log_retweets"]  # track the generated annotation types we have, start with log_retweets from the raw data
        for _, row in tqdm(generated_metaphor_scores.iterrows(), desc="Adding generated metaphor scores into dictionary", total=len(generated_metaphor_scores)):
            id_str = str(row['id_str'])
            if id_str in combined_data:
                # add each each annotation type as a separate field in the combined data dictionary
                for col in generated_metaphor_scores.columns:
                    if col != 'id_str':
                        clean_col = col.lower().replace(" ", "_")  # lowercase the column name, replace whitespace with underscore for consistency
                        combined_data[id_str][clean_col] = row[col]
                        if clean_col not in generated_ann_types:
                            generated_ann_types.append(clean_col)
            else:
                raise ValueError(f"Tweet id {id_str} in generated metaphor scores not found in raw tweet data.")


        # add hand annotations to the combined data dictionary
        if self.config.labeled_only:
            for _, row in tqdm(annotated_data.iterrows(), desc="Adding annotated data into dictionary", total=len(annotated_data)):
                id_str = row['id_str']
                if id_str in combined_data:
                    combined_data[id_str]['human_metaphor_score'] = row['percent_yes']
                    combined_data[id_str]['human_source_domain'] = row['concept']
                else:
                    if not self.config.labeled_only:
                        pass  # if not filtering to labeled data only, we can skip these annotations, but if we are filtering to labeled data only, then this is an error because all annotated data should be in the raw data
                    raise ValueError(f"Tweet id {id_str} in annotated data not found in raw tweet data.")
            # filter to only labeled data
            combined_data = {k: v for k, v in combined_data.items() if v['human_metaphor_score'] is not None and v['human_source_domain'] is not None}
            self.logger.info(f"Filtered to {len(combined_data)} labeled tweets only for processing.")

        else:
            # filter just to entries that have generated annotations
            first_ann_type = generated_ann_types[1]
            combined_data = {k: v for k, v in combined_data.items() if first_ann_type in v and v[first_ann_type] is not None}
            self.logger.info(f"Filtered to {len(combined_data)} tweets with generated labels only for processing.")

  
        # create the database
        engine = self.create_database()

        # process the tweets
        self.preprocess_documents(engine, combined_data)

        # add metaphor document annotations

        if self.config.labeled_only:
            labeled_data = {k: v for k, v in combined_data.items() if v['human_metaphor_score'] is not None and v['human_source_domain'] is not None}
            self.add_document_annotations(engine, list(labeled_data.keys()), 
                {k: v['human_metaphor_score'] for k, v in labeled_data.items()}, label_type="human_metaphor_score")

            self.add_document_annotations(engine, list(labeled_data.keys()), 
                {k: v['human_source_domain'] for k, v in labeled_data.items()}, label_type="human_source_domain")
        
        # # add generated annotation types as document annotations
        # for ann_type in generated_ann_types:
        #     self.add_document_annotations(engine, list(combined_data.keys()), 
        #         {k: v[ann_type] for k, v in combined_data.items() if ann_type in v and v[ann_type] is not None}, label_type=ann_type)
        
    def load_lcc_data(self, raw_data_path: str):
        """
        Load and process LCC data, return huggingface dataset.
        """
        if self.config.dataset == "lcc-small":
            size = "small"
        elif self.config.dataset == "lcc-large":
            size = "large"
        else:
            raise ValueError(f"Unsupported LCC dataset: {self.config.dataset}. Supported datasets are: lcc-small, lcc-large")
        
        data_path = f"{raw_data_path}/en_{size}.xml"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"LCC data file not found at {data_path}. Please download the raw data and place it there to proceed.")

        # Assuming 'data.xml' contains your XML data
        with open(data_path, 'r') as f:
            xml_text = f.read()
            # edit xml_text to preserve <LmSource></LmSource> and <LmTarget></LmTarget> span tags
            # for some reason xmltodict removes these tags otherwise   
            xml_text = re.sub(r'<LmSource>', r'SOURCE_START', xml_text)
            xml_text = re.sub(r'</LmSource>', r'SOURCE_END', xml_text)
            xml_text = re.sub(r'<LmTarget>', r'TARGET_START', xml_text)
            xml_text = re.sub(r'</LmTarget>', r'TARGET_END', xml_text)
            ordered_dict = xmltodict.parse(xml_text)


        def get_annotation_content(doc, skipped_annotations: int) -> tuple:
            """
            Get the human metaphor annotation content from the document annotations.
            """
            try:
                annotation_content = doc['Annotations']["MetaphoricityAnnotations"]["MetaphoricityAnnotation"]
            except Exception: # no metaphor annotation, skip
                skipped_annotations += 1
                return skipped_annotations, None

            if type(annotation_content) is list: # only add annotations with full agreement
                scores = [ann["@score"] for ann in annotation_content]
                if all(score == scores[0] for score in scores):
                    human_metaphor_score = str(int(float(scores[0])))
                else:
                    skipped_annotations += 1
                    human_metaphor_score = None

            # straightforward case, just one annotation
            elif type(annotation_content) is dict:
                human_metaphor_score = str(int(float(annotation_content["@score"])))
            
            else:
                self.logger.error(f"Unexpected annotation content type: {annotation_content}")
                raise ValueError()
            
            return skipped_annotations, human_metaphor_score

        
        data = list(ordered_dict.values())[0]["LmInstance"]
        data_formatted = {}
        skipped_annotations = 0
        
        for doc in tqdm(data, desc="Formatting raw LCC data to add to db", total=len(data)):

            skipped_annotations, human_metaphor_score = get_annotation_content(doc, skipped_annotations)
            
            text_content = doc['TextContent']  # gather text from Prev, Current, Next
            text, included_segments = [], []
            for segment in ["Prev", "Current", "Next"]:
                if segment in text_content and text_content[segment]:
                    new_text = text_content[segment]
                    if segment == "Current":
                        # ERROR: this gives wrong indices because of re.sub below
                        # capture the target and source strings
                        
                        source_match = re.search(r'SOURCE_START(.*?)SOURCE_END', new_text)
                        target_match = re.search(r'TARGET_START(.*?)TARGET_END', new_text)

                        source_str = source_match.group(1).replace('SOURCE_START', '').replace('SOURCE_END', '').strip()
                        target_str = target_match.group(1).replace('TARGET_START', '').replace('TARGET_END', '').strip()

                        # get character indices of the spans
                        source_spans = list(source_match.span()) 
                        target_spans = list(target_match.span())
                    
                        # adjust spans to be relative to the full text later
                        if source_spans[0] < target_spans[0]: # source comes first
                            target_spans[0] -= (len('SOURCE_START') + len('SOURCE_END'))
                            target_spans[1] -= (len('SOURCE_START') + len('SOURCE_END') + len('TARGET_START') + len('TARGET_END'))
                            
                            source_spans[1] -= (len('SOURCE_START') + len('SOURCE_END'))
                        
                        else: # target comes first
                            source_spans[0] -= (len('TARGET_START') + len('TARGET_END'))
                            source_spans[1] -= (len('TARGET_START') + len('TARGET_END') + len('SOURCE_START') + len('SOURCE_END'))
                            target_spans[1] -= (len('TARGET_START') + len('TARGET_END'))

                        # need to remove the SOURCE_START/END and TARGET_START/END tags 
                        new_text = re.sub(r'SOURCE_START', '', new_text)
                        new_text = re.sub(r'SOURCE_END', '', new_text)
                        new_text = re.sub(r'TARGET_START', '', new_text)
                        new_text = re.sub(r'TARGET_END', '', new_text)

                        # double check spans
                        assert new_text[source_spans[0]:source_spans[1]] == source_str, f"Source span text mismatch: ({source_spans[0]}, {source_spans[1]}) {new_text[source_spans[0]:source_spans[1]]} != {source_str} (text: {new_text})"
                        assert new_text[target_spans[0]:target_spans[1]] == target_str, f"Target span text mismatch: ({target_spans[0]}, {target_spans[1]}) {new_text[target_spans[0]:target_spans[1]]} != {target_str} (text: {new_text})"



                    included_segments.append(segment)
                    text.append(new_text)
            text = " ".join(text).strip()

            cur_doc = {
                'text': text,
                'included_segments': included_segments,
                'target_concept': doc['@targetConcept'],
                'target_str': target_str,
                'target_spans': target_spans,
                'source_str': source_str,
                'source_spans': source_spans,
                'human_metaphor_score': human_metaphor_score
            }
            
            data_formatted[doc['@id']] = cur_doc
 

        # create the database
        engine = self.create_database()

        # process the tweets
        self.preprocess_documents(engine, data_formatted)

        # add human hand annotations
        self.add_document_annotations(engine, list(data_formatted.keys()), 
            {k: v['human_metaphor_score'] for k, v in data_formatted.items() if v['human_metaphor_score'] is not None}, \
                label_type="human_metaphor_score")

        # add target concept document annotations
        self.add_document_annotations(engine, list(data_formatted.keys()), 
            {k: v['target_concept'] for k, v in data_formatted.items()}, \
                label_type="human_target_concept")

        # add source and target span and string annotations
        self.add_document_annotations(engine, list(data_formatted.keys()), 
            {k: v['target_spans'][0] for k, v in data_formatted.items()}, \
                label_type="target_char_start")
        
        self.add_document_annotations(engine, list(data_formatted.keys()), 
            {k: v['target_spans'][1] for k, v in data_formatted.items()}, \
                label_type="target_char_end")
        
        self.add_document_annotations(engine, list(data_formatted.keys()), 
            {k: v['target_str'] for k, v in data_formatted.items()}, \
                label_type="target_string")
        
        self.add_document_annotations(engine, list(data_formatted.keys()), 
            {k: v['source_spans'][0] for k, v in data_formatted.items()}, \
                label_type="source_char_start")
        
        self.add_document_annotations(engine, list(data_formatted.keys()), 
            {k: v['source_spans'][1] for k, v in data_formatted.items()}, \
                label_type="source_char_end")
        
        self.add_document_annotations(engine, list(data_formatted.keys()), 
            {k: v['source_str'] for k, v in data_formatted.items()}, \
                label_type="source_string")

        if skipped_annotations > 0:
            self.logger.warning(f"Skipped {skipped_annotations} documents with multiple metaphor annotations.")
    

    def load_preprocessed_data(self) -> Session:
        """
        Load and process text data based on desired dataset and labels.
        """
        
        if self.config.dataset.lower() not in self.accepted_datasets:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}. Supported datasets are: {self.accepted_datasets}")

        # set up output path
        self.output_path = self.config.db_path
        if os.path.exists(self.output_path):  
            self.logger.info(f"Found existing database at {self.output_path}. Loading data from there...")

        else:
            if self.config.skip_load:
                raise FileNotFoundError(f"'skip_load' option selected, but db doesn't exist.")

            # create the database and load data from raw files
            self.logger.info(f"No previously saved data found. Loading data from {self.config.dataset} dataset...")


        # check db against raw data, unless specifically directed not to
        if not self.config.skip_load:

            # ensure the raw data path is there
            raw_data_path = f"{self.config.raw_data_dir}/{self.config.dataset.lower()}"
            if not os.path.exists(raw_data_path):  # load tsv files
                raise FileNotFoundError(f"Raw data file not found at {raw_data_path}. Please download the raw data and place it there to proceed.")

            if self.config.dataset in ['subframes_immigration', 'subframes_guns', 'subframes_abortion']:
                self.create_subframes_db(raw_data_path)
            elif 'tweets_immigration' == self.config.dataset:
                self.create_tweets_immigration_db(raw_data_path)
            elif 'lcc-small' == self.config.dataset or 'lcc-large' == self.config.dataset:
                self.load_lcc_data(raw_data_path)    
            elif 'podcasts' == self.config.dataset:
                self.load_podcast_data(raw_data_path)

        # either way, load the database
        engine = create_engine(f"sqlite:///{self.output_path}", echo=False)
        Session = sessionmaker(bind=engine)
        return Session()


    def load_raw_tweets(self, processed_only: bool, metaphors_only:bool, metaphor_thresh) -> pd.DataFrame:
        if self.config.labeled_only and processed_only:
            self.logger.warning(f"Both 'labeled_only' and 'processed_only' set to true which does not apply for immigration tweet dataset. Proceeding with 'labeled_only'...")

        if self.config.labeled_only == False and metaphors_only:
            self.logger.warning(f"'metaphors_only' option selected while processing unlabelled dataset. Ignoring this and proceeding...")

        # LOAD THE SMALLER, HAND ANNOTATED DATASET
        if self.config.labeled_only:
            tweet_ann_path = f"{self.config.env_vars['RAW_DATA_DIR']}/tweets_immigration/annotated_data_with_text.tsv"
            df = pd.read_csv(tweet_ann_path, sep='\t')
            df["id"] = df["id_str"].astype(str)
            df = df.drop_duplicates(subset="id_str")

            # open the other raw data file to add ideology column
            ideology_path = f"{self.config.env_vars['RAW_DATA_DIR']}/tweets_immigration/tweet_data_2018-2019.tsv"
            ideology_df = pd.read_csv(ideology_path, sep='\t')

            ideology_df["id"] = ideology_df["id_str"].astype(str)
            ideology_df = ideology_df.drop_duplicates(subset="id")
            ideology_df = ideology_df[["id", "ideology"]]

            df = df.merge(ideology_df, how="left", on="id")

            # check that all the tweets have an ideology label
            valid_ideology_df = df.dropna(subset=["ideology"])
            assert len(valid_ideology_df) == len(df), f"{len(df) - len(valid_ideology_df)} rows don't have ideology value"

            if metaphors_only:
                df["met_class"] = df["percent_yes"].apply(lambda x: 1 if x >= metaphor_thresh else 0)
                df = df[df["met_class"] == 1]

        
        else: # LOAD THE BIG DATASET WITH LLM-INFERRED LABELS
            
            # load all of the tweets from the raw directory
            df = pd.read_csv(f"{self.config.env_vars['RAW_DATA_DIR']}/tweets_immigration/tweet_data_2018-2019.tsv", sep='\t')
            df["id"] = df["id_str"].astype(str)
            df = df.drop_duplicates(subset="id")

            if processed_only:
                # load sdps to see which files we processed
                sdp_path = f"{self.config.env_vars['RESULTS_DIR']}/metaphor_classification/llm/tweets_immigration_source_verb_target_noun_binary_met_class_qwen.json"
                processed_sdps = load_json(sdp_path)
                processed_sdps = processed_sdps["data"]
                sdp_ids = list(processed_sdps.keys())
                doc_ids = [t_id.split("_")[0] for t_id in sdp_ids]
                doc_ids = set(list(doc_ids))

                # filter down to only the data we processed
                df = df[df["id"].isin(doc_ids)]

        if "ideology" in df.columns:
            df["polarity"] = df["ideology"].apply(lambda x: "left" if x <= 0 else "right")
        return df
    
    def load_raw_podcasts(self):
        raw_data_path = f"{self.config.env_vars['RAW_DATA_DIR']}/podcasts"
        metadata_path = f"{raw_data_path}/metadata"

        podcast_data = []
        metadata_dir = Path(metadata_path)
        for file in metadata_dir.iterdir():
            if file.is_file():
                if file.name.endswith(".json"):
                    polarity, channel = file.name.split("-")
                    channel = channel.replace(".json", "").replace("_", " ")
                    metadata = load_json(file)
                    for ep_id, ep_data in metadata["episodes"].items():
                        podcast_data.append({
                            "id": ep_id,
                            "polarity": polarity,
                            "year": pd.to_datetime(ep_data["episode_date"]).year,
                            "channel": channel,
                            "duration": ep_data["duration"]
                        })

        return pd.DataFrame.from_dict(podcast_data)


    def load_raw_subframes_data(self, dataset: str, processed_only: bool = False) -> pd.DataFrame:
        # load pickle files
        raw_data_path = f"{self.config.env_vars['RAW_DATA_DIR']}/{dataset}"
        with open(f"{raw_data_path}/article_data.pkl", "rb") as f:
            article_data = pickle.load(f)  # list of dicts [urls, dates, headlines, article_text, labels] (7 dicts total)

        # headlines = article_data[2]
        # article_text = article_data[3]
        # polarity_labels = article_data[4]  # dict of article_id -> right/left

        df = pd.DataFrame() # get df where each row corresponds to an article, columns are labels
        df["id"] = list(article_data[2].keys())
        df["polarity"] = list(article_data[4].values())

        if processed_only:
            # load sdps to see which files we processed
            sdp_path = f"{self.config.env_vars['RESULTS_DIR']}/metaphor_classification/llm/{dataset}_source_verb_target_noun_binary_met_class_qwen.json"
            processed_sdps = load_json(sdp_path)
            processed_sdps = processed_sdps["data"]
            sdp_ids = list(processed_sdps.keys())
            doc_ids = [t_id.split("_")[0] for t_id in sdp_ids]
            doc_ids = set(list(doc_ids))

            # filter down to only the data we processed
            df = df[df["id"].isin(doc_ids)]

        return df

    def load_raw_data(self, processed_only: bool=False,  metaphors_only:bool=False, metaphor_thresh:float=0.3) -> pd.DataFrame:

        if self.config.dataset.lower() not in self.accepted_datasets:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}. Supported datasets are: {self.accepted_datasets}")
        
         # ensure the raw data path is there
        raw_data_path = f"{self.config.raw_data_dir}/{self.config.dataset.lower()}"
        if not os.path.exists(raw_data_path):  # load tsv files
            raise FileNotFoundError(f"Raw data file not found at {raw_data_path}. Please download the raw data and place it there to proceed.")

        if self.config.dataset in ['subframes_immigration', 'subframes_guns', 'subframes_abortion']:
            return self.load_raw_subframes_data(self.config.dataset_out_name, processed_only)
        
        elif 'tweets_immigration' == self.config.dataset:
            return self.load_raw_tweets(processed_only, metaphors_only, metaphor_thresh)

        elif self.config.dataset == "podcasts":
            return self.load_raw_podcasts()
        
        elif 'lcc-small' == self.config.dataset or 'lcc-large' == self.config.dataset:
            raise NotImplementedError

    def load_metaphor_classification_results(self, metaphors_only:bool = False, with_ads:bool = False) -> pd.DataFrame:
        self.logger.info(f"Loading qwen 'binary metaphor' (specified best model) classification results...")
        llm_ann_path = f"{self.config.results_dir}/metaphor_classification/llm/{self.config.dataset_out_name}_source_verb_target_noun_binary_met_class_qwen.json"
        if with_ads:
            if self.config.dataset_out_name == "podcasts":
                llm_ann_path = llm_ann_path.replace(".json", "_with_ads.json")
            else:
                raise ValueError(f"'with_ads' setting for metaphor_classification_results() not available for '{self.config.dataset_out_name}' dataset")
        df = get_llm_metaphor_binary_df(llm_ann_path, self.logger)

        if metaphors_only:
            df = df[df["llm_met_class"] == True]

        return df
    
    def load_old_metaphor_classification_results(self, metaphors_only:bool = False, met_thresh:float = 0.4):
        self.logger.info(f"Loading qwen 'metaphor score' (specified best model for previous project) classification results...")
        llm_ann_path = f"{self.config.results_dir}/metaphor_classification/llm/{self.config.dataset_out_name}_source_verb_score_qwen.json"
        df = get_llm_metaphor_score_df(llm_ann_path)

        if metaphors_only:
            df = df[df["llm_met_score"] >= 0.4]

        return df
