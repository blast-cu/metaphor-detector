from tqdm import tqdm
import logging
import concurrent
from pydantic import BaseModel
from typing import List
import random
import pandas as pd

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from src.utils import load_json, get_token, VERB_PATHS
from src.db_config import Sentence, Token, DocAnnotation
from src.data_loading import DatasetLoader

"""
Helper class for prompt engineering 'experiments.llm_metaphor_classifier'
"""

class SourceCharTargetAF(BaseModel):
    source_characterizes_target: bool
    explanation: str

class SourceVerbBinaryAF(BaseModel):
    # {'classification': str, 'explanation': 'your reasoning for the answer'}
    classification: str
    explanation: str

class SourceVerbScoreAF(BaseModel):
    # {'metaphor_salience_score': float, 'explanation': 'your reasoning for the answer'}
    metaphor_salience_score: float
    explanation: str

class SourceVerbScoreFinalAF(BaseModel):
    # {'metaphor_salience_score': float}
    metaphor_salience_score: float

class MetaphorClassPromptEngineer:
    """
    Match prompt file to desired input data and output format
    """

    def __init__(self, config, logger: logging.Logger, num_workers: int = 8, shuffle_docs: bool = False):

        self.config = config
        self.prompt_file = config.prompt_file
        self.logger = logger
        self.num_workers = num_workers
        self.shuffle_docs = shuffle_docs
        self.env_vars = self.config.env_vars

    def get_af(self):
        """
        Match answer format to prompt file
        """
        if self.prompt_file == "source_characterizes_target.json":
            answer_format = SourceCharTargetAF
        elif self.prompt_file in ["source_verb_bin_class.json", "source_verb_target_noun_binary_met_class.json"]:
            answer_format = SourceVerbBinaryAF
        elif self.prompt_file == "source_verb_score.json":
            answer_format = SourceVerbScoreAF
        elif self.prompt_file == "source_verb_score_final.json":
            answer_format = SourceVerbScoreFinalAF
        else:
            self.logger.error(f"No answer format specified for prompt file: {self.prompt_file}")
            raise ValueError

        return answer_format
    
    def get_shuffled_docs(self, sdps: dict) -> List:
        """
        shuffle the documents to get a good mix of processed results
        """
        # keep sdps from documents consecutive so we process full docs
        doc_dict = {}
        for sdp_id, sdp in tqdm(sdps.items(), desc="creating doc dict for shuffling docs"):
            doc_id = sdp_id.split("_")[0]
            sdp["sdp_id"] = sdp_id
            
            if doc_id not in doc_dict:
                doc_dict[doc_id] = []
            doc_dict[doc_id].append(sdp)
        
        # shuffle document order
        doc_ids = list(doc_dict.keys())
        random.seed(self.config.random_seed)
        random.shuffle(doc_ids)

        # add sdps in new order
        shuffled_sdps = {}
        for doc_id in doc_ids:
            doc_sdps = doc_dict[doc_id] # list of sdps for the doc
            
            for sdp in doc_sdps:
                sdp_id = sdp["sdp_id"]
                del sdp["sdp_id"]
                shuffled_sdps[sdp_id] = sdp

        return shuffled_sdps
        
    def get_data(self, input_file:str = None):

        if self.prompt_file == "metaphor_bin_score.json": # PROMPTS THAT REQUIRE MULTIPLE STEPS
            # read filtered results, hard coded for now
            filtered_file_path = f"{self.env_vars['RESULTS_DIR']}/llm_ann/llm_source_classifier/source_characterizes_target.json"
            data_formatted = self.format_filtered_documents(\
                filtered_file_path=filtered_file_path, ann_key="source_characterizes_target", ann_val=True)

        else: 
            if input_file: # read from csv

                df = pd.read_csv(input_file).set_index("id")
                data_formatted = df.to_dict(orient="index")
                

            else: # NEED DOCS FROM DATABASE
                data_loader = DatasetLoader(self.config)
                data_loader.load_preprocessed_data()
                db_path = self.config.db_path

                # load sdps 
                self.logger.info("Loading SDPs for annotation...")
                rb_results = load_json(f"{self.env_vars['RESULTS_DIR']}/get_candidate_metaphors/rule_based/{self.config.dataset_out_name}_metaphor_paths.json")["data"]

                # temporary sampling
                sample_ids = ['1035286406267060226_0_15_16_2', '1067902006222704640_3_0_1_2', '1022699854060023810_4_5_10_1', '1074734223351791616_0_10_12_6', '1053296719570468869_0_11_13_1', '1072572411575447554_0_38_39_2', '1095710925032747010_0_5_7_1', '1146661241256943617_2_13_17_2', '1012150185433157632_0_13_14_2', '1199672874191052801_0_5_9_1', '979106332367294464_0_2_11_9', '1086028727409172480_0_4_6_1', '1082264386935361536_1_1_3_1', '1075100065683906560_1_2_3_1', '1017825924316909569_0_5_7_9', '1106170535195496448_5_7_8_1', '1082805783094542336_1_0_3_2', '1141448016215322624_0_8_12_1', '1144787019001819136_0_1_3_2', '1128453994077085696_1_10_13_1', '1168201752144777216_3_10_11_2', '1142721273124085760_0_3_5_2', '1131190659157774341_2_1_2_2', '1088631838296412161_1_10_12_1', '1086171173086363649_0_23_27_1', '980830315005460481_0_1_3_2', '1143945662205255682_0_8_10_1', '1038424893216489472_2_4_5_1', '1089525569400856578_0_8_10_1', '1153103981599764481_1_4_5_1', '1009567329691357185_1_10_11_1', '1071108645591617536_0_7_8_1', '971246766153756672_0_0_1_2', '1058162560564895745_3_2_4_1', '1078651704571240448_0_43_47_2', '1076120665235881984_0_1_3_1', '1145522854718558208_0_15_17_1', '977304981551353857_3_12_14_7', '1090032109597544449_2_8_10_2', '1086189091555852290_1_7_12_2', '1115954558776348672_0_4_6_1', '1057288126195003398_2_0_1_2', '1080615344731959296_1_4_10_2', '1120403864719691783_1_5_6_2', '1005710801783549952_1_3_5_2', '1006321450519748608_0_5_6_1', '1207342429143216130_2_1_2_1', '1145718127663226880_0_13_21_2', '981926186841444353_1_0_2_2', '967880541914660864_0_15_16_1', '1082051531464036352_1_10_12_1', '1122551430580187136_0_4_5_1', '1073844077127360512_0_0_15_2', '1009449260868407296_1_14_17_1', '1166028444561395713_0_11_12_1', '1007990305327439873_2_3_4_1', '1075245414243319809_0_0_3_1', '1100088330941878272_2_21_22_1', '1157504654571442177_3_0_2_2', '1067304300290818048_1_5_7_1', '1161274845235163136_0_0_1_2', '1146428907857620993_2_1_3_2', '966668761792172032_0_2_4_1', '1115436087441264641_0_5_6_1', '1082045957666033664_0_1_2_2', '1072877734223929344_0_3_5_1', '1128826537178832897_2_0_4_1', '1123598528926765060_1_0_1_2', '1076490773158416384_0_2_4_2', '1032447849802465281_0_2_3_2', '1072394440948502528_0_13_14_2', '1081315981815959552_1_4_14_9', '1121094740420845569_1_4_6_2', '1096606063313518592_2_10_12_2', '1118377352348819457_4_0_3_2', '1009410708889534472_0_8_9_1', '1100498719240278019_2_5_9_1', '1060787823660412928_0_5_6_1', '1166930350360150016_1_0_2_2', '1129863663148576768_1_12_13_1', '979622554288697346_0_6_7_2', '1150387133347237890_0_23_24_2', '1068607665184608256_0_17_18_2', '1066894120025817088_0_2_3_2', '1021718327758970880_3_10_13_1', '1105337218422300672_0_5_6_2', '1140427386053419009_1_2_3_2', '1126840494393311232_0_20_21_2', '1032750853030129666_0_18_20_1', '1073257215165902849_1_1_4_2', '1089312833446334465_2_1_4_2', '1057059752218386432_1_11_12_2', '1125463448589090816_2_14_15_1', '1058369956293816321_2_7_8_1', '1004124845255340033_1_2_3_1', '988146062413119488_0_2_4_1', '1026220669841883136_2_4_6_9', '1162555560430067712_0_2_4_2', '1019613017846185984_0_17_20_1', '1153057664747880449_5_2_5_2']
        
                if self.shuffle_docs:
                    rb_results = self.get_shuffled_docs(rb_results)

                filter_verbs = True
                if self.prompt_file in ["source_verb_bin_class.json", "source_verb_score.json", "source_verb_score_final.json", "source_verb_target_noun_binary_met_class.json"]: 
                    filter_verbs = True
                    
                # certain prompts also need the target word for context
                add_target_word = False
                if self.prompt_file == "source_verb_target_noun_binary_met_class.json":
                    add_target_word = True

                data_formatted = self.plain_sdp_format_documents(
                    rb_results, db_path, add_target_word=add_target_word, filter_verbs=filter_verbs
                )
        
        return data_formatted
    
    def format_path(self, sdp_id, sdp, add_target_word=False, filter_verbs=True) -> dict:
        # get sentence id by removing token index from token id
        source_token = sdp["source_token"]
        target_token = sdp["target_token"]
        sentence_text = sdp["sentence_text"]

        if filter_verbs: # if pos is set but target is not, we're focusing on that rather than a source or target word
            if add_target_word:
                text = f"Specified Verb: '{source_token["text"]}'\n Target Noun: '{target_token["text"]}'\n Sentence: '{sentence_text}'"
            else:
                text =  f"Specified Verb: '{source_token["text"]}'\n Sentence: '{sentence_text}'"
        
        else:
            if add_target_word:
                text = f"Source Word: '{source_token["text"]}'\nTarget Word: '{target_token["text"]}\n Sentence: '{sentence_text}'"
            
            else: # don't need target token for this prompt
                text = f"Potential Source Word: '{source_token["text"]}'\n Sentence: '{sentence_text}'"
        
        docs_formatted = {"sample_id": sdp_id, "source_word": source_token["text"], "target_word": target_token["text"], "text": text}
        return docs_formatted

    def plain_sdp_format_documents(self, valid_paths, db_path, add_target_word=False, filter_verbs=False):
        """
        Format documents into sentences for annotation.
        """
        data_formatted = {}
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        SessionLocal = sessionmaker(bind=engine)

        if self.config.testing:
            valid_paths = valid_paths[:100]

        def call_formatter(path_item):
            with SessionLocal() as session:
                path_id = path_item[0]
                path = path_item[1]
                res = self.format_path(path_id, path, add_target_word=add_target_word, filter_verbs=filter_verbs)
            return res

        total_docs = len(valid_paths)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(call_formatter, path_item) for path_item in valid_paths.items()]

            with tqdm(total=total_docs, desc=f"Formatting docs {self.num_workers} workers") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        # Get the results of process_doc() for each doc
                        doc = future.result() # returns list of dicts
                        if doc["sample_id"] in data_formatted:
                            self.logger.error("Duplicate sample ids for formatted documents")
                            raise ValueError

                        data_formatted[doc["sample_id"]] = {"text": doc["text"], "source_word": doc["source_word"], "target_word": doc["target_word"]}

                        # Update the progress bar
                        pbar.update(1)

                    except Exception as e:
                        self.logger.exception(f"Error processing doc: {e}")

        return data_formatted
    
    
    def format_filtered_documents(filtered_file_path: str, ann_key: str, ann_val, logger: logging.Logger):
        """
        Class for formatting documents that have already been LLM-filtered, 
        use same text field for valid documents
        
        :param filtered_file_path: path to pre-filtered json
        :param logger: for printing
        """
        filtered_data = load_json(filtered_file_path, logger)["data"]
        data_formatted = {}
        for id, info in filtered_data.items():
            if info["annotation"][ann_key] == ann_val: 
                data_formatted[id] = {
                    "text": info["text"]
                }
            # else skip
        return data_formatted
