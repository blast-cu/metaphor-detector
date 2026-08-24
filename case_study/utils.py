import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import \
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from src.db_config import Document, CoreferenceMention, DocAnnotation
from src.data_loading import DatasetLoader
from src.utils import save_to_json, load_json, VERB_PATHS

"""
utility functions for performing analysis
"""

annotator_ids = {
    "bagyasree": 0,
    "grayson": 1,
    "mreedul": 2,
    "zohar": 3,
    "alex": 4
}

# MATPLOT_STYLE = 'seaborn-v0_8-darkgrid'
MATPLOT_STYLE = 'whitegrid'

# set colors for plots 
blue_hex    = "#1E3888"
yellow_hex  = "#F4D06F"
lt_blue_hex = "#BDD5EA"
red_hex     = "#9C3848"
white_hex   = "#F7F7FF"

BI_PART_COLORS = {
    "left": blue_hex,
    "right": red_hex
}

LLM_MET_THRESH = 0.4

def set_matplot_style():
    sns.set_style("whitegrid")
    # plt.style.use(MATPLOT_STYLE)
    # sns.set_style(MATPLOT_STYLE)
    # sns.set_theme()
    # sns.set_context("paper", font_scale=1.5)

    # set palette
    # sns.set_palette("deep")

def process_metadata(path):
    # loop over each json file
    metadata = {}
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            file_path = os.path.join(path, filename)
            pod_name = filename.replace(".json", "")
            spin, pod_name = pod_name.split("-")
            data = load_json(file_path)
            # process data as needed, e.g. extract episode length, release date, etc."]
            # flatten to create entry for each episode
            for idx, e in enumerate(data["episodes"].items()):
                k, ep_data = e

                ep_data["podcast_name"] = data["title"]
                ep_data["network"] = data["author"]
                ep_data["spin"] = spin

                metadata[f"{pod_name}_episode_{idx}"] = ep_data

    return metadata

def get_path_info(dataset):
    path_info = load_json(f"results/get_candidate_metaphors/rule_based/{dataset}_metaphor_paths_scores.json")["data"]
    path_info_dicts = [{
        "doc_id": r["sdp"][0]["token_id"].split("_")[0], "sent_id": r["sdp"][0]["token_id"].split("_")[1], 
        "sdp": r["sdp"],
        "matches": [item["path"] for item in r["met_paths"]]} for r in path_info]
    
    path_df = pd.DataFrame.from_dict(path_info_dicts).explode("matches")
    path_df = path_df[path_df["matches"].isin(VERB_PATHS)]

    def get_id(row):
        first_token_id = row["sdp"][0]['token_id']
        last_token_index = row["sdp"][-1]['token_id'].split("_")[-1]
        return first_token_id + "_" + last_token_index + "_" + str(row["matches"])

    path_df["id"] = path_df.apply(get_id, axis=1)
    path_df = path_df.set_index("id")
    return path_df.to_dict(orient='index')


def get_target_coref(metaphor_data, session):
    target_tokens = []
    for k, met in tqdm(metaphor_data.items(), desc="collecting coreference data for targets"):
        
        target_token = met["target"]
        token_id = target_token["token_id"]
        target_token_idx = token_id.split("_")[-1]
        
        sentence_id = token_id.split("_")[:-1]
        sentence_id = "_".join(sentence_id)
        
        # get mentions with matching sentence
        mentions = session.query(CoreferenceMention).filter(CoreferenceMention.sentence_id == sentence_id).all()

        # iterate over mentions to get the one where the target token is betwen start and end
        rep_text = target_token["text"]
        for m in mentions:
            start_token_idx = m.start_token.id.split("_")[-1]
            end_token_idx = m.end_token.id.split("_")[-1]

            if target_token_idx >= start_token_idx and target_token_idx <= end_token_idx:
                rep_text = m.coreference_chain.representative_text

        target_tokens.append(rep_text)
        met["target_rep_text"] = rep_text
    
    return target_tokens, metaphor_data


def get_mend_df(config, interim_path):
    config.skip_load = True
    session = DatasetLoader(config).load_data()

    os.makedirs(interim_path, exist_ok=True)

    doc_csv_path = f"{interim_path}/tweet_anns.csv"
    if os.path.exists(doc_csv_path): # load from there
        print(f"Loading document annotations from '{doc_csv_path}'")
        doc_df = pd.read_csv(doc_csv_path)

    else: # compile data and save
        docs = session.query(Document).all()
        print(f"Collected {len(docs)} documents")

        # compile into list of dicts - takes a bout 
        doc_dicts = [doc.to_dict() for doc in tqdm(docs, desc="creating doc dicts")] # 30 s
        for idx, doc in tqdm(enumerate(docs), desc="adding annotations"):
            annotations = [(ann.annotation_type, ann.annotation_value) for ann in doc.doc_annotations]
            doc_dicts[idx].update(annotations)

        doc_df = pd.DataFrame.from_dict(doc_dicts)
        doc_df.to_csv(doc_csv_path, index=False)
        print(f"Saved document annotations to '{doc_csv_path}'")

    return doc_df

def get_thresh_score(human_score, thresh):
        return human_score >= thresh

def rep_mend_analysis(ann_df: pd.DataFrame, prob_col: str, final_col_name:str, thresholds: list = None):
    """
    replicate roc-auc analysis from mend paper
    
    :param ann_df: df with mend hand annotations and out metaphor scores
    :param prob_col: column with our metaphor scores
    :param final_col_name: final name of roc-auc column
    """
    ra_df = ann_df

    str_thresholds = thresholds
    if thresholds is None: 
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        str_thresholds =[f"{int(t * 100)}%" for t in thresholds]

    # for each threshold, get hand annotated score
    roc_auc_dict = {"threshold": [], final_col_name: []}
    for idx, t in enumerate(thresholds):
        ra_df[f"{t}_thresh"] = ra_df["human_metaphor_score"].apply(get_thresh_score, thresh=t)

        if ra_df[f"{t}_thresh"].sum() == 0:
            print(f"WE HAVE A PROBLEM: no positive values for threshold {t}")

        score = roc_auc_score(ra_df[f'{t}_thresh'], ra_df[prob_col])
        
        roc_auc_dict["threshold"].append(str_thresholds[idx])
        roc_auc_dict[final_col_name].append(score)

    roc_auc_df = pd.DataFrame.from_dict(roc_auc_dict).round(3).set_index("threshold")
    return roc_auc_df

def analyze_our_threshold(ann_df: pd.DataFrame, final_col_name, thresholds=None):
    ra_df = ann_df

    str_thresholds = thresholds
    if thresholds is None: 
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # for each threshold, get hand annotated score
    roc_auc_dict = {"threshold": [], "tn": [], "fp": [], "fn": [], "tp": [], f"{final_col_name}_p": [], f"{final_col_name}_r": [], f"{final_col_name}_f1": []}
    for idx, t in enumerate(thresholds):
        ra_df[f"{t}_thresh"] = ra_df["llm_met_score"].apply(get_thresh_score, thresh=t)

        if ra_df[f"{t}_thresh"].sum() == 0:
            print(f"WE HAVE A PROBLEM: no positive values for threshold {t}")

        cm = confusion_matrix(ra_df["human_metaphor_class"], ra_df[f'{t}_thresh'])
        tn, fp, fn, tp = cm.ravel()

        p = precision_score(ra_df["human_metaphor_class"], ra_df[f'{t}_thresh'])
        r = recall_score(ra_df["human_metaphor_class"], ra_df[f'{t}_thresh'])
        f1 = f1_score(ra_df["human_metaphor_class"], ra_df[f'{t}_thresh'])

        roc_auc_dict["threshold"].append(thresholds[idx])
        roc_auc_dict["tn"].append(tn)
        roc_auc_dict["fp"].append(fp) # number of samples we correctly predicted positive
        roc_auc_dict["fn"].append(fn)
        roc_auc_dict["tp"].append(tp)

        roc_auc_dict[f"{final_col_name}_p"].append(p)
        roc_auc_dict[f"{final_col_name}_r"].append(r)
        roc_auc_dict[f"{final_col_name}_f1"].append(f1)
    
    roc_auc_df = pd.DataFrame.from_dict(roc_auc_dict).round(3).set_index("threshold")
    return roc_auc_df


def add_cm_col(j_df, col, threshold):
    j_df["human_met_class"] = j_df["human_metaphor_score"].apply(get_thresh_score, thresh=threshold)

    # find examples where human met class is neg, but our class is pos
    def get_cm_class(row):
        t_class = row["human_met_class"]
        p_class = row[col]

        if t_class is False and p_class is False:
            return 'tn'
        elif t_class is True and p_class is False:
            return 'fn'
        elif t_class is True and p_class is True:
            return 'tp'
        elif t_class is False and p_class is True:
            return 'fp'

    j_df["cm_class"] = j_df.apply(get_cm_class, axis=1)
    return j_df


def export_cm_examples(j_df, raw_llm_df, our_pred_col, report_type, save_path, our_score_col = None):
    report_df = j_df[j_df["cm_class"] == report_type]
    report_df = report_df.sort_values(by="human_metaphor_score", ascending=True)
    report_dict = {}
    for _, row in report_df.iterrows():
        doc_id = row["id"]

        # gather our results
        our_results = []
        our_res_df = raw_llm_df[raw_llm_df["doc_id"] == doc_id]

        if our_score_col:
            our_res_df = our_res_df.sort_values(["sentence_id", our_score_col], ascending=True)
        else:
            our_res_df = our_res_df.sort_values(["sentence_id", our_pred_col], ascending=True)

        for _, r in our_res_df.iterrows():
            our_results.append({
                "class": r[our_pred_col],
                "score": r[our_score_col] if our_score_col else None,
                "met_path": r["met_path"], 
                "sentence": r["sentence_text"],
                "explanation": r["explanation"]})

        report_dict[doc_id] = {
            "human_metaphor_score": row["human_metaphor_score"],
            "text": row["text"],
            "our_results": our_results
        }
    save_to_json(report_dict, save_path, f"{report_type}.json")



