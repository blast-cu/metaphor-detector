import pandas as pd
import numpy as np
from sklearn.metrics import \
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from src.db_config import DocAnnotation

# LCC SPECIFIC
# match sentence scores
def format_lcc_llm_res_df_for_analysis(lcc_session, llm_df):
    "get human annotations for each llm result (SDP level)"
    def get_document_score(doc_id):
        val = lcc_session.query(DocAnnotation).filter_by(id=f"{doc_id}_human_metaphor_score").one_or_none().annotation_value
        return int(val)

    def get_source_str(doc_id):
        source_str = lcc_session.query(DocAnnotation).filter_by(id=f"{doc_id}_source_string").one_or_none().annotation_value
        return source_str.lower().strip()
    
    def get_target_str(doc_id):
        target_str = lcc_session.query(DocAnnotation).filter_by(id=f"{doc_id}_target_string").one_or_none().annotation_value
        return target_str.lower().strip()

    def get_target_domain(doc_id):
        target_domain = lcc_session.query(DocAnnotation).filter_by(id=f"{doc_id}_human_target_concept").one_or_none()
        if not target_domain: return np.nan
        return target_domain.annotation_value.lower().strip()

    # create a dataframe of the hand-annotated data (this should be bigger than our df)
    doc_df = pd.DataFrame()
    doc_df["doc_id"] = llm_df["doc_id"].unique()
    doc_df["human_met_score"] = doc_df["doc_id"].apply(get_document_score)
    doc_df["human_source_str"] = doc_df["doc_id"].apply(get_source_str)
    doc_df["human_target_str"] = doc_df["doc_id"].apply(get_target_str)
    doc_df["human_target_domain"] = doc_df["doc_id"].apply(get_target_domain)

    # merge with llm df for analysis
    joined_df = llm_df.merge(doc_df, on="doc_id", how="left")

    # check that the source and target strings match
    def pairs_match(row):
        assert row["target_word"] == row["human_target_str"], f"target strings don't match '{row["target_word"]}' != '{row["human_target_str"]}'"
        assert row["source_word"] == row["human_source_str"], f"source strings don't match '{row["source_word"]}' != '{row["human_source_str"]}'"
        return
    
    joined_df.apply(pairs_match, axis=1)

    expected_cols = ["doc_id", "human_met_score", "llm_met_class", "human_target_domain"]
    joined_df = joined_df[expected_cols]

    # drop entries with -1 score
    joined_df = joined_df[joined_df["human_met_score"] != -1]

    return joined_df

def get_thresh_score(human_score, thresh):
        return human_score >= thresh

def thresholds_f1_table(ann_df: pd.DataFrame, class_col: str, final_col_name: str = "", thresholds: list = None):
    """
    Calculates f1 score of our annotations using the mend ones as true values. Uses thresholds in increments
    of 0.1 to classify the mend hand annotations as pos/neg.
    
    :param ann_df: df with mend hand annotations and out binary metaphor classifications
    :param class_col: column name of our classifications
    :param final_col_name: label for the final metric columns
    """
    ra_df = ann_df

    str_thresholds = thresholds
    if thresholds is None: 
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        str_thresholds =[f"{int(t * 100)}%" for t in thresholds]

    # for each threshold, get hand annotated score
    f1_dict = {"threshold": [], "tn": [], "fp": [], "fn": [], "tp": [], f"{final_col_name}_p": [], f"{final_col_name}_r": [], f"{final_col_name}_f1": []}
    for idx, t in enumerate(thresholds):
        ra_df[f"{t}_thresh"] = ra_df["human_metaphor_score"].apply(get_thresh_score, thresh=t)

        if ra_df[f"{t}_thresh"].sum() == 0:
            print(f"WE HAVE A PROBLEM: no positive values for threshold {t}")

        cm = confusion_matrix(ra_df[f'{t}_thresh'], ra_df[class_col])
        tn, fp, fn, tp = cm.ravel()

        f1_dict["threshold"].append(str_thresholds[idx])
        f1_dict["tn"].append(tn)
        f1_dict["fp"].append(fp) # number of samples we correctly predicted positive
        f1_dict["fn"].append(fn)
        f1_dict["tp"].append(tp)

        f1_dict[f"{final_col_name}_p"].append(precision_score(ra_df[f'{t}_thresh'], ra_df[class_col])*100)
        f1_dict[f"{final_col_name}_r"].append(recall_score(ra_df[f'{t}_thresh'], ra_df[class_col])*100)
        f1_dict[f"{final_col_name}_f1"].append(f1_score(ra_df[f'{t}_thresh'], ra_df[class_col])*100)


    roc_auc_df = pd.DataFrame.from_dict(f1_dict).round(2).set_index("threshold")
    return roc_auc_df