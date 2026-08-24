"""
This script fine-tunes RoBERTa for source word classification using a labeled dataset. Taking a 
sentence as the input, RoBERTa encodes each (sub)token into a dense vector representation; a 
linear classification layer is then applied to the vector representation of the candidate source 
token to predict its metaphoricity on a four-point scale from 0 to 3.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, \
    RobertaConfig, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
import shutil

import torch # for handling memory issues
import gc

import wandb
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import logging
import numpy as np

from src.span_classifier import split_data, prepare_dataset, \
    RobertaSpanForSequenceClassification, DataCollatorForSpanClassification
from src.utils import set_seed, save_to_json, load_env, get_logger, load_json
from src.data_loading import DatasetLoader
from src.experiment_config import ExperimentConfig
from src.db_config import Document, DocAnnotation
from sklearn.model_selection import KFold, train_test_split

def test(trainer, test_tokenized, ds, model_path):
    
    test_results = trainer.predict(test_tokenized)
    print("Test metrics:", test_results.metrics)
    wandb.log(test_results.metrics)

    # # logits, labels = test_results.predictions, test_results.label_ids
    preds = np.argmax(test_results.predictions, axis=-1)
    preds = preds.tolist()
    labels = test_results.label_ids.tolist()

    out_test_results = {}
    out_test_results["results"] = test_results.metrics
    out_test_results["train"] = pd.DataFrame(ds["train"].to_dict()).to_dict(orient='records')
    out_test_results["validation"] = pd.DataFrame(ds["validation"].to_dict()).to_dict(orient='records')

    out_test_results["test"] = pd.DataFrame(ds["test"].to_dict()).to_dict(orient='records')
    for idx, item in enumerate(ds["test"]):
        out_test_results["test"][idx]["prediction"] = preds[idx]

    save_to_json(out_test_results, model_path, "test_results.json")

    # save scikit-learn classification report to csv
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)
    report_df.to_csv(f"{model_path}/test_classification_report.csv", index=True)


def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, predictions),
        "macro_f1": f1_score(p.label_ids, predictions, zero_division=0, average="macro"),
        "weighted_f1": f1_score(p.label_ids, predictions, zero_division=0, average="weighted"), 
        "micro_f1": f1_score(p.label_ids, predictions, zero_division=0, average="micro"), 
        "precision": precision_score(p.label_ids, predictions, zero_division=0, average="macro"),
        "recall": recall_score(p.label_ids, predictions, zero_division=0, average="macro"),
    }

    
def get_documents(session, binary, logger, doc_filter=None):
    documents = session.query(Document).all()
    formatted_docs = []
    skipped = 0

    if doc_filter is not None:
        assert len(doc_filter) > 0, f"doc_filter is of length 0, no data will be collected for training"
        assert isinstance(doc_filter[0], str), f"Expected data to be a string, but got {type(doc_filter[0]).__name__}"

    for doc in tqdm(documents, desc="Formatting documents for classification"):

        if doc_filter is None or doc.id in doc_filter:

            start_span = session.query(DocAnnotation).filter_by(id=f"{doc.id}_source_char_start").one_or_none()
            end_span = session.query(DocAnnotation).filter_by(id=f"{doc.id}_source_char_end").one_or_none()
            source_text = session.query(DocAnnotation).filter_by(id=f"{doc.id}_source_string").one_or_none()
            human_metaphor_score_ann = session.query(DocAnnotation).filter_by(id=f"{doc.id}_human_metaphor_score").one_or_none()
            target_concept = session.query(DocAnnotation).filter_by(id=f"{doc.id}_human_target_concept").one_or_none()

            if start_span is None or end_span is None or source_text is None or \
                human_metaphor_score_ann is None or human_metaphor_score_ann.annotation_value == "-1":
                logger.debug(f"Skipping document {doc.id} due to missing annotations.")
                logger.debug(f"start_span: {start_span}, end_span: {end_span}, source_text: {source_text}, human_metaphor_score_ann: {human_metaphor_score_ann}")

                skipped += 1
                continue
            
            # convert this to a binary task if needed
            label = int(human_metaphor_score_ann.annotation_value)
            if binary: 
                label = 1 if label > 1 else 0

            formatted_docs.append({
                "id": doc.id,
                "text": doc.text,
                "span_token_indices": (int(start_span.annotation_value), int(end_span.annotation_value)),
                "source_text": source_text.annotation_value,
                "label": label,
                "target_concept": target_concept.annotation_value if target_concept is not None else None
            })

    if skipped > 0:
        logger.warning(f"Skipped {skipped} documents due to missing annotations.")

    return formatted_docs

def load_training_documents(args, config, env_vars, model_path, LOGGER):

    document_path = f"{model_path}/lcc_documents.json"
    if args.binary:
        documents_path = documents_path.replace(".json", "_binary.json")

    if args.filter_sdps:
        document_path = document_path.replace(".json", "_filtered.json")

    if os.path.exists(document_path):
        LOGGER.info(f"Loading pre-saved documents from {document_path}...")
        documents = load_json(document_path)
    
    else:
        LOGGER.info("Loading documents from database...")
        data_loader = DatasetLoader(config)
        ds = data_loader.load_preprocessed_data()
        doc_filter = None

        if args.filter_sdps:
            analysis_id_path = f"{env_vars["INTERIM_DIR"]}/lcc_analysis_ids.csv"
            doc_id_df = pd.read_csv(analysis_id_path)
            doc_filter = doc_id_df["doc_id"].astype("str").tolist()

        documents = get_documents(ds, args.binary, LOGGER, doc_filter=doc_filter)
        save_to_json(documents, model_path, document_path.split("/")[-1])
    
    return documents

def create_splits(folds, generalize, documents, seed):

    
    if generalize: # set up the generalizability study
        data_splits = []
        target_concepts = [d["target_concept"] for d in documents]
        target_concepts = list(set(target_concepts))
        for tc in target_concepts:
            test_ids = [item["id"] for item in documents if item["target_concept"] == tc]
            test_split = [item for item in documents if item["id"] in test_ids]
            train_split = [item for item in documents if item["id"] not in test_ids]

            train_split, val_split = train_test_split(train_split, test_size=0.1,random_state=seed)
            data_splits.append((f"{tc}_test", (train_split, val_split, test_split)))
    
    elif folds > 1:
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        X = np.array(documents)
        data_splits = []
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            X_train, X_val = train_test_split(X_train, test_size=0.1,random_state=seed)
            data_splits.append((f"fold_{fold_idx}", (list(X_train), list(X_val), X_test.tolist())))

    else:
        # split into train, val, test
        train_split, val_split, test_split = \
            split_data(documents=documents, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=seed)
        data_splits = [("", (train_split, val_split, test_split))]
    
    return data_splits


def main(args):

    gc.collect()
    torch.cuda.empty_cache()

    # load data
    # set up environment and seed
    MODEL_MAX_LEN = 512

    env_vars = load_env()
    seed = env_vars["RANDOM_SEED"]
    set_seed(seed)

    LOGGER = get_logger("train_src_class", level=logging.INFO)
    LOGGER.info("Starting script to train source classifier...")

    model_path = "data/models"
    output_model_name = "source_classifier"
    if args.binary:
        output_model_name = output_model_name + "_binary"   
    if args.filter_sdps:
        output_model_name = output_model_name + "_filtered"
    if args.generalize:
        output_model_name = output_model_name + "_generalizability_study"

    this_model_path = f"{config.task_dir}/{output_model_name}"
    os.makedirs(this_model_path, exist_ok=True)

    # when using fast tokenizer, must disable parallelism 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load data from xml files
    config = ExperimentConfig(
        task_type="metaphor_classification",
        task_name="roberta",
        dataset="lcc-large",
        logger=LOGGER,
        env_vars=env_vars,
        labeled_only=True,
        device=env_vars["DEVICE"],
        random_seed=seed,
        skip_load=True # SKIPPING LOAD FOR SPEEEED
    )

    # 1) tokenizer & dataset
    model_name = "roberta-base"
    tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    documents = load_training_documents(args, config, env_vars, model_path, LOGGER)
    LOGGER.info(f"Loaded {len(documents)} training documents.")

    # create folds if necessary
    data_splits = create_splits(args.folds, args.generalize, documents, seed)
    LOGGER.info(f"Created {len(data_splits)} splits for training")

    for fold_name, (train_split, val_split, test_split) in data_splits:
    
        # Instead of load_dataset for inline lists, construct Dataset directly:
        train_ds = Dataset.from_list(train_split)
        val_ds = Dataset.from_list(val_split)
        test_ds = Dataset.from_list(test_split)
        ds = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

        tokenized = prepare_dataset(ds["train"], tokenizer, text_col="text", label_col="label", index_col="span_token_indices", max_length=MODEL_MAX_LEN)
        val_tokenized = prepare_dataset(ds["validation"], tokenizer, text_col="text", label_col="label", index_col="span_token_indices", max_length=MODEL_MAX_LEN)
        test_tokenized = prepare_dataset(ds["test"], tokenizer, text_col="text", label_col="label", index_col="span_token_indices", max_length=MODEL_MAX_LEN)

        # model creation
        num_labels = len(list(set([doc["label"] for doc in documents])))
        config = RobertaConfig.from_pretrained(model_name, num_labels=num_labels)
        model = RobertaSpanForSequenceClassification.from_pretrained(model_name, config=config, span_pooling="mean")

        collator = DataCollatorForSpanClassification(tokenizer)

        cur_model_path = this_model_path
        if args.folds > 0 or args.generalize:
            cur_model_path = f"{this_model_path}/{fold_name}"
            os.makedirs(cur_model_path, exist_ok=True)
        progress_model_path = f"{cur_model_path}-progress"
        
        training_args = TrainingArguments(
            output_dir= progress_model_path,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            num_train_epochs=args.train_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            learning_rate=2e-5,
            load_best_model_at_end=True,
            remove_unused_columns=False,  # important when model expects custom inputs like span_indices,
            seed=seed,
            report_to="wandb"  # Enables Weights & Biases integration
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            eval_dataset=val_tokenized,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics
        )

        # 4) Train!
        trainer.train()
        test(trainer, test_tokenized, ds, cur_model_path)
        
        # Save final model
        trainer.save_model(cur_model_path)
        
        # clean up
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache() # If using GPU, helps with overall system stability

        # remove progress models in directory
        try:
            shutil.rmtree(progress_model_path)
            LOGGER.info(f"Directory '{progress_model_path}' and all its contents have been removed.")
        except OSError as e:
            LOGGER.warning(f"Error: {progress_model_path} : {e.strerror}")


    # if doing folds, need to save a complete classification report
    if args.folds > 1 or args.generalize:
        all_preds = []
        all_labels = []
        for fold_name, _ in data_splits:
            fold_res_path = f"{this_model_path}/{fold_name}/test_results.json"
            fold_test_res = load_json(fold_res_path)["test"]
            preds = [r["prediction"] for r in fold_test_res]
            labels = [r["label"] for r in fold_test_res]

            all_preds.extend(preds)
            all_labels.extend(labels)
        
        # after they're all gathered, export classification report
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(3)
        report_df.to_csv(f"{this_model_path}/test_classification_report.csv", index=True)

        # also a binary classification report
        bin_labels = [0 if l == 0 else 1 for l in labels]
        bin_preds = [0 if p == 0 else 1 for p in preds]
        bin_report = classification_report(bin_labels, bin_preds, output_dict=True, zero_division=0)
        bin_report_df = pd.DataFrame(bin_report).transpose().round(3)
        bin_report_df.to_csv(f"{this_model_path}/test_binary_classification_report.csv", index=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use a rb approach to extract all sentences which could invoke a metaphor.")
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=50,
        help="Number of epochs to train."
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=0,
        help="Number of folds for 5-fold cross validation"
    )
    parser.add_argument(
        "--binary",
        action='store_true',
        help="Whether to convert task to binary metaphor classification."
    )
    parser.add_argument(
        "--filter_sdps",
        action="store_true",
        help="Train and test on same set as us"
    )
    parser.add_argument(
        "--generalize",
        action="store_true"
    )
    args = parser.parse_args()
    main(args)