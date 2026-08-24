# Discovering Conceptual Metaphors Across Topics and Media Types

Code for the paper of the same name. This repository contains the code and experiments for discovering conceptual metaphors in online political discourse using unsupervised methods. The pipeline extracts candidate linguistic metaphor relations, classifies them using LLMs, and clusters them to identify pervasive conceptual metaphors in a discourse.

full paper: https://arxiv.org/abs/2608.06652

---

# Installation

1. Clone the repository.

```bash
git clone <repo-url>
cd <repo-name>
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create a `.env` file defining the following environment variables:

```bash
PYTHONPATH=/path/to/repo
export REPO_PATH=/path/to/repo
export RAW_DATA_DIR=/path/to/raw/data/storage
export DB_DIR=/path/to/store/generated/db
export INTERIM_DIR=/path/to/interim/data/storage
export RESULTS_DIR=/path/to/results/dir
export RANDOM_SEED=42
```

---

# Data

Preprocessed database downloads will be added in a future release.

## tweets-immigration

Immigration discourse dataset used for the primary case study.

Paper:  
https://aclanthology.org/2025.acl-long.398/

Dataset:  
https://aclanthology.org/2025.acl-long.398/

---

## subframes (immigration, gun control, abortion)

Frame-annotated political discourse datasets.

Paper:  
https://aclanthology.org/2020.emnlp-main.620/

Repository:  
https://github.com/ShamikRoy/Subframe-Prediction

---

## LCC Metaphor Dataset

Used for training and evaluating metaphor classifiers.

Paper:  
https://aclanthology.org/L16-1668/

Repository:  
https://github.com/lcc-api/metaphor

Dataset:  
https://github.com/lcc-api/metaphor/blob/main/LCC_Metaphor_Dataset.large.tar.gz

---

# Metaphor Extraction Pipeline

The full pipeline consists of the following stages:

### Extract Candidate Metaphor Paths
Generate candidate source–target metaphor pairs.

```
experiments/get_candidate_metaphor_paths.py
```

---

### Score Candidate Metaphors with an LLM
Classify metaphoricity using prompting.

```
experiments/llm_metaphor_classifier.py
```

---

### Export High-Confidence Metaphors
Filter metaphor candidates using a score threshold.

```
scripts/export_metaphors.py
```

---

### Source Domain Grouping

1. Export embeddings of LLM explanation texts

```
experiments/export_metaphor_embeddings.py
```

2. Cluster embeddings to identify source domains.

---

### Target Domain Grouping

1. Cluster target nouns

```
scripts/cluster.py
```

2. Identify noun types (person/place/thing)

```
experiments/llm_feature_extractor.py
```

3. Extract type-specific semantic features

```
experiments/llm_feature_extractor.py
```



---

## Constrained K-Means Clustering

Constrained k-means (PCKMeans) incorporates cannot-link constraints derived from target group features, preventing items with different target groups from being assigned to the same cluster.

#### Step 1: Generate Cannot-Link Constraints

Constraints are computed once and cached to disk as LMDB databases. This step is separate from clustering since constraint generation is O(n²) and should not be repeated across experiments.

Requires a features file produced by the embeddings export step (e.g. `{dataset}_{embedding_name}_embeddings_features.pkl`).

```bash
python -m src.clustering.constraint_generator \
    --dataset tweets_immigration \
    --labeled_only \
    --embedding_model sbert \
    --embedding_name mets_only_qwen_expl \
    --feature_key target_group
```

Output databases are written to:
```
$RESULTS_DIR/constraints/sbert/{dataset}_{embedding_name}_{feature_key}_flat.db
$RESULTS_DIR/constraints/sbert/{dataset}_{embedding_name}_{feature_key}_graph.db
```

#### Step 2: Run Constrained Clustering

`--feature_key` is used both for loading the pre-computed constraint databases and for purity evaluation.

```
w_cl = 0.01, 0.05, 0.1
```

```bash
python -m experiments.cluster_embeddings \
    --use_pckmeans \
    --k $k \
    --dataset tweets_immigration \
    --labeled_only \
    --embedding_model sbert \
    --embedding_name mets_only_qwen_expl \
    --feature_key target_group \
    --w_cl $w_cl
```

---

### Predict Source Domains Using Cluster Features

Metaphor-level features:

```bash
python -m experiments.cluster_lr \
    --dataset tweets_immigration \
    --labeled_only \
    --label concept \
    --met_embedding_name mets_only_qwen_expl \
    --metaphors_only \
    --predict_over_met_docs
```

Tweet-level features:

```bash
python -m experiments.cluster_lr \
    --dataset tweets_immigration \
    --labeled_only \
    --label concept \
    --tweet_embedding_name full_tweet \
    --predict_over_met_docs
```

---

### Combined Features Evaluation

Train models using **all combinations of k** for both feature types.

```bash
python -m experiments.cluster_lr \
    --dataset tweets_immigration \
    --labeled_only \
    --label concept \
    --tweet_embedding_name full_tweet \
    --met_embedding_name mets_only_qwen_expl \
    --metaphors_only \
    --metaphor_thresh 0.3 \
    --predict_over_met_docs
```

---

### Evaluate All Configurations

Compute Calinski-Harabasz, exact-match purity (top-25% and all points), and logistic regression metrics (train/test accuracy and F1) for every combination of k and w_cl in a single pass. Outputs a CSV to `$RESULTS_DIR/eval/{dataset}_{embedding_name}_cluster_eval.csv`.

```bash
python -m eval.cluster_eval \
    --dataset tweets_immigration \
    --labeled_only \
    --embedding_model sbert \
    --embedding_name mets_only_qwen_expl \
    --feature_key target_group \
    --label concept \
    --predict_over_met_docs \
    --k_list 25 50 75 100 125 150 175 200 225 250 275 300 \
    --w_cl_list 0.01 0.05 0.1
```

Requires KMeans and PCKMeans cluster files to already exist on disk (run `experiments.cluster_embeddings` first). To evaluate a different dataset, update `--dataset`, `--embedding_name`, `--label`, and `--feature_key` accordingly.




# Replicating Metaphor Detection Results

### Generate Candidate Metaphor Pairs (LCC Dataset)

```bash
python -m experiments.get_candidate_metaphor_paths \
    --dataset lcc-large \
    --labeled_only \
    --skip_load
```

---

### Binary Metaphor Classification

```bash
python -m experiments.llm_metaphor_classifier \
    --dataset lcc-large \
    --skip_load \
    --labeled_only \
    --out_file lcc_source_verb_binary_qwen.json \
    --prompt_file source_verb_bin_class.json \
    --host $host_ip \
    --port $port \
    --annotation_config qwen3.yaml \
    --shots 0
```

---

### RoBERTa 5-fold Setting

```bash
python -m experiments.train_source_classifier \
    --filter_sdps \
    --train_epochs 20 \
    --folds 5
```

---

### RoBERTa Out-of-Domain Setting

```bash
python -m experiments.train_source_classifier \
    --filter_sdps \
    --train_epochs 20 \
    --generalize
```

---

### Evaluation

Reproduce evaluation metrics in:

```
eval/binary_metaphor_eval.ipynb
```


## Cite Us!

If you find this repo useful, please cite us in your work as: 

```bash
@misc{leto2026discoveringconceptualmetaphorstopics,
      title={Discovering Conceptual Metaphors Across Topics and Media Types}, 
      author={Alexandria Leto and Rohan Das and Juan Vásquez and Abram Handler and Maria Leonor Pacheco},
      year={2026},
      eprint={2608.06652},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.06652}, 
}
```