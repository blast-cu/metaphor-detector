import json
import os
import argparse
import logging
import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from src.utils import get_logger, load_env, load_json, save_to_json, get_token, VERB_PATHS
from src.experiment_config import ExperimentConfig
from analysis.utils import get_target_coref, get_path_info
from src.data_loading import DatasetLoader

"""
This script...
"""


class KmeansClusterer():

    def __init__(self, data: list[str], logger: logging.Logger):

        data = [d.lower().strip() for d in data]
        data = list(set(data))  # remove duplicates
        logger.info(f"{len(data)} words after removing duplicates")

        self.vectorizer = TfidfVectorizer()
        self.data = data
        self.vectors = self.vectorizer.fit_transform(data)
        self.logger = logger

    def fit(self, k):
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=14) # Specify the number of clusters
        kmeans.fit(self.vectors)
        labels = kmeans.labels_
        inertia = kmeans.inertia_
        return labels, inertia
    
    def clean_labels(self, labels):
        cluster_dict = {}
        for i, label in enumerate(labels):
            current_data = self.data[i]
            label = str(label)
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(current_data)
        
        for cluster, item in cluster_dict.items():  # sort alphabetically
            cluster_dict[cluster] = sorted(item)

        # sort keys
        cluster_dict = dict(sorted(cluster_dict.items(), key=lambda item: int(item[0])))
        return cluster_dict


    def run(self, k):
        labels, inertia = self.fit(k)
        self.cluster_dict = self.clean_labels(labels)
        return inertia

    def print_clusters(self):
        for cluster, item in self.cluster_dict.items():
            print(f">> Cluster {cluster}")
            for d in item:
                print(d)
            print("\n")

    def save_clusters(self, file_path: str, file_name: str, add_notes: bool = False):

        out_cluster_dict = self.cluster_dict
        if add_notes: 
            out_cluster_dict = {}
            for k, v in self.cluster_dict.items():
                out_cluster_dict[k] = {
                    "cluster_title": "",
                    "notes": "",
                    "words": v
                }
        save_to_json(out_cluster_dict, file_path, file_name)


class KmeansSeededClusterer(KmeansClusterer):
    """
    Performs seeded clustering
    """
    def __init__(self, data: list[str], initial_centers: list[str]):
        super().__init__(data)
        self.k = len(initial_centers)
        self.initial_centers = self.vectorizer.transform(initial_centers)
        self.initial_centers = self.initial_centers.toarray()

    def fit(self):
        kmeans = KMeans(n_clusters=self.k, init=self.initial_centers, random_state=14)
        kmeans.fit(self.vectors)
        labels = kmeans.labels_
        return labels

    def run(self):
        labels = self.fit()
        self.cluster_dict = super().clean_labels(labels)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(
        description="Analyze a dataset using Kmeans clustering."
    )
    parser.add_argument(
        "--dataset", type=str,
        help="dataset to cluster"
    )
    parser.add_argument(
        "--file_path", type=str,
        help="Path to the file of results to analyze",
    )
    parser.add_argument(
        "--k", type=int,
        default=0,
        help="Number of clusters.",
    )
    parser.add_argument(
        "--elbow", 
        action="store_true",
        help="Whether to do elbow analysis.",
    )
    parser.add_argument(
        "--source", 
        action="store_true"
    )
    parser.add_argument(
        "--target", 
        action="store_true"
    )
    parser.add_argument(
        "--add_notes", 
        action="store_true",
        help="If set, adds notes section to output json for annotations."
    )
    args = parser.parse_args()        

    ### SET UP ENVIRONMENT VARIABLES AND LOGGER
    LOGGER = get_logger(f"clustering", level=logging.INFO)
    env_vars = load_env(logger=LOGGER)

    if not args.source and not args.target:
        LOGGER.error("Must either specify --source or --target flag")
    elif args.source and args.target:
        LOGGER.error("Must either specify --source or --target flag, can't set both")

    if args.k == 0 and not args.elbow:
        LOGGER.error("Must either specify k value or set elbow method")
    elif args.k > 0 and args.elbow:
        LOGGER.error("Pick a lane")


    LOGGER.info("Starting script to run LLM annotator...")

    REPO_PATH = env_vars["REPO_PATH"]
    RAW_DATA_DIR = env_vars["RAW_DATA_DIR"]
    DB_DIR = env_vars["DB_DIR"]
    RANDOM_SEED = env_vars["RANDOM_SEED"]
    random.seed(RANDOM_SEED)
    DEVICE = env_vars["DEVICE"]

    # TODO: fix hard-coding for tweets here
    data = load_json(args.file_path)

    # filter based on threshold
    to_cluster = []
    word_type = "target" if args.target else "source"
    for k, v in tqdm(data.items(), desc="gathering metaphorical data"):
        if args.target: # need coreference
            to_cluster.append(v["target_rep_text"])
        
        else:
            to_cluster.append(v["source"]["text"])             
        
    
    clusterer = KmeansClusterer(to_cluster, LOGGER)
    out_path = "results/clusters"
    os.makedirs(out_path, exist_ok=True)
    
    if args.k > 0:
        LOGGER.info(f"generating clusters with k = {args.k}")
        # call class
        clusterer.run(k=args.k)
        # clusterer.print_clusters()

        clusterer.save_clusters(f"{out_path}/", f"{args.dataset}_{word_type}_k{args.k}.json", args.add_notes)
    
    else: # eval with elbow method
        inertias = []
        k_range = range(1, 31)
        for k in tqdm(k_range, desc="performing elbow test"):
            inertia = clusterer.run(k=k)
            inertias.append(inertia)

        # plot results
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, inertias, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia (WCSS)')
        plt.grid(True)
        plt.show()


