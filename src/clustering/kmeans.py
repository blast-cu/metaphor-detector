import os
import pickle
import random

import numpy as np
from sklearn.cluster import KMeans

from src.clustering.metrics.purity import Purity
from src.clustering.metrics.cluster_metrics import ClusterMetrics

class KMeansClustering:
    def __init__(self, n_clusters, config, cluster_name, feature_data=None, feature_key=None):
        self.logger = config.logger

        self.n_clusters = n_clusters
        self.cluster_name = cluster_name
        self.cluster_path = f"{config.task_dir}/{config.dataset_out_name}"
        self.feature_data = feature_data
        self.feature_key = feature_key

        self.random_state = config.random_seed
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def kmeans(self, X):
        self.logger.info(f"Clustering with {self.n_clusters} clusters...")
        clustering_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init='k-means++', n_init=10)
        clustering_model.fit(X)
        cluster_assignment = clustering_model.labels_
        cluster_centroids = clustering_model.cluster_centers_

        # Create clustering data for purity and regression
        clustering_data = {
            "number_cluster": self.n_clusters,
            "embeddings": X,
            "labels": cluster_assignment,
            "cluster_centers": cluster_centroids
        }

        # Compute and print purity results
        if self.feature_data is not None and self.feature_key is not None:
            self.logger.info("=== Purity Results ===")
            purity = Purity(self.feature_data, clustering_data, self.feature_key, logger=self.logger)
            purity.compute_purity()
            purity.print_results()

        # Compute and print cluster quality metrics
        self.logger.info("=== Cluster Quality Metrics ===")
        cluster_metrics = ClusterMetrics(clustering_data, logger=self.logger)
        cluster_metrics.print_results()
        self.logger.info("==============================")

        # Save clustering results
        os.makedirs(self.cluster_path, exist_ok=True)
        out_path = self.cluster_path + f"/{self.cluster_name}_clusters_{self.n_clusters}.pkl"
        self.logger.info(f"Saving results to {out_path}...")
        with open(out_path, 'wb') as f:
            pickle.dump(clustering_data, f, protocol=pickle.HIGHEST_PROTOCOL)
