import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix


class Purity:
    def __init__(self, data, clustering_data, feature_key, logger=None):
        """
        Args:
            data: Embeddings/features dict keyed by str(i); each entry has a 'features' dict
            clustering_data: Clustering results dict with 'labels', 'embeddings',
                             'cluster_centers', 'number_cluster'
            feature_key: Feature key string or list of feature key strings to evaluate purity on
            logger: Logger instance
        """
        self.data = data
        self.clustering_data = clustering_data
        self.feature_keys = [feature_key] if isinstance(feature_key, str) else list(feature_key)
        self.logger = logger

        self.results = {
            "25": {"exact_match_purity": None, "feature_purity": None},
            "100": {"exact_match_purity": None, "feature_purity": None},
        }

    def _get_top_k_indices(self, top_k):
        """
        For each cluster, return (global_idx, cluster_label) for the top_k% of points
        closest to the cluster centroid.
        """
        clusters = {i: [] for i in range(self.clustering_data["number_cluster"])}
        for idx, label in enumerate(self.clustering_data["labels"]):
            clusters[label].append(idx)

        selected = []
        for cluster_idx, members in clusters.items():
            cluster_embs = self.clustering_data["embeddings"][members]
            centroid = self.clustering_data["cluster_centers"][cluster_idx]
            distances = np.linalg.norm(cluster_embs - centroid, axis=1)
            k = max(1, int(len(members) * top_k / 100))
            closest_local = np.argsort(distances)[:k]
            for local_idx in closest_local:
                selected.append((members[local_idx], cluster_idx))

        return selected

    def get_exact_match_purity(self, selected):
        """
        Exact match purity: a point's "class" is the full combination of all feature_key values.
        Two points match only when every specified feature value is identical.

        Args:
            selected: list of (global_idx, cluster_label) tuples
        Returns:
            float purity score in [0, 1]
        """
        log = self.logger.info if self.logger else print

        def canonicalize(idx):
            features = self.data[str(idx)].get("features", {})
            return tuple(sorted((k, features[k]) for k in self.feature_keys if k in features))

        unique_classes = list(set(canonicalize(idx) for idx, _ in selected))
        class_mapping = {cls: i for i, cls in enumerate(unique_classes)}

        true_labels = [class_mapping[canonicalize(idx)] for idx, _ in selected]
        predicted_labels = [cluster_label for _, cluster_label in selected]

        cm = confusion_matrix(true_labels, predicted_labels)
        purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)

        log(f"Unique feature combinations: {len(unique_classes)}")
        log(f"Clusters: {len(set(predicted_labels))}")
        log(f"Exact match purity: {purity * 100:.2f}%")

        return purity

    def get_feature_purity(self, selected):
        """
        Feature-based purity: for each feature key, compute the average per-cluster purity
        (fraction of points in the cluster that share the most common value for that feature).

        Args:
            selected: list of (global_idx, cluster_label) tuples
        Returns:
            (dict of {feature_key: purity_score}, overall_purity float)
        """
        log = self.logger.info if self.logger else print

        cluster_data = {}
        for idx, cluster_label in selected:
            features = self.data[str(idx)].get("features")
            if features is not None:
                cluster_data.setdefault(cluster_label, []).append(features)

        total = len(selected)
        feature_purities = {}
        for key in self.feature_keys:
            majority_counts = []
            for points in cluster_data.values():
                values = [p[key] for p in points if key in p]
                if values:
                    most_common_count = Counter(values).most_common(1)[0][1]
                    majority_counts.append(most_common_count)
            if majority_counts:
                feature_purities[key] = float(sum(majority_counts) / total)

        overall_purity = float(np.mean(list(feature_purities.values()))) if feature_purities else 0.0

        log("Feature-based purity:")
        for key, score in feature_purities.items():
            log(f"  {key}: {score * 100:.2f}%")
        log(f"  Overall: {overall_purity * 100:.2f}%")

        return feature_purities, overall_purity

    def compute_purity(self):
        log = self.logger.info if self.logger else print
        for k in [25, 100]:
            log(f"--- Purity @ top {k}% ---")
            selected = self._get_top_k_indices(top_k=k)
            exact = self.get_exact_match_purity(selected)
            feature_purities, overall = self.get_feature_purity(selected)
            self.results[str(k)]["exact_match_purity"] = exact
            self.results[str(k)]["feature_purity"] = {
                "key_purities": feature_purities,
                "overall_purity": overall,
            }

    def print_results(self):
        log = self.logger.info if self.logger else print
        log("=== PURITY RESULTS ===")
        for k in [25, 100]:
            log(f"--- Top {k}% ---")
            r = self.results[str(k)]
            if r["exact_match_purity"] is not None:
                log(f"Exact Match Purity: {r['exact_match_purity'] * 100:.2f}%")
            if r["feature_purity"] is not None:
                fp = r["feature_purity"]
                for key, score in fp["key_purities"].items():
                    log(f"{key} Purity: {score * 100:.2f}%")
                log(f"Overall Feature Purity: {fp['overall_purity'] * 100:.2f}%")
        log("======================")
