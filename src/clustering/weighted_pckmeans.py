import os
import pickle
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, triu
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.clustering.initializer.base_initializer import BaseInitializer
from src.clustering.initializer.cl_kmeans_plus_plus import KMeansPlusPlusInit, InitializationStrategy
from src.clustering.metrics.purity import Purity
from src.clustering.metrics.cluster_metrics import ClusterMetrics
from src.experiment_config import ExperimentConfig


@dataclass
class ClusteringMetrics:
    """Metrics for tracking clustering performance"""

    inertia: float  # Sum of squared distances to centers
    constraint_violations: int  # Number of constraint violations
    total_cost: float  # Combined objective function value
    violated_constraints: List[Tuple[int, int]]  # List of violated constraints

class ConstrainedKMeans:
    """
    Implementation of KMeans clustering with Pairwise Cannot-Link constraints
    """

    def __init__(self,
                 n_clusters: int,
                 initializer: BaseInitializer = None,
                 w_cl: float = 1.0,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 early_stopping_tol: int = 10,
                 random_state: Optional[int] = None,
                 centroid_percentile: Optional[float] = None,
                 pairwise_percentile: Optional[float] = None):
        """
        Initialize the Pairwise Constrained KMeans algorithm

        Args:
            n_clusters: Number of clusters
            initializer: Cluster center initializer
            w_cl: Weight for cannot-link constraints
            max_iter: Maximum number of iterations
            tol: Convergence tolerance for centroid movement
            early_stopping_tol: Number of iterations with no improvement before early stopping
            random_state: Random seed
            centroid_percentile: Percentile threshold for distance to cluster centers (e.g., 25 for top 25%)
            pairwise_percentile: Percentile threshold for pairwise distances (e.g., 10 for bottom 10%)
        """

        self.n_clusters = n_clusters
        self.initializer = initializer
        self.w_cl = w_cl
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping_tol = early_stopping_tol
        self.random_state = random_state
        self.centroid_percentile = centroid_percentile
        self.pairwise_percentile = pairwise_percentile
        self.logger = None

        # Attributes that will be set during fitting
        self.cluster_centers_: Optional[npt.NDArray[np.float64]] = None
        self.labels_: Optional[npt.NDArray[np.int64]] = None
        self.n_iter_: int = 0
        self.history_: List[ClusteringMetrics] = []

        # Enhanced tracking attributes
        self.violation_counts: Dict[Tuple[int, int], int] = {}
        self.violations_per_iteration: List[int] = []
        self.persistent_violations: List[Tuple[int, int]] = []
        self.all_violations_history: List[List[Tuple[int, int]]] = []

    def _compute_distances(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute distances between points and cluster centers"""

        return np.array([
            np.sum((X - center) ** 2, axis=1)
            for center in self.cluster_centers_
        ])
    
    def _compute_dual_thresholds(self, X: npt.NDArray[np.float64], assignments: npt.NDArray[np.int64]) -> Tuple[float, float]:
        """Compute global centroid and pairwise distance thresholds using efficient numpy operations"""
        
        # Compute global centroid threshold
        centroid_threshold = None
        if self.centroid_percentile is not None:
            # Compute distance from each point to its assigned cluster center
            assigned_centers = self.cluster_centers_[assignments]  # (n_samples, n_features)
            distances_to_assigned_centers = np.linalg.norm(X - assigned_centers, axis=1)
            centroid_threshold = np.percentile(distances_to_assigned_centers, self.centroid_percentile)
        
        # Compute global pairwise threshold
        pairwise_threshold = None
        if self.pairwise_percentile is not None:
            # Use vectorized pairwise distance computation on a sample
            n_samples = min(1000, len(X))  # Limit for performance
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            sample_X = X[sample_indices]
            
            # Vectorized pairwise distance computation
            # Broadcasting: (n, 1, d) - (1, n, d) -> (n, n, d)
            diff = sample_X[:, np.newaxis, :] - sample_X[np.newaxis, :, :]
            # Compute squared distances: (n, n)
            squared_distances = np.sum(diff ** 2, axis=2)
            # Take square root and get upper triangular part (avoid duplicates)
            distances = np.sqrt(squared_distances)
            upper_triangle = np.triu_indices(n_samples, k=1)
            pairwise_distances = distances[upper_triangle]
            
            pairwise_threshold = np.percentile(pairwise_distances, self.pairwise_percentile)
        
        return centroid_threshold, pairwise_threshold

    def _get_constraint_eligible_items(self, X: npt.NDArray[np.float64], assignments: npt.NDArray[np.int64], centroid_threshold: Optional[float]) -> npt.NDArray[np.bool_]:
        """Return a boolean mask of items eligible for constraint enforcement."""
        if centroid_threshold is not None:
            assigned_centers = self.cluster_centers_[assignments]
            distances_to_assigned_centers = np.linalg.norm(X - assigned_centers, axis=1)
            return distances_to_assigned_centers <= centroid_threshold
        else:
            return np.ones(len(X), dtype=np.bool_)

    @staticmethod
    def _build_constraint_graph(n_samples: int,
                            cl_constraints: List[Tuple[int, int]]) -> Dict[int, Set[int]]:
        """Build constraint graph for efficient lookup"""

        constraint_graph = {i: set() for i in range(n_samples)}
        
        # Pre-sort constraints once to avoid repeated sorting
        sorted_constraints = []
        
        for i, j in tqdm(cl_constraints):
            # Ensure i < j ordering consistently
            if i > j:
                sorted_constraints.append((int(j), int(i)))
            else:
                sorted_constraints.append((int(i), int(j)))
            
            # Add to constraint graph
            constraint_graph[int(i)].add(int(j))
            constraint_graph[int(j)].add(int(i))

        return constraint_graph, sorted_constraints

    def _find_violations(self, X: npt.NDArray[np.float64], assignments: npt.NDArray[np.int64],
                       eligible_items: npt.NDArray[np.bool_], pairwise_threshold: Optional[float]) -> List[Tuple[int, int]]:
        """Find all violated constraints, filtered by dual threshold eligibility."""
        ii, jj = self._cl_ii, self._cl_jj
        mask = (assignments[ii] == assignments[jj]) & eligible_items[ii] & eligible_items[jj]
        if pairwise_threshold is not None:
            cand_ii, cand_jj = ii[mask], jj[mask]
            pw = np.linalg.norm(X[cand_ii] - X[cand_jj], axis=1)
            mask2 = pw <= pairwise_threshold
            return list(zip(cand_ii[mask2].tolist(), cand_jj[mask2].tolist()))
        return list(zip(ii[mask].tolist(), jj[mask].tolist()))

    def _count_violations(self,
                         X: npt.NDArray[np.float64],
                         assignments: npt.NDArray[np.int64],
                         eligible_items: npt.NDArray[np.bool_],
                         pairwise_threshold: Optional[float]) -> Tuple[int, List[Tuple[int, int]]]:
        """Count number of constraint violations and return list of violated constraints"""

        violations = self._find_violations(X, assignments, eligible_items, pairwise_threshold)
        return len(violations), violations

    def _compute_metrics(self,
                        X: npt.NDArray[np.float64],
                        assignments: npt.NDArray[np.int64]) -> ClusteringMetrics:
        """Compute clustering metrics"""

        # Calculate inertia
        distances = self._compute_distances(X)
        min_distances = np.min(distances, axis=0)
        inertia = np.sum(min_distances)

        # Only check violations if constraints have weight
        if self.w_cl > 0:
            # Compute thresholds once
            centroid_threshold, pairwise_threshold = self._compute_dual_thresholds(X, assignments)
            
            # Get eligible items based on centroid threshold
            eligible_items = self._get_constraint_eligible_items(X, assignments, centroid_threshold)
            
            num_violations, violated_constraints = self._count_violations(X, assignments, eligible_items, pairwise_threshold)
        else:
            num_violations = 0
            violated_constraints = []

        # Calculate total cost
        total_cost = inertia + self.w_cl * num_violations

        return ClusteringMetrics(inertia, num_violations, total_cost, violated_constraints)

    def _assign_points(self,
                      X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Assign points to clusters considering cannot-link constraints with dual thresholds"""

        n_samples = X.shape[0]
        assignments = np.zeros(n_samples, dtype=np.int64)

        # Calculate base distances to all centers
        distances = self._compute_distances(X)

        # Add small epsilon to prevent exactly equal distances
        distances += np.random.uniform(0, 1e-10, distances.shape)
        
        if self.w_cl == 0:
            return np.argmin(distances, axis=0)

        initial_assignments = np.argmin(distances, axis=0)
        centroid_threshold, pairwise_threshold = self._compute_dual_thresholds(X, initial_assignments)
        eligible_items = self._get_constraint_eligible_items(X, initial_assignments, centroid_threshold)

        for i in range(n_samples):
            cluster_costs = distances[:, i].copy()
            for j in self._ci[self._cp[i]:self._cp[i + 1]]:
                if j < i and eligible_items[i] and eligible_items[j]:
                    if pairwise_threshold is None or np.linalg.norm(X[i] - X[j]) <= pairwise_threshold:
                        cluster_costs[assignments[j]] += self.w_cl
            assignments[i] = np.argmin(cluster_costs)

        return assignments

    def _update_centers(self,
                       X: npt.NDArray[np.float64],
                       assignments: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Update cluster centers"""

        new_centers = np.zeros_like(self.cluster_centers_)

        for k in range(self.n_clusters):
            mask = (assignments == k)
            if np.any(mask):
                new_centers[k] = np.mean(X[mask], axis=0)
            else:
                # If cluster is empty, keep old center
                new_centers[k] = self.cluster_centers_[k]

        return new_centers

    def _update_violation_statistics(self, violated_constraints: List[Tuple[int, int]]):
        """Update violation tracking statistics"""

        # Skip violation tracking if constraints have no weight
        if self.w_cl == 0:
            return
        
        # Use sorted tuples instead of frozensets (~3x smaller per key)
        violated_set = {(min(a, b), max(a, b)) for a, b in violated_constraints}

        self.violations_per_iteration.append(len(violated_set))

        # For persistent violations tracking, use set operations
        if not hasattr(self, 'potential_persistent_violations'):
            self.potential_persistent_violations = violated_set
        else:
            self.potential_persistent_violations &= violated_set

        for constraint in violated_set:
            self.violation_counts[constraint] = self.violation_counts.get(constraint, 0) + 1

    def fit(self,
            X: npt.NDArray[np.float64],
            constraint_matrix: Optional[csr_matrix] = None,
            skip_init: bool = False) -> 'ConstrainedKMeans':
        """
        Fit the Constrained KMeans clustering model.

        Args:
            X: Training data
            constraint_matrix: Symmetric CSR boolean matrix encoding cannot-link pairs
            skip_init: If True, use existing cluster centers for initialization
        """
        if skip_init:
            if self.cluster_centers_ is None:
                raise ValueError("Custom initialization requires cluster_centers_ to be set")
        else:
            self.logger.info("Initializing cluster centers...")
            self.cluster_centers_ = self.initializer.initialize(
                X, self.n_clusters, constraint_matrix, self.random_state
            )

        self.violation_counts = {}
        self.violations_per_iteration = []
        self.all_violations_history = []

        # Build fast lookup structures from the CSR matrix
        if constraint_matrix is not None and self.w_cl > 0:
            cm = constraint_matrix.tocsr()
            self._ci = cm.indices.copy()   # neighbor index array
            self._cp = cm.indptr.copy()    # row pointer array
            upper = triu(cm, k=1).tocsr()
            self._cl_ii, self._cl_jj = upper.nonzero()  # unique constraint pairs
        else:
            self._ci = np.empty(0, dtype=np.int32)
            self._cp = np.zeros(X.shape[0] + 1, dtype=np.int32)
            self._cl_ii = np.empty(0, dtype=np.int32)
            self._cl_jj = np.empty(0, dtype=np.int32)

        # Initialize history and tracking variables
        self.history_ = []
        best_cost = float('inf')
        best_centers = None
        best_labels = None
        iterations_without_improvement = 0

        # Main clustering loop
        self.logger.info("Starting clustering iterations...")
        for iteration in tqdm(range(self.max_iter)):
            # Get new assignments
            new_assignments = self._assign_points(X)
            
            # Update centers
            new_centers = self._update_centers(X, new_assignments)
            
            # Compute metrics
            metrics = self._compute_metrics(X, new_assignments)
            self.history_.append(metrics)
            
            # Log constraint violations for this iteration
            self.logger.info(f"Iteration {iteration + 1}: Constraint violations = {metrics.constraint_violations}")

            # Update violation statistics only if needed
            if self.w_cl > 0:
                self._update_violation_statistics(metrics.violated_constraints)
            # Free the per-iteration violation list; count is preserved in history
            metrics.violated_constraints = []
            
            # Check for improvement
            if metrics.total_cost < best_cost:
                best_cost = metrics.total_cost
                best_centers = new_centers.copy()
                best_labels = new_assignments.copy()
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            
            # Early stopping checks
            if iterations_without_improvement >= self.early_stopping_tol:
                break
            
            # Check for convergence
            if iteration > 0:
                center_shift = np.sum((new_centers - self.cluster_centers_) ** 2)
                if center_shift < self.tol:
                    break
            
            self.cluster_centers_ = new_centers
            self.labels_ = new_assignments

        # Set best found solution
        if best_centers is not None:
            self.cluster_centers_ = best_centers
            self.labels_ = best_labels

        self.n_iter_ = iteration + 1

        if self.w_cl > 0 and hasattr(self, 'potential_persistent_violations'):
            self.persistent_violations = list(self.potential_persistent_violations)
        else:
            self.persistent_violations = []

        # self.sorted_constraints.close()
        # self.constraint_graph.close()

        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Predict cluster labels for new data"""

        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before making predictions")

        distances = self._compute_distances(X)
        return np.argmin(distances, axis=0)

    def get_violation_statistics(self) -> Dict:
        """Get comprehensive violation statistics"""

        if self.w_cl <= 0:
            return None
        
        if not hasattr(self, 'violation_counts') or not self.violation_counts:
            return {"error": "Model has not been fitted yet"}

        # Calculate frequency of violations
        total_iterations = len(self.violations_per_iteration)
        violation_frequency = {constraint: count / total_iterations
                             for constraint, count in self.violation_counts.items()}

        # Find most frequently violated constraints
        most_violated = sorted(self.violation_counts.items(),
                              key=lambda x: x[1], reverse=True)[:10]

        # Analyze violation trends
        violation_trend = "decreasing" if self.violations_per_iteration[-1] < self.violations_per_iteration[0] else "increasing"
        if len(self.violations_per_iteration) > 2:
            # Check if violations are consistently decreasing
            is_decreasing = all(self.violations_per_iteration[i] >= self.violations_per_iteration[i+1]
                               for i in range(len(self.violations_per_iteration)-1))
            if is_decreasing:
                violation_trend = "consistently decreasing"

            # Check if violations are consistently increasing
            is_increasing = all(self.violations_per_iteration[i] <= self.violations_per_iteration[i+1]
                              for i in range(len(self.violations_per_iteration)-1))
            if is_increasing:
                violation_trend = "consistently increasing"

        return {
            "total_unique_violations": len(self.violation_counts),
            "violations_per_iteration": self.violations_per_iteration,
            "persistent_violations": self.persistent_violations,
            "most_violated_constraints": most_violated,
            "violation_trend": violation_trend,
            "violation_frequency": violation_frequency
        }

    def pckmeans(self,
                 X: npt.NDArray[np.float64],
                 constraint_matrix: csr_matrix,
                 config: ExperimentConfig,
                 cluster_name: str,
                 skip_init: bool = False,
                 feature_data: Optional[dict] = None,
                 feature_key=None) -> 'ConstrainedKMeans':
        """
        Main entry point. Handles initialization, fits the model,
        prints cluster quality metrics, and saves results.
        """
        self.logger = config.logger

        if skip_init:
            self.logger.info("Using scikit-learn KMeans for initialization...")
            sk_kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init='k-means++', n_init=5)
            sk_kmeans.fit(X)
            self.cluster_centers_ = sk_kmeans.cluster_centers_
        else:
            if self.initializer is None:
                self.initializer = KMeansPlusPlusInit(
                    strategy=InitializationStrategy.CONSTRAINT_AWARE,
                    w_cl=self.w_cl,
                    logger=self.logger
                )

        self.fit(X=X,
                 constraint_matrix=constraint_matrix,
                 skip_init=skip_init)

        clustering_data = {
            "number_cluster": self.n_clusters,
            "embeddings": X,
            "labels": self.labels_,
            "cluster_centers": self.cluster_centers_
        }

        # Compute and print purity results
        if feature_data is not None and feature_key is not None:
            self.logger.info("=== Purity Results ===")
            purity = Purity(feature_data, clustering_data, feature_key, logger=self.logger)
            purity.compute_purity()
            purity.print_results()

        self.logger.info("=== Cluster Quality Metrics ===")
        cluster_metrics = ClusterMetrics(clustering_data, logger=self.logger)
        cluster_metrics.print_results()
        self.logger.info("==============================")

        self.save(config, cluster_name, X)
        return self

    def save(self, config, cluster_name: str, embeddings):
        """Save clustering results to disk."""

        self.logger.info("Saving clustering results...")

        clustering_data = {
            "number_cluster": self.n_clusters,
            "embeddings": embeddings,
            "labels": self.labels_,
            "cluster_centers": self.cluster_centers_,
            "w_cl": self.w_cl,
            "centroid_percentile": self.centroid_percentile,
            "pairwise_percentile": self.pairwise_percentile,
            "violations": self.get_violation_statistics()
        }

        out_dir = f"{config.task_dir}/{config.dataset_out_name}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{cluster_name}_clusters_{self.n_clusters}_w{self.w_cl}.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(clustering_data, f, protocol=pickle.HIGHEST_PROTOCOL)