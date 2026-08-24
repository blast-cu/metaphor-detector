from enum import Enum
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src.clustering.initializer.base_initializer import BaseInitializer


class InitializationStrategy(Enum):
    """Enumeration of available initialization strategies"""
    
    STANDARD = "standard"  # Standard K-means++ without constraint awareness
    CONSTRAINT_AWARE = "constraint_aware"  # K-means++ with constraint awareness

class KMeansPlusPlusInit(BaseInitializer):
    """Configurable KMeans++ initialization with optional constraint awareness"""

    def __init__(self,
                 strategy: InitializationStrategy = InitializationStrategy.STANDARD,
                 w_cl: float = 1.0,
                 logger=None):
        """
        Initialize the optimized KMeans++ initializer

        Args:
            strategy: Initialization strategy to use
            w_cl: Weight for constraint satisfaction (only used if strategy is CONSTRAINT_AWARE)
            logger: Logger instance for progress messages
        """

        self.strategy = strategy
        self.w_cl = w_cl
        self.logger = logger
        self._distance_cache = {}

    @staticmethod
    def _build_constraint_matrix(n_samples: int,
                               constraints: List[Tuple[int, int]]) -> csr_matrix:
        """
        Build sparse constraint matrix for efficient lookup and vectorized operations
        
        Returns a sparse matrix where M[i,j] = 1 if points i and j have a constraint
        """

        # Initialize rows, cols and data for sparse matrix construction
        rows = []
        cols = []
        
        # Add constraints to the sparse matrix (symmetric)
        for i, j in constraints:
            rows.extend([i, j])
            cols.extend([j, i])

            
        # Create sparse matrix with binary values
        data = np.ones(len(rows), dtype=np.bool_)
        constraint_matrix = csr_matrix((data, (rows, cols)), 
                                     shape=(n_samples, n_samples), 
                                     dtype=np.bool_)
        
        return constraint_matrix

    @staticmethod
    def _compute_point_norms(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Precompute squared L2 norms of all points"""

        return np.sum(X * X, axis=1)
    
    def _compute_distances_to_centers(self,
                                    X: npt.NDArray[np.float64],
                                    centers: npt.NDArray[np.float64],
                                    X_norm: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Efficiently compute distances from all points to cluster centers
        
        Args:
            X: Input data of shape (n_samples, n_features)
            centers: Current centers of shape (n_centers, n_features)
            X_norm: Precomputed squared L2 norms of input points
            
        Returns:
            Distances matrix of shape (n_centers, n_samples)
        """

        n_centers = centers.shape[0]
        n_samples = X.shape[0]
        
        # Check cache for this center configuration
        cache_key = hash(centers.tobytes())
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        if n_centers == 0:
            return np.ones((1, n_samples))
            
        # Compute center norms
        center_norms = np.sum(centers * centers, axis=1)
        
        # Compute dot products
        # (x-y)² = x² + y² - 2xy
        dots = X @ centers.T  # shape: (n_samples, n_centers)
        
        # Compute distances efficiently using broadcasting
        # Equivalent to: 
        # distances[i,j] = ||X[j]||² + ||centers[i]||² - 2⟨X[j], centers[i]⟩
        distances = -2 * dots
        distances = distances.T  # shape: (n_centers, n_samples)
        distances += center_norms[:, np.newaxis]
        distances += X_norm[np.newaxis, :]

        # Ensure no negative distances due to numerical precision
        distances = np.maximum(distances, 0)
        
        # Cache result
        self._distance_cache[cache_key] = distances
        return distances

    @staticmethod
    def _compute_min_distances(all_distances: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute minimum distance from each point to any center"""

        return np.min(all_distances, axis=0)
        
    def _compute_constraint_bonus(self,
                                center_indices: List[int],
                                constraint_matrix: csr_matrix,
                                n_samples: int) -> npt.NDArray[np.float64]:
        """
        Compute constraint satisfaction bonus for all points
        
        Args:
            center_indices: Indices of selected centers
            constraint_matrix: Sparse matrix of constraints
            n_samples: Total number of data points
            
        Returns:
            Array of constraint bonuses for each point
        """

        if not center_indices or self.strategy != InitializationStrategy.CONSTRAINT_AWARE:
            return np.zeros(n_samples)
            
        # Create mask for already selected centers
        selected_mask = np.zeros(n_samples, dtype=bool)
        selected_mask[center_indices] = True
        
        # Get constraints with selected centers
        if len(center_indices) > 0:
            # Extract submatrix for center indices (all constraints with centers)
            center_constraints = constraint_matrix[:, center_indices]
            
            # Sum across rows to get number of constraints each point has with centers
            satisfied_constraints = center_constraints.sum(axis=1).A1  # A1 converts to 1D array
            
            # Apply weight
            constraint_bonus = self.w_cl * satisfied_constraints
            
            # Zero out already selected centers
            constraint_bonus[selected_mask] = 0
            
            return constraint_bonus
        return np.zeros(n_samples)
        
    def initialize(self,
                  X: npt.NDArray[np.float64],
                  n_clusters: int,
                  constraint_matrix: csr_matrix = None,
                  random_state: Optional[int] = None) -> npt.NDArray[np.float64]:
        """
        Initialize cluster centers using optimized KMeans++
        
        Args:
            X: Input data of shape (n_samples, n_features)
            n_clusters: Number of clusters to initialize
            cl_constraints: List of cannot-link constraints
            random_state: Random seed for reproducibility
            
        Returns:
            Initial cluster centers of shape (n_clusters, n_features)
        """

        if random_state is not None:
            np.random.seed(random_state)
            
        n_samples, n_features = X.shape
        centers = np.zeros((n_clusters, n_features))
        center_indices = []
        
        # Clear distance cache
        self._distance_cache = {}

        log = self.logger.info if self.logger else print

        # Precompute point norms (constant throughout initialization)
        log("Precomputing point norms...")
        X_norm = self._compute_point_norms(X)

        # constraint_matrix is pre-built by the caller; no need to rebuild it here

        # Select first center randomly
        first_idx = np.random.randint(n_samples)
        centers[0] = X[first_idx]
        center_indices.append(first_idx)

        # Compute initial distances to first center
        all_distances = self._compute_distances_to_centers(X, centers[:1], X_norm)

        # Select remaining centers
        log("Selecting remaining cluster centers for initialization...")
        for k in tqdm(range(1, n_clusters)):
            # Get minimum distances to existing centers
            min_distances = self._compute_min_distances(all_distances)
            
            # Add constraint bonus only if using constraint-aware strategy
            if self.strategy == InitializationStrategy.CONSTRAINT_AWARE and self.w_cl > 0:
                constraint_bonus = self._compute_constraint_bonus(
                    center_indices, constraint_matrix, n_samples
                )
                scores = min_distances + constraint_bonus
            else:
                scores = min_distances
            
            # Zero out already selected points
            mask = np.ones(n_samples, dtype=bool)
            mask[center_indices] = False
            masked_scores = scores.copy()
            masked_scores[~mask] = 0
            
            # Select next center
            if np.sum(masked_scores) == 0:
                # If all scores are 0, choose randomly from remaining points
                remaining = list(set(range(n_samples)) - set(center_indices))
                next_idx = np.random.choice(remaining)
            else:
                # Choose proportional to scores
                probs = masked_scores / np.sum(masked_scores)
                next_idx = np.random.choice(n_samples, p=probs)
            
            # Update centers and indices
            centers[k] = X[next_idx]
            center_indices.append(next_idx)
            
            # Efficiently update distances by computing only for the new center
            new_center_distances = self._compute_distances_to_centers(
                X, centers[k:k+1], X_norm
            )
            
            # Append to existing distances matrix
            all_distances = np.vstack([all_distances, new_center_distances])
                
        return centers