import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import Optional

from scipy.sparse import csr_matrix


class BaseInitializer(ABC):
    """Abstract base class for cluster initializers"""

    @abstractmethod
    def initialize(self,
                  X: npt.NDArray[np.float64],
                  n_clusters: int,
                  constraint_matrix: Optional[csr_matrix] = None,
                  random_state: Optional[int] = None) -> npt.NDArray[np.float64]:
        pass