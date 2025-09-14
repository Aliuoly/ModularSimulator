
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

class Trajectory(ABC):
    
    @abstractmethod
    def __call__(self, t) -> float | NDArray[np.float64]:
        pass
