from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING
from numpy.typing import NDArray
import numpy as np
if TYPE_CHECKING:
    from modular_simulation.quantities import UsableResults

class Calculation(ABC):
    @abstractmethod
    def calculate(self, 
                  usable_results: "UsableResults"
                  ) -> Union[float, NDArray[np.float64]]:
        pass