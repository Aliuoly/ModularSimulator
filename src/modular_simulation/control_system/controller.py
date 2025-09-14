from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING
from numpy.typing import NDArray
import numpy as np
if TYPE_CHECKING:
    from modular_simulation.quantities import UsableResults
    from modular_simulation.control_system import Trajectory


class Controller(ABC):

    pv_tag: str
    sp_trajectory: "Trajectory"
    

    @abstractmethod
    def update(
            self, 
            usable_results: "UsableResults",
            t: float,
            ) -> Union[float, NDArray[np.float64]]:
        pass