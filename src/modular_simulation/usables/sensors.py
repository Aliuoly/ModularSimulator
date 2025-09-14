from numpy.typing import NDArray
import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modular_simulation.quantities import MeasurableQuantities

class MeasurementFunction(ABC):
    @abstractmethod
    def __call__(self, measurable_quantities: "MeasurableQuantities") -> float | NDArray[np.float64]:
        pass

class Sensor(ABC):

    measurement_function: MeasurementFunction

    @abstractmethod
    def measure(self, measurable_quantities: "MeasurableQuantities") -> float | NDArray[np.float64]:
        pass


