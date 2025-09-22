from modular_simulation.measurables.base_indexed_model import BaseIndexedModel
from numpy.typing import NDArray
import numpy as np


class DummyModel(BaseIndexedModel):
    A: NDArray[np.float64]
    B: NDArray[np.float64]
    C: float

dummy_model = DummyModel(
    A = np.array([0.0, 1.0]),
    B = np.array([2.0, 3, 4.0, 5]),
    C = 9.0
)

print(dummy_model._index_map.__members__)

