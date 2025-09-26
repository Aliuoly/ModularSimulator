from modular_simulation.measurables.base_classes import BaseIndexedModel
from timeit import timeit
import numpy as np

class Dummy(BaseIndexedModel):
    A: float = 1.0
    B: float = 2.0
    C: float = 3.0

dummy = Dummy()
num = 100_000
out = np.empty(3)
print(timeit(dummy.to_array, number = num))
print(timeit(dummy.to_array_ver2, number = num))
print(timeit(dummy.to_array_ver3, globals = {'out': out}, number = num))

print(timeit(dummy.to_array_ver4, globals = {'out': out}, number = num))