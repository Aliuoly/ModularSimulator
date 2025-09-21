import timeit
from modular_simulation.usables import TimeValueQualityTriplet
import numpy as np

SIZE = 100000

def make_triplet(t):
    return TimeValueQualityTriplet(t, t, True)

def list_append():
    list = []
    for i in range(SIZE):
        list += [make_triplet(i)]
    
def ndarray_preallocate():
    array = np.zeros(SIZE, dtype = TimeValueQualityTriplet)
    for i in range(SIZE):
        array[i] = make_triplet(i)

print(timeit.timeit(list_append, number = 1000))

print(timeit.timeit(ndarray_preallocate, number = 1000))
