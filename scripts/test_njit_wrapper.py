import numba
from timeit import timeit

a = 1.0
b = 2.0
def py_func():
    temp = a + b
    for i in range(1000):
        temp *= i/100
    return temp

def wrapper():
    compiled = numba.njit()(py_func)
    return compiled

@numba.njit
def compiled_func():
    temp = a + b
    for i in range(1000):
        temp *= i/100
    return temp

class Dummy:
    def __init__(self, fast = True):
        if fast:
            self.py_func = numba.njit()(self.py_func)

    @staticmethod
    def py_func():
        temp = a + b
        for i in range(1000):
            temp *= i/100
        return temp


compiled = wrapper()

print(f"njit wrapped function ran for : {timeit(compiled, setup=compiled, number = 10_000)}")
print(f"python function ran for : {timeit(py_func, setup = py_func, number = 10_000)}")
print(f"straight compiled function ran for :{timeit(compiled_func, setup = compiled_func, number = 10_000)}")


dummy1 = Dummy(fast = True)
func1 = dummy1.py_func
print(f"njit wrapper static method ran for :{timeit(func1, setup=func1, number = 10_000)}")
dummy2 = Dummy(fast = False)
func2 = dummy2.py_func
print(f"python static method ran for :{timeit(func2, setup=func2, number = 10_000)}")

func3 = numba.njit()(func2)
print(f"python compiled non class method ran for {timeit(func3, setup=func3, number = 10_000)}")
