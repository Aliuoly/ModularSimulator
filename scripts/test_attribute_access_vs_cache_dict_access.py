from timeit import timeit

class CallableDummy:
    def __init__(self, tag, a, b):
        self.tag = tag
        self.a = a
        self.b = b
    
    def update(self):
        return self.a ** abs(self.b)

class Dummy:
    def __init__(self, callables_list):
        self.callables_list = callables_list
        self._results = {}
    
    def update(self):
        for callable in self.callables_list:
            self._results[callable.tag] = callable.update()

class DummyVer2:
    def __init__(self, callables_list):
        self.callables_list = callables_list
        self._results = {}
        self._cached_callables = {callable.tag: callable.update for callable in callables_list}
    
    def update(self):
        for tag, method in self._cached_callables.items():
            self._results[tag] = method()

class DummyVer3:
    def __init__(self, callables_list):
        self.callables_list = callables_list
        self._results = {}
    
    def update(self):
        self._results.update(
            {callable.tag: callable.update() for callable in self.callables_list}
        )


callables = [CallableDummy('A', 1, 1), CallableDummy('B', 2, 2), CallableDummy('C',3,3),CallableDummy('D',4,4)]
dummy1 = Dummy(callables)
dummy2 = DummyVer2(callables)
dummy3 = DummyVer3(callables)

print(timeit(dummy1.update, number = 100_0000))
print(timeit(dummy2.update, number = 100_0000))
print(timeit(dummy3.update, number = 100_0000))