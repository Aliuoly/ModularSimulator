import cProfile
import io
import pstats

import pytest

pytest.importorskip("matplotlib")

from .run_simulation import fast_system

sensors = fast_system.usable_quantities.sensors
sensor = next(s for s in sensors if s.measurement_tag == "F_in")

profiler = cProfile.Profile()
try:
    profiler.enable()
    for _ in range(10000):
        sensor.measure(0.0)
finally:
    profiler.disable()

s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
ps.print_stats(30)
print(s.getvalue())
