"""Quick profiling harness for the fast irreversible system variant."""
from __future__ import annotations

import cProfile
import io
import pstats

import pytest

pytest.importorskip("matplotlib")

from .run_simulation import make_systems

systems = make_systems()
fast_system = systems["fast"]

sensor = next(s for s in fast_system.sensors if s.measurement_tag == "F_in")

profiler = cProfile.Profile()
try:
    profiler.enable()
    for _ in range(10_000):
        sensor.measure(0.0)
finally:
    profiler.disable()

s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
ps.print_stats(30)
print(s.getvalue())
