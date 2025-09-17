from run_simulation import fast_system
import cProfile
import pstats
import io

sensor = fast_system.usable_quantities.measurement_definitions["F_in"]
measurables = fast_system.measurable_quantities
profiler = cProfile.Profile()
profiler.enable()
for _ in range(10000):
    sensor.measure(measurables, 0.0)
profiler.disable()
s = io.StringIO()
sortby = pstats.SortKey.CUMULATIVE
ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(30)
print(s.getvalue())
