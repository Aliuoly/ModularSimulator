import cProfile
import pstats
import io

# Import your existing simulation setups
from run_simulation import normal_system
from astropy.units import Unit


print("--- Starting Profiling Session ---")
profiler = cProfile.Profile()
profiler.enable()
normal_system.step(24 * Unit("hour"))
profiler.disable()
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s)
ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats("modular_simulation|gas_phase_polymerization|scipy")
print(s.getvalue())
