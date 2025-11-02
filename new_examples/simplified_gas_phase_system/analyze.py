import cProfile
import pstats
import io

# Import your existing simulation setups
from run_simulation import system
from tqdm import tqdm

# --- Simulation Parameters ---
# Use a high number of steps to get meaningful profile data
N_STEPS = 3000  # A good number for profiling

def run_simulation(system, iterations: int):
    for i in tqdm(range(iterations)):
        system.step()

profiler = cProfile.Profile()
profiler.enable()
run_simulation(system, N_STEPS)
profiler.disable()
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s)
ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats("modular_simulation|simple_Irreversible_system|astropy|scipy")
print(s.getvalue())
