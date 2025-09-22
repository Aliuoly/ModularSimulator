import cProfile
import pstats
import io

# Import your existing simulation setups
from run_simulation import readable_system
import numpy as np
from tqdm import tqdm

# --- Simulation Parameters ---
# Use a high number of steps to get meaningful profile data
N_STEPS = 30000  # A good number for profiling
DT = 60
sp_segments = 10
np.random.seed(0)
sp_trajs = []
for i in range(sp_segments):
    val = np.random.rand()
    sp_trajs += [val] * int(N_STEPS/sp_segments)
if len(sp_trajs) < N_STEPS:
    sp_trajs += [sp_trajs[-1]] * N_STEPS - len(sp_trajs)
else:
    sp_trajs = sp_trajs[:N_STEPS]

def run_simulation(system, iterations: int, dt: float):
    """A simple function to run the simulation steps."""
    for i in tqdm(range(iterations)):
        system.step()
        system.extend_controller_trajectory(cv_tag = 'B', value = sp_trajs[i])

systems = { "readable": readable_system,}
print("--- Starting Profiling Session ---")
print(f"Running {N_STEPS} steps with dt={DT} for system .")

for name, system in systems.items():
    # prime the systems for JIT compilation
    run_simulation(system, 1, DT)
    # --- Profile each version ---
    print(f"\nProfiling the System '{name}'...")
    profiler = cProfile.Profile()
    profiler.enable()
    run_simulation(system, N_STEPS, DT)
    profiler.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.FILENAME
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.sort_stats(pstats.SortKey.FILENAME).print_stats("modular_simulation|simple_Irreversible_system")
    print(s.getvalue())

    print(f"System '{name}' profiling complete. ")

#plot(systems)