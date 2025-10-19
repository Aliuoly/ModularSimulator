import cProfile
import pstats
import io

# Import your existing simulation setups
from run_simulation import make_systems
import numpy as np
from tqdm import tqdm

# --- Simulation Parameters ---
# Use a high number of steps to get meaningful profile data
N_STEPS = 3000  # A good number for profiling
sp_segments = 10
np.random.seed(0)
sp_trajs = []
for i in range(sp_segments):
    val = np.random.rand()
    sp_trajs += [val] * int(N_STEPS/sp_segments)
if len(sp_trajs) < N_STEPS:
    sp_trajs.append([sp_trajs[-1]] * (N_STEPS - len(sp_trajs)))
else:
    sp_trajs = sp_trajs[:N_STEPS]

def run_simulation(system, iterations: int):
    for i in tqdm(range(iterations)):
        system.step()
        system.extend_controller_trajectory(cv_tag = 'B', value = sp_trajs[i])

systems = make_systems()
print("--- Starting Profiling Session ---")
print(f"Running {N_STEPS} steps for system .")

for name, system in systems.items():
    # prime the systems for JIT compilation
    run_simulation(system, 1)
    # --- Profile each version ---
    print(f"\nProfiling the System '{name}'...")
    profiler = cProfile.Profile()
    profiler.enable()
    run_simulation(system, N_STEPS)
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats("modular_simulation|examples|scipy")
    print(s.getvalue())

    print(f"System '{name}' profiling complete. ")

#plot(systems)