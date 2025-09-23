import cProfile
import pstats
import io

from run_simulation import readable_system, fast_system, plot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N_STEPS = 3000
DT = 60
sp_segments = 10
np.random.seed(0)
sp_trajs = []
for i in range(sp_segments):
    val = np.random.rand() * 0.02
    sp_trajs += [val] * int(N_STEPS / sp_segments)
if len(sp_trajs) < N_STEPS:
    sp_trajs += [sp_trajs[-1]] * (N_STEPS - len(sp_trajs))
else:
    sp_trajs = sp_trajs[:N_STEPS]



def run_simulation(system, iterations: int, dt: float):
    for i in tqdm(range(iterations)):
        system.step()
        system.extend_controller_trajectory(cv_tag = 'B', value = sp_trajs[i])


systems = {"readable": readable_system}
print("--- Starting Profiling Session ---")
print(f"Running {N_STEPS} steps with dt={DT} for the energy balance system.")

for name, system in systems.items():
    run_simulation(system, 1, DT)
    print(f"\nProfiling the System '{name}'...")
    profiler = cProfile.Profile()
    profiler.enable()
    run_simulation(system, N_STEPS, DT)
    profiler.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(
        "modular_simulation|simple_irreversible_energy_balance_system"
    )
    print(s.getvalue())
    print(f"System '{name}' profiling complete.")

plot(systems)
