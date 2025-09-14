import cProfile
import pstats
import io

# Import your existing simulation setups
from run_simulation import readable_system, fast_system
import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
# Use a high number of steps to get meaningful profile data
N_STEPS = 10000  # A good number for profiling
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
    pid_controller = system.controllable_quantities.control_definitions["F_in"]
    for i in range(iterations):
        system.step(dt)
        pid_controller.sp_trajectory.change(sp_trajs[i])

def plot(systems):
    plt.figure(figsize=(12, 8))
    linestyles = ['-','--']
    j = 0
    for name, system in systems.items():
        history = system._history

        # Plot Concentration of B
        plt.subplot(2, 2, 1)
        plt.plot([s["B"] for s in history], label = name, linestyle = linestyles[j])
        plt.title("Concentration of B")
        plt.xlabel("Time Step")
        plt.ylabel("[B] (mol/L)")
        plt.grid(True)

        # Plot Inlet Flow Rate (F_in)
        plt.subplot(2, 2, 2)
        plt.plot([s['F_in'] for s in history], label = name, linestyle = linestyles[j])
        plt.title("Inlet Flow Rate (F_in)")
        plt.xlabel("Time Step")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        # Plot Reactor Volume (V)
        plt.subplot(2, 2, 3)
        plt.plot([s['V'] for s in history], label = name, linestyle = linestyles[j])
        plt.title("Reactor Volume (V)")
        plt.xlabel("Time Step")
        plt.ylabel("Volume (L)")
        plt.grid(True)

        # Plot Outlet Flow Rate (F_out)
        plt.subplot(2, 2, 4)
        plt.plot([s['F_out'] for s in history], label = name, linestyle = linestyles[j])
        plt.title("Outlet Flow Rate (F_out)")
        plt.xlabel("Time Step")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        j += 1
        
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.legend()
    plt.tight_layout()
    plt.show()

systems = {"fast": fast_system, "readable": readable_system}
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
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats("modular_simulation|simple_Irreversible_system")
    print(s.getvalue())

    print(f"System '{name}' profiling complete. ")

plot(systems)