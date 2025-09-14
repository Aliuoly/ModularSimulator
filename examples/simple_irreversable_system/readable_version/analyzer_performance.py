import cProfile
import pstats

# Import your existing simulation setups
from system_simulation import irreversable_system
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

def plot(system):
    history = system._history
    plt.figure(figsize=(12, 8))

    # Plot Concentration of B
    plt.subplot(2, 2, 1)
    plt.plot([s["B"] for s in history])
    plt.title("Concentration of B")
    plt.xlabel("Time Step")
    plt.ylabel("[B] (mol/L)")
    plt.grid(True)

    # Plot Inlet Flow Rate (F_in)
    plt.subplot(2, 2, 2)
    plt.plot([s['F_in'] for s in history])
    plt.title("Inlet Flow Rate (F_in)")
    plt.xlabel("Time Step")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    # Plot Reactor Volume (V)
    plt.subplot(2, 2, 3)
    plt.plot([s['V'] for s in history])
    plt.title("Reactor Volume (V)")
    plt.xlabel("Time Step")
    plt.ylabel("Volume (L)")
    plt.grid(True)

    # Plot Outlet Flow Rate (F_out)
    plt.subplot(2, 2, 4)
    plt.plot([s['F_out'] for s in history])
    plt.title("Outlet Flow Rate (F_out)")
    plt.xlabel("Time Step")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# PRIME IT
run_simulation(irreversable_system, N_STEPS, DT)

print("--- Starting Profiling Session ---")
print(f"Running {N_STEPS} steps with dt={DT} for each system.")

# --- Profile the Readable Version ---
print("\nProfiling the System...")
profiler_readable = cProfile.Profile()
profiler_readable.enable()
run_simulation(irreversable_system, N_STEPS, DT)
profiler_readable.disable()
plot(irreversable_system)
profiler_readable.dump_stats("./readable_stats.prof")
stats = pstats.Stats("./readable_stats.prof")
stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(30)

print("System profiling complete. Stats saved to './readable_stats.prof'")