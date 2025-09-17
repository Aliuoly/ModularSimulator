import cProfile
import pstats
import io

from run_simulation import readable_system, fast_system
import numpy as np
import matplotlib.pyplot as plt

N_STEPS = 30000
DT = 60
sp_segments = 10
np.random.seed(0)
sp_trajs = []
for i in range(sp_segments):
    val = np.random.rand()
    sp_trajs += [val] * int(N_STEPS / sp_segments)
if len(sp_trajs) < N_STEPS:
    sp_trajs += [sp_trajs[-1]] * (N_STEPS - len(sp_trajs))
else:
    sp_trajs = sp_trajs[:N_STEPS]


def run_simulation(system, iterations: int, dt: float):
    pid_controller = system.controllable_quantities.control_definitions["F_in"]
    for i in range(iterations):
        system.step(dt)
        pid_controller.sp_trajectory.change(sp_trajs[i])


def plot(systems):
    plt.figure(figsize=(14, 10))
    linestyles = ["-", "--"]
    for j, (name, system) in enumerate(systems.items()):
        history = system.measured_history
        t = history["time"]

        plt.subplot(3, 2, 1)
        plt.step(t, history["B"], linestyle=linestyles[j])
        plt.title("Concentration of B")
        plt.xlabel("Time Step")
        plt.ylabel("[B] (mol/L)")
        plt.grid(True)

        plt.subplot(3, 2, 2)
        plt.step(t, history["F_in"], linestyle=linestyles[j])
        plt.title("Inlet Flow Rate (F_in)")
        plt.xlabel("Time Step")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        plt.subplot(3, 2, 3)
        plt.step(t, history["V"], linestyle=linestyles[j])
        plt.title("Reactor Volume (V)")
        plt.xlabel("Time Step")
        plt.ylabel("Volume (L)")
        plt.grid(True)

        plt.subplot(3, 2, 4)
        plt.step(t, history["F_out"], linestyle=linestyles[j])
        plt.title("Outlet Flow Rate (F_out)")
        plt.xlabel("Time Step")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        plt.subplot(3, 2, 5)
        plt.step(t, history["T"], linestyle=linestyles[j])
        plt.title("Reactor Temperature (T)")
        plt.xlabel("Time Step")
        plt.ylabel("Temperature (K)")
        plt.grid(True)

        plt.subplot(3, 2, 6)
        plt.step(t, history["T_J"], linestyle=linestyles[j])
        plt.title("Jacket Temperature (T_J)")
        plt.xlabel("Time Step")
        plt.ylabel("Temperature (K)")
        plt.grid(True)

    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.legend()
    plt.tight_layout()
    plt.show()


systems = {"readable": readable_system, "fast": fast_system}
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

# plot(systems)
