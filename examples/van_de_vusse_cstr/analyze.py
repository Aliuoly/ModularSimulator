"""Simple profiler for the Van de Vusse CSTR example."""
from __future__ import annotations

import cProfile
import io
import pstats

from tqdm import tqdm

from run_simulation import make_systems

N_STEPS = 3000


def run_simulation(system, iterations: int) -> None:
    for _ in tqdm(range(iterations)):
        system.step()


if __name__ == "__main__":
    systems = make_systems()

    print("--- Starting Profiling Session ---")
    print(f"Running {N_STEPS} steps per system.")

    for name, system in systems.items():
        run_simulation(system, 5)

        print(f"\nProfiling the system '{name}'...")
        profiler = cProfile.Profile()
        profiler.enable()
        run_simulation(system, N_STEPS)
        profiler.disable()

        buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=buffer)
        stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats("modular_simulation|van_de_vusse|scipy")
        print(buffer.getvalue())

        print(f"System '{name}' profiling complete.")

