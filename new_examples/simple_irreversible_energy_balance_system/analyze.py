"""Profiling helper for the energy balance example."""

from __future__ import annotations

import cProfile
import io
import pstats

from tqdm import tqdm

from run_simulation import system

N_STEPS = 3000


def run_simulation(iterations: int) -> None:
    for _ in tqdm(range(iterations)):
        system.step()


if __name__ == "__main__":
    print("--- Starting Profiling Session ---")
    print(f"Running {N_STEPS} steps for the energy balance system.")

    # Warm up potential JIT compilation.
    run_simulation(1)

    profiler = cProfile.Profile()
    profiler.enable()
    run_simulation(N_STEPS)
    profiler.disable()

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats("modular_simulation|energy_balance|scipy")
    print(buffer.getvalue())
