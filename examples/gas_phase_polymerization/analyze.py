"""Profiling utility for the gas-phase polymerization example."""

from __future__ import annotations

import cProfile
import io
import pstats

from tqdm import tqdm

from run_simulation import system

N_STEPS = 5000


def run(iterations: int) -> None:
    for _ in tqdm(range(iterations)):
        system.step()


if __name__ == "__main__":
    print("--- Starting Profiling Session ---")
    print(f"Running {N_STEPS} steps for the gas-phase reactor.")

    run(1)
    profiler = cProfile.Profile()
    profiler.enable()
    run(N_STEPS)
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(
        "modular_simulation|gas_phase_polymerization"
    )
    print(buf.getvalue())
