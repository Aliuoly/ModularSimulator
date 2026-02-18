"""Deterministic Split Reversal Demo

This module exposes a minimal, deterministic demo for a hypothetical
split-reversal operation in the connection layer. It is designed to be
importable by smoke tests and to provide a stable API surface for
benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class SplitCase:
    name: str
    input_signal: float
    expected_output: float


def _compute_output(input_signal: float) -> float:
    # Deterministic, simple transformation for demonstration purposes
    return input_signal * 0.8 + 1.0


def run_demo() -> List[Tuple[str, float]]:
    """Run a deterministic set of split-reversal scenarios.

    Returns a list of tuples: (case_name, computed_output).
    """
    cases = [
        SplitCase("case_a", input_signal=1.0, expected_output=_compute_output(1.0)),
        SplitCase("case_b", input_signal=2.0, expected_output=_compute_output(2.0)),
        SplitCase("case_c", input_signal=3.0, expected_output=_compute_output(3.0)),
    ]

    results: List[Tuple[str, float]] = []
    for c in cases:
        # Use the deterministic function to compute output
        out = _compute_output(c.input_signal)
        results.append((c.name, out))

    return results


__all__ = ["SplitCase", "run_demo"]
