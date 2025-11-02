"""Run the Van de Vusse CSTR example with the new System/ProcessModel API."""
from __future__ import annotations

import logging
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt

from modular_simulation.framework import System
from modular_simulation.plotting import plot_triplet_series
from modular_simulation.utils.wrappers import hour

from component_definition import controllers, sensors, calculations
from process_definition import VanDeVusseProcessModel


def make_systems(record_history: bool = False) -> Dict[str, System]:
    """Build normal and numba-accelerated Van de Vusse systems."""
    base_kwargs = dict(
        dt=hour(0.01),
        sensors=sensors,
        calculations=calculations,
        controllers=controllers,
        record_history=record_history,
        show_progress=False,
    )
    return {
        "normal": System(process_model=VanDeVusseProcessModel(), use_numba=False, **base_kwargs),
        "fast": System(process_model=VanDeVusseProcessModel(), use_numba=True, **base_kwargs),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mpl.set_loglevel("warning")

    pv_kwargs = {"linestyle": "-"}
    sp_kwargs = {"linestyle": "--", "alpha": 0.5}

    systems = make_systems()
    for label, system in systems.items():
        system.step(hour(120))

        history = system.measured_history
        sensor_hist = history["sensors"]
        calc_hist = history["calculations"]
        sp_hist = system.setpoint_history

        plt.figure(figsize=(12, 10))

        ax = plt.subplot(3, 2, 1)
        plot_triplet_series(
            ax,
            sensor_hist["T"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plot_triplet_series(
            ax,
            sp_hist["T.sp"],
            style="step",
            line_kwargs=sp_kwargs,
            label=f"SP {label}",
        )
        plt.title("Reactor Temperature")
        plt.xlabel("Time [h]")
        plt.ylabel("Temperature [°C]")
        plt.grid(True)

        ax = plt.subplot(3, 2, 2)
        plot_triplet_series(
            ax,
            calc_hist["Qk"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plt.title("Jacket Heat Duty")
        plt.xlabel("Time [h]")
        plt.ylabel("Qk [kJ/h]")
        plt.grid(True)

        ax = plt.subplot(3, 2, 3)
        plot_triplet_series(
            ax,
            sensor_hist["Ca"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plt.title("Concentration of A")
        plt.xlabel("Time [h]")
        plt.ylabel("[A] [mol/L]")
        plt.grid(True)

        ax = plt.subplot(3, 2, 4)
        plot_triplet_series(
            ax,
            sensor_hist["Cb"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plot_triplet_series(
            ax,
            sp_hist["Cb.sp"],
            style="step",
            line_kwargs=sp_kwargs,
            label=f"SP {label}",
        )
        plt.title("Concentration of B")
        plt.xlabel("Time [h]")
        plt.ylabel("[B] [mol/L]")
        plt.grid(True)

        ax = plt.subplot(3, 2, 5)
        plot_triplet_series(
            ax,
            sensor_hist["Tk"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plt.title("Jacket Temperature")
        plt.xlabel("Time [h]")
        plt.ylabel("Temperature [°C]")
        plt.grid(True)

        ax = plt.subplot(3, 2, 6)
        plot_triplet_series(
            ax,
            sensor_hist["Tj_in"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plt.title("Jacket Inlet Temperature")
        plt.xlabel("Time [h]")
        plt.ylabel("Temperature [°C]")
        plt.grid(True)

        plt.tight_layout()
    plt.show()

