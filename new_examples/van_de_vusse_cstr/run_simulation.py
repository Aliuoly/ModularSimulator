"""Run the Van de Vusse CSTR example with the new System/ProcessModel API."""

from __future__ import annotations

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt

from modular_simulation.framework import System
from modular_simulation.plotting import plot_triplet_series
from modular_simulation.utils.wrappers import hour

from component_definition import control_elements, sensors, calculations  # pyright: ignore[reportImplicitRelativeImport]
from process_definition import VanDeVusseProcessModel  # pyright: ignore[reportImplicitRelativeImport]


def make_systems(record_history: bool = False) -> dict[str, System]:
    return {
        "normal": System(
            dt=hour(0.01),
            process_model=VanDeVusseProcessModel(),
            sensors=sensors,
            calculations=calculations,
            control_elements=control_elements,
            record_history=record_history,
            show_progress=False,
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mpl.set_loglevel("warning")

    pv_kwargs = {"linestyle": "-"}
    sp_kwargs = {"linestyle": "--", "alpha": 0.5}

    systems = make_systems()
    for label, system in systems.items():
        system.step(hour(120))

        history = system.history

        plt.figure(figsize=(12, 10))

        ax = plt.subplot(3, 2, 1)
        plot_triplet_series(
            ax,
            history["T"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plot_triplet_series(
            ax,
            history["T.sp"],
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
            history["Qk"],
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
            history["Ca"],
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
            history["Cb"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plot_triplet_series(
            ax,
            history["Cb.sp"],
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
            history["Tk"],
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
            history["Tj_in"],
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
