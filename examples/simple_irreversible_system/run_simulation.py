"""Entry point for the simple irreversible reaction example."""
from __future__ import annotations

import logging
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.units import Quantity, Unit

from modular_simulation.core import create_system
from modular_simulation.interfaces import PIDController, Trajectory
from modular_simulation.plotting import plot_triplet_series

from .sensor_definitions import create_sensors
from .system_definitions import IrreversibleModel


def create_controllers() -> list[PIDController]:
    """Build the PID hierarchy used in the example."""

    return [
        PIDController(
            cv_tag="B",
            mv_tag="F_in",
            sp_trajectory=Trajectory(0.5, Unit("mol/L")),
            Kp=1.0e-1,
            Ti=100.0,
            mv_range=(0 * Unit("L/s"), 100 * Unit("L/s")),
        )
    ]


def make_systems(**overrides: Any):
    """Instantiate the simulation systems for the example."""

    model = IrreversibleModel(
        V=0.0,
        A=0.0,
        B=0.0,
        F_in=0.0,
        F_out=0.0,
        k=1.0e-3,
        Cv=1.0e-1,
        CA_in=1.0,
    )

    dt = 30.0 * Unit("s")
    system = create_system(
        dt=dt,
        dynamic_model=model,
        sensors=create_sensors(),
        controllers=create_controllers(),
        use_numba=overrides.get("use_numba", False),
    )

    numba_system = create_system(
        dt=dt,
        dynamic_model=model,
        sensors=create_sensors(),
        controllers=create_controllers(),
        use_numba=True,
    )

    return {"normal": system, "fast": numba_system}


def plot_results(name: str, system) -> None:
    """Plot the time-series history for a completed simulation."""

    history = system.measured_history
    sensor_hist = history["sensors"]
    sp_hist = system.setpoint_history

    pv_kwargs = {"linestyle": "-"}
    sp_kwargs = {"linestyle": "--", "alpha": 0.5}

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(2, 2, 1)
    plot_triplet_series(ax, sensor_hist["B"], style="step", line_kwargs=pv_kwargs, label=name)
    plot_triplet_series(ax, sp_hist["B"], style="step", line_kwargs=sp_kwargs, label=f"{name} sp")
    plt.title("Concentration of B")
    plt.xlabel("Time")
    plt.ylabel("[B] (mol/L)")
    plt.grid(True)

    ax = plt.subplot(2, 2, 2)
    plot_triplet_series(ax, sensor_hist["F_in"], style="step", line_kwargs=pv_kwargs, label=name)
    plt.title("Inlet Flow Rate (F_in)")
    plt.xlabel("Time")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    ax = plt.subplot(2, 2, 3)
    plot_triplet_series(ax, sensor_hist["V"], style="step", line_kwargs=pv_kwargs, label=name)
    plt.title("Reactor Volume (V)")
    plt.xlabel("Time")
    plt.ylabel("Volume (L)")
    plt.grid(True)

    ax = plt.subplot(2, 2, 4)
    plot_triplet_series(ax, sensor_hist["F_out"], style="step", line_kwargs=pv_kwargs, label=name)
    plt.title("Outlet Flow Rate (F_out)")
    plt.xlabel("Time")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    plt.tight_layout()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mpl.set_loglevel("warning")

    systems = make_systems()
    for label, system in systems.items():
        system.step(duration=Quantity(5.0, Unit("day")))
        system.extend_controller_trajectory(cv_tag="B", value=0.2)
        system.step(duration=Quantity(5.0, Unit("day")))
        plot_results(label, system)

    for idx in range(4):
        plt.subplot(2, 2, idx + 1)
        plt.legend(systems.keys())

    plt.show()
