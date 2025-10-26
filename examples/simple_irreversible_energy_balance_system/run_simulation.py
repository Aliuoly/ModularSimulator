"""Simulation harness for the irreversible reaction with energy balance example."""
from __future__ import annotations

import logging
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.units import Quantity, Unit

from modular_simulation.core import create_system
from modular_simulation.interfaces import PIDController, SampledDelayedSensor, Trajectory
from modular_simulation.plotting import plot_triplet_series

from .system_definitions import EnergyBalanceModel


def create_sensors() -> list[SampledDelayedSensor]:
    return [
        SampledDelayedSensor(measurement_tag="F_out", unit=Unit("L/s")),
        SampledDelayedSensor(
            measurement_tag="F_in",
            unit=Unit("L/s"),
            coefficient_of_variance=0.05,
        ),
        SampledDelayedSensor(
            measurement_tag="B",
            unit=Unit("mol/L"),
            coefficient_of_variance=0.05,
            sampling_period=900.0,
            deadtime=900.0,
        ),
        SampledDelayedSensor(measurement_tag="V", unit=Unit("L")),
        SampledDelayedSensor(measurement_tag="T", unit=Unit("K")),
        SampledDelayedSensor(measurement_tag="T_J", unit=Unit("K")),
        SampledDelayedSensor(measurement_tag="T_J_in", unit=Unit("K")),
        SampledDelayedSensor(measurement_tag="jacket_flow", unit=Unit("L/s")),
    ]


def create_controllers() -> list[PIDController]:
    return [
        PIDController(
            mv_tag="F_in",
            cv_tag="V",
            sp_trajectory=Trajectory(1.0e3, unit=Unit("L")),
            Kp=1.0e-2,
            Ti=100.0,
            mv_range=(0.0 * Unit("L/s"), 1.0e6 * Unit("L/s")),
        ),
        PIDController(
            mv_tag="T_J_in",
            cv_tag="T_J",
            sp_trajectory=Trajectory(300, unit=Unit("K")),
            Kp=1.0e-1,
            Ti=50.0,
            mv_range=(200 * Unit("K"), 350 * Unit("K")),
            cascade_controller=PIDController(
                mv_tag="T_J",
                cv_tag="T",
                sp_trajectory=Trajectory(300, Unit("K")),
                Kp=1.0e-1,
                Ti=100.0,
                mv_range=(200 * Unit("K"), 350 * Unit("K")),
                cascade_controller=PIDController(
                    mv_tag="T",
                    cv_tag="B",
                    sp_trajectory=Trajectory(0.02, unit=Unit("mol/L"))
                    .hold(duration=15e3)
                    .hold(15e3, 0.05)
                    .hold(15e3, 0.1)
                    .hold(15e3, 0.01),
                    Kp=2.0e-1,
                    Ti=5.0,
                    mv_range=(250.0 * Unit("K"), 350.0 * Unit("K")),
                ),
            ),
        ),
    ]


def make_systems(**overrides: Any):
    model = EnergyBalanceModel()

    dt = 30.0 * Unit("s")
    sensors = create_sensors()
    controllers = create_controllers()

    system = create_system(
        dt=dt,
        dynamic_model=model,
        sensors=sensors,
        controllers=controllers,
        use_numba=overrides.get("use_numba", False),
    )
    fast_system = create_system(
        dt=dt,
        dynamic_model=model,
        sensors=create_sensors(),
        controllers=create_controllers(),
        use_numba=True,
    )
    return {"normal": system, "fast": fast_system}


def plot_results(name: str, system) -> None:
    history = system.measured_history
    sensor_hist = history["sensors"]
    sp_hist = system.setpoint_history

    pv_kwargs = {"linestyle": "-"}
    sp_kwargs = {"linestyle": "--", "alpha": 0.5}

    plt.figure(figsize=(14, 10))
    ax = plt.subplot(4, 2, 1)
    plot_triplet_series(ax, sensor_hist["B"], style="step", line_kwargs=pv_kwargs, label=name)
    plot_triplet_series(ax, sp_hist["B"], style="step", line_kwargs=sp_kwargs, label=f"{name} sp")
    plt.title("Concentration of B")
    plt.xlabel("Time")
    plt.ylabel("[B] (mol/L)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 2)
    plot_triplet_series(ax, sensor_hist["F_in"], style="step", line_kwargs=pv_kwargs, label=name)
    plt.title("Inlet Flow Rate (F_in)")
    plt.xlabel("Time")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 3)
    plot_triplet_series(ax, sensor_hist["V"], style="step", line_kwargs=pv_kwargs, label=name)
    plt.title("Reactor Volume (V)")
    plt.xlabel("Time")
    plt.ylabel("Volume (L)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 4)
    plot_triplet_series(ax, sensor_hist["F_out"], style="step", line_kwargs=pv_kwargs, label=name)
    plt.title("Outlet Flow Rate (F_out)")
    plt.xlabel("Time")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 5)
    plot_triplet_series(ax, sensor_hist["T"], style="step", line_kwargs=pv_kwargs, label=name)
    plt.title("Reactor Temperature")
    plt.xlabel("Time")
    plt.ylabel("Temperature (K)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 6)
    plot_triplet_series(ax, sensor_hist["T_J"], style="step", line_kwargs=pv_kwargs, label=name)
    plt.title("Jacket Temperature")
    plt.xlabel("Time")
    plt.ylabel("Temperature (K)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 7)
    plot_triplet_series(ax, sensor_hist["T_J_in"], style="step", line_kwargs=pv_kwargs, label=name)
    plt.title("Jacket Inlet Temperature")
    plt.xlabel("Time")
    plt.ylabel("Temperature (K)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 8)
    plot_triplet_series(ax, sensor_hist["jacket_flow"], style="step", line_kwargs=pv_kwargs, label=name)
    plt.title("Jacket Flow")
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

    for idx in range(8):
        plt.subplot(4, 2, idx + 1)
        plt.legend(systems.keys())

    plt.show()
