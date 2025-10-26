"""Simulation runner for the Van de Vusse CSTR example."""
from __future__ import annotations

import logging
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.units import Unit

from modular_simulation.core import create_system
from modular_simulation.interfaces import PIDController, SampledDelayedSensor, Trajectory
from modular_simulation.plotting import plot_triplet_series

from .system_definitions import HeatDutyCalculation, VanDeVusseModel


def create_sensors() -> list[SampledDelayedSensor]:
    return [
        SampledDelayedSensor(measurement_tag="Ca", sampling_period=0.1, unit=Unit("mol/L")),
        SampledDelayedSensor(measurement_tag="Cb", sampling_period=0.1, unit=Unit("mol/L")),
        SampledDelayedSensor(measurement_tag="T", sampling_period=0.1, unit=Unit("K")),
        SampledDelayedSensor(measurement_tag="Tk", sampling_period=0.1, unit=Unit("K")),
        SampledDelayedSensor(measurement_tag="Tj_in", sampling_period=0.1, unit=Unit("K")),
    ]


def create_calculations(model: VanDeVusseModel) -> list[HeatDutyCalculation]:
    return [
        HeatDutyCalculation(
            heat_duty_tag="Qk",
            Tk_tag="Tk",
            T_tag="T",
            area=model.AR,
            kw=model.kw,
        )
    ]


def create_controllers() -> list[PIDController]:
    return [
        PIDController(
            mv_tag="Tj_in",
            cv_tag="T",
            sp_trajectory=Trajectory(80.0)
            .hold(30.0)
            .step(10)
            .hold(30.0)
            .step(-5),
            mv_range=(10.0, 110.0),
            Kp=2.0,
            Ti=0.5,
            cascade_controller=PIDController(
                mv_tag="T",
                cv_tag="Cb",
                sp_trajectory=Trajectory(0.20)
                .hold(40.0)
                .step(0.10)
                .hold(40.0)
                .step(-0.2),
                mv_range=(50.0, 120.0),
                Kp=20.0,
                Ti=1.0,
            ),
        )
    ]


def make_systems(**overrides: Any):
    model = VanDeVusseModel()
    dt = 0.01  # hours (~36 seconds)

    sensors = create_sensors()
    calculations = create_calculations(model)
    controllers = create_controllers()

    system = create_system(
        dt=dt,
        dynamic_model=model,
        sensors=sensors,
        calculations=calculations,
        controllers=controllers,
        use_numba=overrides.get("use_numba", False),
    )

    fast_system = create_system(
        dt=dt,
        dynamic_model=model,
        sensors=create_sensors(),
        calculations=create_calculations(model),
        controllers=create_controllers(),
        use_numba=True,
    )
    return {"normal": system, "fast": fast_system}


def plot_results(label: str, system) -> None:
    history = system.measured_history
    sensor_hist = history["sensors"]
    calc_hist = history["calculations"]
    sp_hist = system.setpoint_history

    pv_kwargs = {"linestyle": "-"}
    sp_kwargs = {"linestyle": "--", "alpha": 0.5}

    plt.figure(figsize=(12, 10))
    ax = plt.subplot(3, 2, 1)
    plot_triplet_series(ax, sensor_hist["T"], style="step", line_kwargs=pv_kwargs, label=label)
    plot_triplet_series(ax, sp_hist["T"], style="step", line_kwargs=sp_kwargs, label=f"{label} sp")
    plt.title("Reactor Temperature")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [°C]")
    plt.grid(True)

    ax = plt.subplot(3, 2, 2)
    plot_triplet_series(ax, calc_hist["Qk"], style="step", line_kwargs=pv_kwargs, label=label)
    plt.title("Jacket Heat Duty")
    plt.xlabel("Time [h]")
    plt.ylabel("Qk")
    plt.grid(True)

    ax = plt.subplot(3, 2, 3)
    plot_triplet_series(ax, sensor_hist["Ca"], style="step", line_kwargs=pv_kwargs, label=label)
    plt.title("Concentration of A")
    plt.xlabel("Time [h]")
    plt.ylabel("[A] [mol/L]")
    plt.grid(True)

    ax = plt.subplot(3, 2, 4)
    plot_triplet_series(ax, sensor_hist["Cb"], style="step", line_kwargs=pv_kwargs, label=label)
    plot_triplet_series(ax, sp_hist["Cb"], style="step", line_kwargs=sp_kwargs, label=f"{label} sp")
    plt.title("Concentration of B")
    plt.xlabel("Time [h]")
    plt.ylabel("[B] [mol/L]")
    plt.grid(True)

    ax = plt.subplot(3, 2, 5)
    plot_triplet_series(ax, sensor_hist["Tk"], style="step", line_kwargs=pv_kwargs, label=label)
    plt.title("Jacket Temperature")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [°C]")
    plt.grid(True)

    ax = plt.subplot(3, 2, 6)
    plot_triplet_series(ax, sensor_hist["Tj_in"], style="step", line_kwargs=pv_kwargs, label=label)
    plt.title("Jacket Inlet Temperature")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [°C]")
    plt.grid(True)

    plt.tight_layout()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mpl.set_loglevel("warning")

    systems = make_systems()
    for label, system in systems.items():
        system.step(nsteps=12_000)
        plot_results(label, system)

    for idx in range(6):
        plt.subplot(3, 2, idx + 1)
        plt.legend(systems.keys())

    plt.show()
