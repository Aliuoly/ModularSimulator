"""Simulation entry-point for the energy balance example."""

from __future__ import annotations

import logging
from functools import partial

import matplotlib.pyplot as plt
import matplotlib as mpl

from modular_simulation.framework import System
from modular_simulation.plotting import plot_triplet_series
from modular_simulation.utils.wrappers import day, second, second_value

from process_definition import EnergyBalanceProcessModel
from component_definition import sensors, calculations, controllers


# --- Assemble the system ----------------------------------------------------
dt = second(30)
system = System(
    dt=dt,
    process_model=EnergyBalanceProcessModel(),
    sensors=sensors,
    calculations=calculations,
    controllers=controllers,
)


def plot(system: System) -> None:
    history = system.measured_history
    sensor_hist = history["sensors"]
    sp_hist = system.setpoint_history

    plt.figure(figsize=(12, 14))
    pv_kwargs = {"linestyle": "-"}
    sp_kwargs = {"linestyle": "--", "alpha": 0.5}
    plot_in_hours = partial(plot_triplet_series, time_converter=lambda t: t / second_value(day(1)) * 24)

    ax = plt.subplot(4, 2, 1)
    plot_in_hours(ax, sensor_hist["F_in"], style="step", line_kwargs=pv_kwargs)
    plt.title("Inlet Flow Rate (F_in)")
    plt.xlabel("Time [h]")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 2)
    plot_in_hours(ax, sensor_hist["F_out"], style="step", line_kwargs=pv_kwargs)
    plt.title("Outlet Flow Rate (F_out)")
    plt.xlabel("Time [h]")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 3)
    plot_in_hours(ax, sensor_hist["V"], style="step", line_kwargs=pv_kwargs)
    plot_in_hours(ax, sp_hist["V.sp"], style="step", line_kwargs=sp_kwargs)
    plt.title("Reactor Volume (V)")
    plt.xlabel("Time [h]")
    plt.ylabel("Volume (L)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 4)
    plot_in_hours(ax, sensor_hist["B"], style="step", line_kwargs=pv_kwargs)
    plt.title("Concentration of B")
    plt.xlabel("Time [h]")
    plt.ylabel("mol/L")
    plt.grid(True)

    ax = plt.subplot(4, 2, 5)
    plot_in_hours(ax, sensor_hist["T"], style="step", line_kwargs=pv_kwargs)
    plot_in_hours(ax, sp_hist["T.sp"], style="step", line_kwargs=sp_kwargs)
    plt.title("Reactor Temperature (T)")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature (K)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 6)
    plot_in_hours(ax, sensor_hist["T_J"], style="step", line_kwargs=pv_kwargs)
    plot_in_hours(ax, sp_hist["T_J.sp"], style="step", line_kwargs=sp_kwargs)
    plt.title("Jacket Temperature (T_J)")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature (K)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 7)
    plot_in_hours(ax, sensor_hist["T_J_in"], style="step", line_kwargs=pv_kwargs)
    plt.title("Jacket Inlet Temperature (T_J_in)")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature (K)")
    plt.grid(True)

    ax = plt.subplot(4, 2, 8)
    plot_in_hours(ax, sensor_hist["jacket_flow"], style="step", line_kwargs=pv_kwargs)
    plt.title("Jacket Flow Rate")
    plt.xlabel("Time [h]")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    for idx in range(8):
        plt.subplot(4, 2, idx + 1)
        plt.legend()

    plt.tight_layout()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mpl.set_loglevel("warning")

    system.step(duration=day(1))
    plot(system)
    plt.show()
