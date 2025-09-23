import logging
from typing import List, TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

from modular_simulation.framework import create_system
from modular_simulation.usables import SampledDelayedSensor
from modular_simulation.plotting import plot_triplet_series

from system_definitions import (
    ConstantTrajectory,
    HeatDutyCalculation,
    PIController,
    VanDeVusseAlgebraicStates,
    VanDeVusseConstants,
    VanDeVusseControlElements,
    VanDeVusseFastSystem,
    VanDeVusseStates,
    VanDeVusseSystem,
)

if TYPE_CHECKING:
    from modular_simulation.usables import Calculation

# Initial conditions based on the literature steady-state
initial_states = VanDeVusseStates(
    Ca=2.2291,
    Cb=1.0417,
    T=79.591,
    Tk=77.69,
)
initial_controls = VanDeVusseControlElements(Tj_in=77.69)
initial_algebraic = VanDeVusseAlgebraicStates()

system_constants = VanDeVusseConstants(
    F=14.19,  # L/h
    Ca0=5.1,  # mol/L
    T0=104.9,  # °C
    k10=1.287e10,  # 1/h
    E1=9758.3,  # K
    dHr1=4.2,  # kJ/mol
    rho=0.9342,  # kg/L
    Cp=3.01,  # kJ/(kg·K)
    kw=4032.0,  # kJ/(h·K·m^2)
    AR=0.215,  # m^2
    VR=10.0,  # L
    mK=5.0,  # kg
    CpK=2.0,  # kJ/(kg·K)
    Fj=10.0,  # kg/h of jacket fluid
)

sensors = [
    SampledDelayedSensor(measurement_tag="Ca", sampling_period=0.1),
    SampledDelayedSensor(measurement_tag="Cb", sampling_period=0.1),
    SampledDelayedSensor(measurement_tag="T", sampling_period=0.1),
    SampledDelayedSensor(measurement_tag="Tk", sampling_period=0.1),
    SampledDelayedSensor(measurement_tag="Tj_in", sampling_period=0.1),
]

calculations: List["Calculation"] = [
    HeatDutyCalculation(
        output_tag="Qk",
        measured_input_tags=["Tk", "T"],
        calculated_input_tags=[],
        constants={},
        kw=system_constants.kw,
        area=system_constants.AR,
    )
]

controllers = [
    PIController(
        mv_tag="Tj_in",
        cv_tag="T",
        sp_trajectory=ConstantTrajectory(80.0),
        mv_range=(10.0, 120.0),
        Kp=2.0,
        Ti=0.5,
    )
]

# Assemble the systems
dt = 0.01  # hours (~36 seconds)
readable_system = create_system(
    dt=dt,
    system_class=VanDeVusseSystem,
    initial_states=initial_states,
    initial_controls=initial_controls,
    initial_algebraic=initial_algebraic,
    sensors=sensors,
    calculations=calculations,
    controllers=controllers,
    system_constants=system_constants,
)

fast_system = create_system(
    dt=dt,
    system_class=VanDeVusseFastSystem,
    initial_states=initial_states,
    initial_controls=initial_controls,
    initial_algebraic=initial_algebraic,
    sensors=sensors,
    calculations=calculations,
    controllers=controllers,
    system_constants=system_constants,
)

systems = {"readable": readable_system}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mpl.set_loglevel("warning")
    plt.figure(figsize=(12, 10))
    linestyles = ["-", "--"]

    for j, (label, system) in enumerate(systems.items()):
        for _ in range(6000):
            system.step()

        system.extend_controller_trajectory(cv_tag = "T", value = 90)

        for _ in range(6000):
            system.step()

        history = system.measured_history
        sensor_hist = history["sensors"]
        calc_hist = history["calculations"]

        ax = plt.subplot(3, 2, 1)
        plot_triplet_series(
            ax,
            sensor_hist["T"],
            style="step",
            line_kwargs={"linestyle": linestyles[j]},
            label=label,
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
            line_kwargs={"linestyle": linestyles[j]},
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
            line_kwargs={"linestyle": linestyles[j]},
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
            line_kwargs={"linestyle": linestyles[j]},
            label=label,
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
            line_kwargs={"linestyle": linestyles[j]},
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
            line_kwargs={"linestyle": linestyles[j]},
            label=label,
        )
        plt.title("Jacket Inlet Temperature")
        plt.xlabel("Time [h]")
        plt.ylabel("Temperature [°C]")
        plt.grid(True)

    for idx in range(6):
        plt.subplot(3, 2, idx + 1)
        plt.legend()

    plt.tight_layout()
    plt.show()
