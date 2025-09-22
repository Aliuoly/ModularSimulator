import logging
from typing import List, TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

from modular_simulation.control_system import PIDController, Trajectory, CascadeController
from modular_simulation.plotting import plot_triplet_series
from modular_simulation.system import create_system
from modular_simulation.usables import SampledDelayedSensor

from system_definitions import (
    EnergyBalanceAlgebraicStates,
    EnergyBalanceConstants,
    EnergyBalanceControlElements,
    EnergyBalanceFastSystem,
    EnergyBalanceStates,
    EnergyBalanceSystem,
)

if TYPE_CHECKING:
    from modular_simulation.usables import Calculation


# 1. Set up the initial conditions and system components.
# =======================================================

initial_states = EnergyBalanceStates(V=100.0, A=1.0, B=0.0, T=350.0, T_J=300.0)
initial_controls = EnergyBalanceControlElements(F_in=1.0, T_J_in=300)
initial_algebraic = EnergyBalanceAlgebraicStates(F_out=1.0)

sensors = [
    SampledDelayedSensor(measurement_tag="F_out"),
    SampledDelayedSensor(measurement_tag="F_in", coefficient_of_variance=0.05),
    SampledDelayedSensor(
        measurement_tag="B",
        coefficient_of_variance=0.05,
        sampling_period=900.0,
        deadtime=900.0,
    ),
    SampledDelayedSensor(measurement_tag="V"),
    SampledDelayedSensor(measurement_tag="T"),
    SampledDelayedSensor(measurement_tag="T_J"),
    SampledDelayedSensor(measurement_tag="T_J_in"),
    SampledDelayedSensor(measurement_tag="jacket_flow"),
]

calculations: List["Calculation"] = []

controllers = [
    PIDController(
        cv_tag="V",
        mv_tag="F_in",
        sp_trajectory=Trajectory(1.0e3),
        Kp=1.0e-2,
        Ti=100.0,
        mv_range=(0.0, 1.0e6),
    ),
    CascadeController( # control B by controlling T (inner loop), which is controlled by F_J_in (inner loop)
        inner_loop = PIDController(
            cv_tag="T",
            mv_tag="T_J_in",
            sp_trajectory=Trajectory(0.5),
            Kp=1.0e1,
            Ti=100.0,
            mv_range=(200, 300),
            inverted=False,
        ),
        outer_loop = PIDController(
            cv_tag="B",
            mv_tag="T",
            sp_trajectory=Trajectory(0.02),
            Kp=1.0e1,
            Ti=100.0,
            mv_range=(250.0, 350.0),
            inverted=False,
        )
    )
    
]

system_constants = EnergyBalanceConstants(
    k0=1.5e9,
    activation_energy=72500.0,
    gas_constant=8.314,
    Cv=2.0,
    CA_in=2.0,
    T_in=300.0,
    reaction_enthalpy=825000.0,
    rho_cp=4000.0,
    overall_heat_transfer_coefficient=500000.0,
    heat_transfer_area=10.0,
    jacket_volume=2000.0,
    jacket_rho_cp=1200.0,
    jacket_flow = 500.0,
)


# --- 2. Assemble and Initialize the System ---
dt = 30.0
readable_system = create_system(
    dt=dt,
    system_class=EnergyBalanceSystem,
    initial_states=initial_states,
    initial_controls=initial_controls,
    initial_algebraic=initial_algebraic,
    sensors=sensors,
    calculations=calculations,
    controllers=controllers,
    system_constants=system_constants,
)

# fast_system = create_system(
#     dt=dt,
#     system_class=EnergyBalanceFastSystem,
#     initial_states=initial_states,
#     initial_controls=initial_controls,
#     initial_algebraic=initial_algebraic,
#     sensors=sensors,
#     calculations=calculations,
#     controllers=controllers,
#     system_constants=system_constants,
# )


# --- 3. Run the Simulation ---
systems = {"readable": readable_system}

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    mpl.set_loglevel("warning")
    plt.figure(figsize=(14, 10))
    linestyles = ["-", "--"]

    for j, (name, system) in enumerate(systems.items()):
        for _ in range(500):
            system.step()

        system.extend_controller_trajectory(cv_tag="B", value=0.005)

        for _ in range(500):
            system.step()

        history = system.measured_history
        sensor_hist = history["sensors"]

        ax = plt.subplot(4, 2, 1)
        plot_triplet_series(
            ax,
            sensor_hist["B"],
            style="step",
            line_kwargs={"linestyle": linestyles[j]},
            label=name,
        )
        plt.title("Concentration of B")
        plt.xlabel("Time")
        plt.ylabel("[B] (mol/L)")
        plt.grid(True)

        ax = plt.subplot(4, 2, 2)
        plot_triplet_series(
            ax,
            sensor_hist["F_in"],
            style="step",
            line_kwargs={"linestyle": linestyles[j]},
            label=name,
        )
        plt.title("Inlet Flow Rate (F_in)")
        plt.xlabel("Time")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        ax = plt.subplot(4, 2, 3)
        plot_triplet_series(
            ax,
            sensor_hist["V"],
            style="step",
            line_kwargs={"linestyle": linestyles[j]},
            label=name,
        )
        plt.title("Reactor Volume (V)")
        plt.xlabel("Time")
        plt.ylabel("Volume (L)")
        plt.grid(True)

        ax = plt.subplot(4, 2, 4)
        plot_triplet_series(
            ax,
            sensor_hist["F_out"],
            style="step",
            line_kwargs={"linestyle": linestyles[j]},
            label=name,
        )
        plt.title("Outlet Flow Rate (F_out)")
        plt.xlabel("Time")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        ax = plt.subplot(4, 2, 5)
        plot_triplet_series(
            ax,
            sensor_hist["T"],
            style="step",
            line_kwargs={"linestyle": linestyles[j]},
            label=name,
        )
        plt.title("Reactor Temperature (T)")
        plt.xlabel("Time")
        plt.ylabel("Temperature (K)")
        plt.grid(True)

        ax = plt.subplot(4, 2, 6)
        plot_triplet_series(
            ax,
            sensor_hist["T_J"],
            style="step",
            line_kwargs={"linestyle": linestyles[j]},
            label=name,
        )
        plt.title("Jacket Temperature (T_J)")
        plt.xlabel("Time")
        plt.ylabel("Temperature (K)")
        plt.grid(True)

        ax = plt.subplot(4, 2, 7)
        plot_triplet_series(
            ax,
            sensor_hist["T_J_in"],
            style="step",
            line_kwargs={"linestyle": linestyles[j]},
            label=name,
        )
        plt.title("Jacket Inlet Temp (T_J_in)")
        plt.xlabel("Time")
        plt.ylabel("Temperature (K)")
        plt.grid(True)

        ax = plt.subplot(4, 2, 8)
        plot_triplet_series(
            ax,
            sensor_hist["jacket_flow"],
            style="step",
            line_kwargs={"linestyle": linestyles[j]},
            label=name,
        )
        plt.title("Jacket Inlet Flow (F_J_in)")
        plt.xlabel("Time")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

    for idx in range(8):
        plt.subplot(4, 2, idx + 1)
        plt.legend()

    plt.tight_layout()
    plt.show()
