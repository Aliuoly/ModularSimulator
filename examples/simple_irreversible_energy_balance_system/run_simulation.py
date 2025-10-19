import logging
from typing import List, TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

from modular_simulation.usables import PIDController, Trajectory
from modular_simulation.plotting import plot_triplet_series
from modular_simulation.framework import create_system
from modular_simulation.usables import SampledDelayedSensor
from astropy.units import Unit #type: ignore


try:
    from .system_definitions import (
        EnergyBalanceAlgebraicStates,
        EnergyBalanceConstants,
        EnergyBalanceControlElements,
        EnergyBalanceStates,
        EnergyBalanceSystem,
    )
except ImportError:  # pragma: no cover - support direct script execution
    from system_definitions import (  # type: ignore
        EnergyBalanceAlgebraicStates,
        EnergyBalanceConstants,
        EnergyBalanceControlElements,
        EnergyBalanceStates,
        EnergyBalanceSystem,
    )

if TYPE_CHECKING:
    from modular_simulation.usables import Calculation


# 1. Set up the initial conditions and system components.
# =======================================================

def make_systems():
    initial_states = EnergyBalanceStates(V=100.0, A=1.0, B=0.0, T=300.0, T_J=300.0)
    initial_controls = EnergyBalanceControlElements(F_in=1.0, T_J_in=300)
    initial_algebraic = EnergyBalanceAlgebraicStates(F_out=1.0)

    sensors = [
        SampledDelayedSensor(measurement_tag="F_out", unit = Unit("cm3/s")),
        SampledDelayedSensor(measurement_tag="F_in", unit = Unit("cm3/s"), coefficient_of_variance=0.05),
        SampledDelayedSensor(
            measurement_tag="B",
            unit = Unit("mol/L"),
            coefficient_of_variance=0.05,
            sampling_period=900.0,
            deadtime=900.0,
        ),
        SampledDelayedSensor(measurement_tag="V", unit = Unit("m3")),
        SampledDelayedSensor(measurement_tag="T", unit = Unit("K")),
        SampledDelayedSensor(measurement_tag="T_J", unit = Unit("K")),
        SampledDelayedSensor(measurement_tag="T_J_in"),
        SampledDelayedSensor(measurement_tag="jacket_flow"),
    ]

    calculations: List["Calculation"] = []

    controllers = [
        PIDController(
            mv_tag="F_in",
            cv_tag="V",
            sp_trajectory=Trajectory(1.0e3, unit = Unit("L")),
            Kp=1.0e-2,
            Ti=100.0,
            mv_range=(0.0 * Unit("L/s"), 1.0e6 * Unit("L/s")),
            ),
        PIDController(
            mv_tag="T_J_in",
            cv_tag="T_J",
            sp_trajectory=Trajectory(300, unit = Unit("K")),
            Kp=1.0e-1,
            Ti=50.0,
            mv_range=(200 * Unit("K"), 350 * Unit("K")),
            cascade_controller = PIDController(
                mv_tag="T_J",
                cv_tag="T",
                sp_trajectory=Trajectory(300, Unit("K")),
                Kp=1.0e-1,
                Ti=100.0,
                mv_range=(200 * Unit("K"), 350 * Unit("K")),
                cascade_controller = PIDController(
                    mv_tag="T",
                    cv_tag="B",
                    sp_trajectory=Trajectory(0.02, unit = Unit("mol/L")).\
                        hold(duration = 15e3).\
                            hold(15e3, 0.05).\
                                hold(15e3, 0.1).\
                                    hold(15e3, 0.01),
                    Kp=2.0e-1,
                    Ti=5.0,
                    mv_range=(250.0 * Unit("K"), 350.0* Unit("K")),
                ),
            ),
        ),
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
        jacket_volume=500000.0,
        jacket_rho_cp=3200.0,
        jacket_flow = 500.0,
    )


    # --- 2. Assemble and Initialize the System ---
    dt = 30.0 * Unit("second")
    system = create_system(
        dt=dt,
        system_class=EnergyBalanceSystem,
        initial_states=initial_states,
        initial_controls=initial_controls,
        initial_algebraic=initial_algebraic,
        sensors=sensors,
        calculations=calculations,
        controllers=controllers,
        system_constants=system_constants,
        use_numba=False
    )

    numba_system = create_system(
        dt=dt,
        system_class=EnergyBalanceSystem,
        initial_states=initial_states,
        initial_controls=initial_controls,
        initial_algebraic=initial_algebraic,
        sensors=sensors,
        calculations=calculations,
        controllers=controllers,
        system_constants=system_constants,
        use_numba=True
    )
    return {"normal": system, "fast": numba_system}

def plot(name, system):
    history = system.measured_history
    sensor_hist = history["sensors"]
    sp_hist = system.setpoint_history
    plt.figure(figsize=(14, 10))
    ax = plt.subplot(4, 2, 1)
    pv_kwargs = {'linestyle': '-'}
    sp_kwargs = {'linestyle': '--', "alpha": 0.5}
    plot_triplet_series(
        ax,
        sensor_hist["B"],
        style="step",
        line_kwargs=pv_kwargs,
        label=name,
    )
    plot_triplet_series(
        ax,
        sp_hist["B"],
        style="step",
        line_kwargs=sp_kwargs,
        label=f"{name} sp",
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
        line_kwargs=pv_kwargs,
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
        line_kwargs=pv_kwargs,
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
        line_kwargs=pv_kwargs,
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
        line_kwargs=pv_kwargs,
        label=name,
    )
    plot_triplet_series(
        ax,
        sp_hist["T"],
        style="step",
        line_kwargs=sp_kwargs,
        label=f"{name} sp",
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
        line_kwargs=pv_kwargs,
        label=name,
    )
    plot_triplet_series(
        ax,
        sp_hist["T_J"],
        style="step",
        line_kwargs=sp_kwargs, 
        label=f"{name} sp",
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
        line_kwargs=pv_kwargs,
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
        line_kwargs=pv_kwargs,
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


systems = make_systems()
normal_system = systems["normal"]
fast_system = systems["fast"]
    



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mpl.set_loglevel("warning")
    # --- 3. Run the Simulation ---
    systems = {
        "normal": normal_system
    }

    for j, (name, system) in enumerate(systems.items()):
        system.step(duration = 1 * Unit("day"))
        plot(name, system)
    
    plt.show()

