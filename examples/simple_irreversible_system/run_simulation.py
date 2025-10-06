import matplotlib.pyplot as plt

try:
    from .system_definitions import (
        IrreversibleStates,
        IrreversibleControlElements,
        IrreversibleAlgebraicStates,
        IrreversibleConstants,
        IrreversibleSystem,
    )
except ImportError:  # pragma: no cover - support direct script execution
    from system_definitions import (  # type: ignore
        IrreversibleStates,
        IrreversibleControlElements,
        IrreversibleAlgebraicStates,
        IrreversibleConstants,
        IrreversibleSystem,
    )
from modular_simulation.usables import SampledDelayedSensor
from modular_simulation.plotting import plot_triplet_series
from modular_simulation.framework import create_system
from modular_simulation.control_system import Trajectory, PIDController
from typing import List, TYPE_CHECKING
import logging
import matplotlib as mpl
from astropy.units import Unit, UnitBase

if TYPE_CHECKING:
    from modular_simulation.usables import Calculation


def make_systems():

    # 1. Set up the initial conditions and system components.
    # =======================================================

    # Initial values for the differential states. Note F_out is now an algebraic state.
    initial_states = IrreversibleStates(V=0.0, A=0.0, B=0.0)

    # Initial values for the control elements.
    initial_controls = IrreversibleControlElements(F_in=0.0)

    # Initial values for algebraic states (can be placeholders, they are calculated).
    initial_algebraic = IrreversibleAlgebraicStates(F_out=0.0)

    # Define the system's physical constants and solver params
    system_constants = IrreversibleConstants(
        k = 1e-3, 
        Cv = 1e-1, 
        CA_in = 1.0
    )
    # The MeasurableQuantities object now holds all state-like data.


    # Define which quantities can be measured by sensors.
    sensors=[
        SampledDelayedSensor(
            measurement_tag = "F_out",
            unit = Unit("L/s"),
        ),
        SampledDelayedSensor(
            measurement_tag = "F_in",
            unit = Unit("L/s"),
            coefficient_of_variance=0.05
        ),
        SampledDelayedSensor(
            measurement_tag = "B",
            unit = Unit("mol/L"),
            coefficient_of_variance=0.05,
            sampling_period = 900,
            deadtime = 900,
        ),
        SampledDelayedSensor(
            measurement_tag = "V",
            unit = Unit("L"),
            faulty_probability = 0.01,
            faulty_aware = True
        ),
    ]
    calculations: List["Calculation"] = []

    # Define the controllers that will manipulate the control elements.
    controllers=[
        PIDController(
            cv_tag="B",
            mv_tag = "F_in",
            sp_trajectory=Trajectory(0.5),
            sp_unit=Unit("mol/L"),
            Kp=1.0e-1,
            Ti=100.0,
            mv_range = (0, 100.)
        )
    ]




    # --- 2. Assemble and Initialize the System ---
    dt = 30.
    normal_system = create_system(
        dt = dt,
        system_class = IrreversibleSystem,
        initial_states = initial_states,
        initial_controls = initial_controls,
        initial_algebraic = initial_algebraic,
        sensors = sensors,
        calculations = calculations,
        controllers = controllers,
        system_constants = system_constants,
        use_numba = False
    )

    fast_system = create_system(
        dt=dt,
        system_class=IrreversibleSystem,
        initial_states=initial_states,
        initial_controls=initial_controls,
        initial_algebraic=initial_algebraic,
        sensors=sensors,
        calculations=calculations,
        controllers=controllers,
        system_constants=system_constants,
        use_numba = True
    )

    return {"fast": fast_system, "normal": normal_system}


systems = make_systems()
fast_system = systems["fast"]
normal_system = systems["normal"]
if __name__ == "__main__":

    logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
    )
    mpl.set_loglevel("warning") # silence matplotlib debug messages
    
    
    pv_kwargs = {'linestyle': '-'}
    sp_kwargs = {'linestyle': '--', "alpha": 0.5}
    systems = make_systems()
    for j, (system_name, system) in enumerate(systems.items()):
        plt.figure(figsize=(12, 8))
        # --- First simulation run ---
        system.step(nsteps = 5000) #type: ignore
        # --- Change the setpoint and continue the simulation ---
        system.extend_controller_trajectory(cv_tag = "B", value = 0.2)
        system.step(nsteps = 5000)  #type: ignore
            

        # 3. Plot the results.
        # =====================

        # The simulation history is stored in the system's `_history` attribute.
        history = system.measured_history  #type: ignore
        sensor_hist = history["sensors"]
        calc_hist = history["calculations"]
        sp_hist= system.setpoint_history
        # Plot Concentration of B
        ax = plt.subplot(2, 2, 1)
        plot_triplet_series(
            ax,
            sensor_hist["B"],
            style="step",
            line_kwargs=pv_kwargs,
            label=system_name,
        )
        plot_triplet_series(
            ax,
            sp_hist["B"],
            style="step",
            line_kwargs=sp_kwargs,
            label=system_name,
        )
        plt.title("Concentration of B")
        plt.xlabel("Time ")
        plt.ylabel("[B] (mol/L)")
        plt.grid(True)

        # Plot Inlet Flow Rate (F_in)
        ax = plt.subplot(2, 2, 2)
        plot_triplet_series(
            ax,
            sensor_hist["F_in"],
            style="step",
            line_kwargs=pv_kwargs,
            label=system_name,
        )
        plt.title("Inlet Flow Rate (F_in)")
        plt.xlabel("Time ")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        # Plot Reactor Volume (V)
        ax = plt.subplot(2, 2, 3)
        plot_triplet_series(
            ax,
            sensor_hist["V"],
            style="step",
            line_kwargs=pv_kwargs,
            label=system_name,
        )
        plt.title("Reactor Volume (V)")
        plt.xlabel("Time ")
        plt.ylabel("Volume (L)")
        plt.grid(True)

        # Plot Outlet Flow Rate (F_out)
        ax = plt.subplot(2, 2, 4)
        plot_triplet_series(
            ax,
            sensor_hist["F_out"],
            style="step",
            line_kwargs=pv_kwargs,
            label=system_name,
        )
        plt.title("Outlet Flow Rate (F_out)")
        plt.xlabel("Time ")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)
        plt.tight_layout()

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.legend(systems.keys())

    
    plt.show()
