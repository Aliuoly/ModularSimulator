import matplotlib.pyplot as plt
from system_definitions import (
    IrreversibleStates,
    IrreversibleControlElements,
    IrreversibleAlgebraicStates,
    IrreversibleConstants,
    IrreversibleSystem
)
from modular_simulation.usables import SampledDelayedSensor
from modular_simulation.plotting import plot_triplet_series
from modular_simulation.system import create_system
from modular_simulation.control_system import Trajectory, PIDController
from typing import List, TYPE_CHECKING
import logging
import matplotlib as mpl

if TYPE_CHECKING:
    from modular_simulation.usables import Calculation



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
    ),
    SampledDelayedSensor(
        measurement_tag = "F_in",
        coefficient_of_variance=0.05
    ),
    SampledDelayedSensor(
        measurement_tag = "B",
        coefficient_of_variance=0.05,
        sampling_period = 900,
        deadtime = 900,
    ),
    SampledDelayedSensor(
        measurement_tag = "V",
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
        Kp=1.0e-1,
        Ti=100.0,
        mv_range = (0, 100.)
    )
]




# --- 2. Assemble and Initialize the System ---
dt = 30.
readable_system = create_system(
    dt = dt,
    system_class = IrreversibleSystem,
    initial_states = initial_states,
    initial_controls = initial_controls,
    initial_algebraic = initial_algebraic,
    sensors = sensors,
    calculations = calculations,
    controllers = controllers,
    system_constants = system_constants,
)

# fast_system = create_system(
#     system_class = IrreversibleFastSystem,
#     initial_states = initial_states,
#     initial_controls = initial_controls,
#     initial_algebraic = initial_algebraic,
#     sensors = sensors,
#     calculations = calculations,
#     controllers = controllers,
#     system_constants = system_constants,
#     solver_options = solver_options,
# )

# --- 3. Run the Simulation ---
systems = {"readable": readable_system}
if __name__ == "__main__":

    logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
    )
    mpl.set_loglevel("warning") # silence matplotlib debug messages
    plt.figure(figsize=(12, 8))
    linestyles = ['-', '--']
    for j, (system_name, system) in enumerate(systems.items()):
        # --- First simulation run ---
        for i in range(5000):
            system.step() #type: ignore

        # --- Change the setpoint and continue the simulation ---
        # Access the controller via the controllable_quantities object.
        system.extend_controller_trajectory(cv_tag = "B", value = 0.2)
        for i in range(5000):
            system.step()  #type: ignore

        # 3. Plot the results.
        # =====================

        # The simulation history is stored in the system's `_history` attribute.
        history = system.measured_history  #type: ignore
        sensor_hist = history["sensors"]
        calc_hist = history["calculations"]
        # Plot Concentration of B
        ax = plt.subplot(2, 2, 1)
        plot_triplet_series(
            ax,
            sensor_hist["B"],
            style="step",
            line_kwargs={"linestyle": linestyles[j]},
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
            line_kwargs={"linestyle": linestyles[j]},
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
            line_kwargs={"linestyle": linestyles[j]},
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
            line_kwargs={"linestyle": linestyles[j]},
            label=system_name,
        )
        plt.title("Outlet Flow Rate (F_out)")
        plt.xlabel("Time ")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.legend(systems.keys())

    plt.tight_layout()
    plt.show()
