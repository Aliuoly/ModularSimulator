import matplotlib.pyplot as plt
from system_definitions import (
    IrreversibleStates,
    IrreversibleControlElements,
    IrreversibleAlgebraicStates,
    IrreversibleSystem,
    IrreversibleFastSystem
)
from modular_simulation.usables import SampledDelayedSensor
from modular_simulation.system import create_system
from modular_simulation.control_system import Trajectory, PIDController
from typing import List, TYPE_CHECKING
from numpy import inf
import logging
import matplotlib as mpl

if TYPE_CHECKING:
    from modular_simulation.usables import Calculation

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s"
)
mpl.set_loglevel("warning") # silence matplotlib debug messages

# 1. Set up the initial conditions and system components.
# =======================================================

# Initial values for the differential states. Note F_out is now an algebraic state.
initial_states = IrreversibleStates(V=0.0, A=0.0, B=0.0)

# Initial values for the control elements.
initial_controls = IrreversibleControlElements(F_in=0.0)

# Initial values for algebraic states (can be placeholders, they are calculated).
initial_algebraic = IrreversibleAlgebraicStates(F_out=0.0)

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
        sampling_period = 0,
        deadtime = 0,
    ),
    SampledDelayedSensor(
        measurement_tag = "V",
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

# Define the system's physical constants and solver params
system_constants = {'k': 1e-3, 'Cv': 1e-1, 'CA_in': 1.0}
solver_options = {'method': 'LSODA'}

# --- 2. Assemble and Initialize the System ---

readable_system = create_system(
    system_class = IrreversibleSystem,
    initial_states = initial_states,
    initial_controls = initial_controls,
    initial_algebraic = initial_algebraic,
    sensors = sensors,
    calculations = calculations,
    controllers = controllers,
    system_constants = system_constants,
    solver_options = solver_options,
)

fast_system = create_system(
    system_class = IrreversibleFastSystem,
    initial_states = initial_states,
    initial_controls = initial_controls,
    initial_algebraic = initial_algebraic,
    sensors = sensors,
    calculations = calculations,
    controllers = controllers,
    system_constants = system_constants,
    solver_options = solver_options,
)

# --- 3. Run the Simulation ---
dt = 30
systems = {'fast': fast_system,"readable": readable_system}
if __name__ == "__main__":
    plt.figure(figsize=(12, 8))
    linestyles = ['-', '--']
    j = 0
    for system in systems.values(): 
        # --- First simulation run ---
        for i in range(5000):
            system.step(dt) #type: ignore

        # --- Change the setpoint and continue the simulation ---
        # Access the controller via the controllable_quantities object.
        system.extend_controller_trajectory(cv_tag = "B").hold(duration = inf, value = 0.2)
        for i in range(5000):
            system.step(dt)  #type: ignore

        # 3. Plot the results.
        # =====================

        # The simulation history is stored in the system's `_history` attribute.
        history = system.history  #type: ignore
        t = history['time']
        # Plot Concentration of B
        plt.subplot(2, 2, 1)
        plt.step(t, history['B'], linestyle = linestyles[j])
        plt.title("Concentration of B")
        plt.xlabel("Time Step")
        plt.ylabel("[B] (mol/L)")
        plt.grid(True)

        # Plot Inlet Flow Rate (F_in)
        plt.subplot(2, 2, 2)
        plt.step(t, history['F_in'], linestyle = linestyles[j])
        plt.title("Inlet Flow Rate (F_in)")
        plt.xlabel("Time Step")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        # Plot Reactor Volume (V)
        plt.subplot(2, 2, 3)
        plt.step(t, history['V'], linestyle = linestyles[j])
        plt.title("Reactor Volume (V)")
        plt.xlabel("Time Step")
        plt.ylabel("Volume (L)")
        plt.grid(True)

        # Plot Outlet Flow Rate (F_out)
        plt.subplot(2, 2, 4)
        plt.step(t, history['F_out'], linestyle = linestyles[j])
        plt.title("Outlet Flow Rate (F_out)")
        plt.xlabel("Time Step")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        j+=1

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.legend(systems.keys())

    plt.tight_layout()
    plt.show()
