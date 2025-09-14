
from system_definitions import (
    IrreversableStates, IrreversableControlElements, IrreversableAlgebraicStates,
    IrreversableFastSystem,
    FlowInSensor, FlowOutSensor, BConcentrationSensor, VolumeSensor,
    PIDController, ConstantTrajectory
)
from modular_simulation.quantities import (
    MeasurableQuantities, UsableQuantities, ControllableQuantities
)
import matplotlib.pyplot as plt

# --- 1. Define Initial Conditions and System Components ---

# Initial values for all measurable quantities
measurable_quantities = MeasurableQuantities(
    states=IrreversableStates(V=0.0, A=0.0, B=0.0),
    control_elements=IrreversableControlElements(F_in=0.0),
    algebraic_states=IrreversableAlgebraicStates(F_out=0.0)
)

# Define how to "measure" values from the system for controllers or logging
usable_quantities = UsableQuantities(
    measurement_definitions={
        'F_out': FlowOutSensor(), # Note the new sensor name
        'F_in': FlowInSensor(),
        'B': BConcentrationSensor(),
        'V': VolumeSensor(),
    },
    calculation_definitions={}
)

# Define the controllers that will manipulate the system's control elements
controllable_quantities = ControllableQuantities(
    control_definitions={
        'F_in': PIDController(
            pv_tag="B",
            sp_trajectory=ConstantTrajectory(0.5),
            Kp=1.0e-1,
            Ti=100.0
        ),
    }
)

# Define the system's physical constants and solver params
system_constants = {'k': 1e-3, 'Cv': 1e-1, 'CA_in': 1.0}
solver_options={'method': 'LSODA'}

# --- 2. Assemble and Initialize the System ---

irreversable_system = IrreversableFastSystem(
    measurable_quantities=measurable_quantities,
    usable_quantities=usable_quantities,
    controllable_quantities=controllable_quantities,
    system_constants=system_constants,
    solver_options=solver_options
)

# --- 3. Run the Simulation ---

if __name__ == "__main__":
    # --- First simulation run ---
    for i in range(5000):
        irreversable_system.step(10)

    # --- Change the setpoint and continue the simulation ---
    # Access the controller via the controllable_quantities object.
    pid_controller = irreversable_system.controllable_quantities.control_definitions["F_in"]
    pid_controller.sp_trajectory.change(0.2)

    for i in range(5000):
        irreversable_system.step(10)

    # 3. Plot the results.
    # =====================

    # The simulation history is stored in the system's `_history` attribute.
    history = irreversable_system._history

    plt.figure(figsize=(12, 8))

    # Plot Concentration of B
    plt.subplot(2, 2, 1)
    plt.plot([s["B"] for s in history])
    plt.title("Concentration of B")
    plt.xlabel("Time Step")
    plt.ylabel("[B] (mol/L)")
    plt.grid(True)

    # Plot Inlet Flow Rate (F_in)
    plt.subplot(2, 2, 2)
    plt.plot([s['F_in'] for s in history])
    plt.title("Inlet Flow Rate (F_in)")
    plt.xlabel("Time Step")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    # Plot Reactor Volume (V)
    plt.subplot(2, 2, 3)
    plt.plot([s['V'] for s in history])
    plt.title("Reactor Volume (V)")
    plt.xlabel("Time Step")
    plt.ylabel("Volume (L)")
    plt.grid(True)

    # Plot Outlet Flow Rate (F_out)
    plt.subplot(2, 2, 4)
    plt.plot([s['F_out'] for s in history])
    plt.title("Outlet Flow Rate (F_out)")
    plt.xlabel("Time Step")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
