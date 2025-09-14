import matplotlib.pyplot as plt
from modular_simulation.quantities import MeasurableQuantities, UsableQuantities, ControllableQuantities
from system_definitions import (
    IrreversibleStates,
    IrreversibleControlElements,
    IrreversibleAlgebraicStates,
    FlowInSensor,
    FlowOutSensor,
    BConcentrationSensor,
    VolumeSensor,
    PIDController,
    ConstantTrajectory,

    IrreversibleSystem,
    IrreversibleFastSystem
)
from copy import deepcopy

# 1. Set up the initial conditions and system components.
# =======================================================

# Initial values for the differential states. Note F_out is now an algebraic state.
initial_states = IrreversibleStates(V=0.0, A=0.0, B=0.0)

# Initial values for the control elements.
initial_controls = IrreversibleControlElements(F_in=0.0)

# Initial values for algebraic states (can be placeholders, they are calculated).
initial_algebraic = IrreversibleAlgebraicStates(F_out=0.0)

# The MeasurableQuantities object now holds all state-like data.
measurable_quantities = MeasurableQuantities(
    states=initial_states,
    control_elements=initial_controls,
    algebraic_states=initial_algebraic
)

# Define which quantities can be measured by sensors.
usable_quantities = UsableQuantities(
    measurement_definitions={
        'F_out': FlowOutSensor(),
        'F_in': FlowInSensor(),
        'B': BConcentrationSensor(),
        'V': VolumeSensor(),
    },
    calculation_definitions={}
)

# Define the controllers that will manipulate the control elements.
controllable_quantities = ControllableQuantities(
    control_definitions={
        'F_in': PIDController(
            pv_tag="B",
            sp_trajectory=ConstantTrajectory(0.5),
            Kp=1.0e-1,
            Ti=100.0
        )
    }
)

# Define the system's physical constants and solver params
system_constants = {'k': 1e-3, 'Cv': 1e-1, 'CA_in': 1.0}
solver_options = {'method': 'LSODA'}

# --- 2. Assemble and Initialize the System ---

readable_system = IrreversibleSystem(
    measurable_quantities=deepcopy(measurable_quantities),
    usable_quantities=deepcopy(usable_quantities),
    controllable_quantities=deepcopy(controllable_quantities),
    system_constants=deepcopy(system_constants),
    solver_options=deepcopy(solver_options)
)

# have to use a copy of all the arguments since they are by-refernce objects
# otherwise, when I iterate over the other system, this one will change too.
fast_system = IrreversibleFastSystem(
    measurable_quantities=measurable_quantities,
    usable_quantities=usable_quantities,
    controllable_quantities=controllable_quantities,
    system_constants=system_constants,
    solver_options=solver_options
)

# --- 3. Run the Simulation ---
systems = {"readable": readable_system, "fast": fast_system}
if __name__ == "__main__":
    plt.figure(figsize=(12, 8))
    linestyles = ['-', '--']
    j = 0
    for system in systems.values():
        # --- First simulation run ---
        for i in range(5000):
            system.step(30)

        # --- Change the setpoint and continue the simulation ---
        # Access the controller via the controllable_quantities object.
        pid_controller = system.controllable_quantities.control_definitions["F_in"]
        pid_controller.sp_trajectory.change(0.2)

        for i in range(5000):
            system.step(30)

        # 3. Plot the results.
        # =====================

        # The simulation history is stored in the system's `_history` attribute.
        history = system._history

        # Plot Concentration of B
        plt.subplot(2, 2, 1)
        plt.plot([s["B"] for s in history], linestyle = linestyles[j])
        plt.title("Concentration of B")
        plt.xlabel("Time Step")
        plt.ylabel("[B] (mol/L)")
        plt.grid(True)

        # Plot Inlet Flow Rate (F_in)
        plt.subplot(2, 2, 2)
        plt.plot([s['F_in'] for s in history], linestyle = linestyles[j])
        plt.title("Inlet Flow Rate (F_in)")
        plt.xlabel("Time Step")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        # Plot Reactor Volume (V)
        plt.subplot(2, 2, 3)
        plt.plot([s['V'] for s in history], linestyle = linestyles[j])
        plt.title("Reactor Volume (V)")
        plt.xlabel("Time Step")
        plt.ylabel("Volume (L)")
        plt.grid(True)

        # Plot Outlet Flow Rate (F_out)
        plt.subplot(2, 2, 4)
        plt.plot([s['F_out'] for s in history], linestyle = linestyles[j])
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
