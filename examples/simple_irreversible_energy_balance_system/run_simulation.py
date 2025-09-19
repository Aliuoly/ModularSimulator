import matplotlib.pyplot as plt
from typing import Dict, TYPE_CHECKING

from system_definitions import (
    EnergyBalanceStates,
    EnergyBalanceControlElements,
    EnergyBalanceAlgebraicStates,
    PIDController,
    ConstantTrajectory,
    EnergyBalanceSystem,
    EnergyBalanceFastSystem,
)
from modular_simulation.usables import SampledDelayedSensor
from modular_simulation.system import create_system

if TYPE_CHECKING:
    from modular_simulation.usables import Calculation
    from modular_simulation.control_system import Controller


# 1. Set up the initial conditions and system components.
# =======================================================

initial_states = EnergyBalanceStates(V=1.0, A=1.0, B=0.0, T=350.0, T_J=300.0)
initial_controls = EnergyBalanceControlElements(F_in=0.5, F_J_in = 0.)
initial_algebraic = EnergyBalanceAlgebraicStates(F_out=0.0)

measurement_definitions = {
    "F_out": SampledDelayedSensor(
        measurement_tag="F_out",
    ),
    "F_in": SampledDelayedSensor(
        measurement_tag="F_in",
        coefficient_of_variance=0.05,
    ),
    "B": SampledDelayedSensor(
        measurement_tag="B",
        coefficient_of_variance=0.05,
        sampling_period=900,
        deadtime=900,
    ),
    "V": SampledDelayedSensor(
        measurement_tag="V",
    ),
    "T": SampledDelayedSensor(
        measurement_tag="T",
    ),
    "T_J": SampledDelayedSensor(
        measurement_tag="T_J",
    ),
    "F_J_in": SampledDelayedSensor(
        measurement_tag="F_J_in",
    ),
}

calculation_definitions: Dict[str, "Calculation"] = {}

control_definitions: Dict[str, "Controller"] = {
    "F_in": PIDController(
        pv_tag="V",
        sp_trajectory=ConstantTrajectory(1.e3),
        Kp=1.0e-2,
        Ti=100.0,
    ),
    "F_J_in": PIDController(
        pv_tag="B",
        sp_trajectory = ConstantTrajectory(0.5),
        Kp = 1.0e6,
        Ti = 1.0,
        inverted = True,
    )
}

system_constants = {
    # --- Parameters We Just Derived ---
    "k0": 1.5e9,                   # Pre-exponential factor [1/min]
    "activation_energy": 72500.0,  # Activation energy [J/mol]
    "reaction_enthalpy": 825000.0,# Reaction enthalpy [J/mol] (exothermic)

    # --- Assumed Physical & Design Constants ---
    "Cv": 1.2,                    # Outlet valve constant [L^0.5/min]
    "CA_in": 2.0,                  # Inlet concentration of A [mol/L]
    "T_in": 300.0,                 # Inlet feed temperature [K]
    "gas_constant": 8.314,         # J/(mol.K)
    
    # Heat Transfer Properties
    "rho_cp": 4000.0,              # Density * Heat Capacity of reactor contents [J/(L.K)]
    "overall_heat_transfer_coefficient": 500000.0, # [W/(m^2.K)] -> converted to J/(min.m^2.K) in code
    "heat_transfer_area": 10.0,    # [m^2]

    # Jacket Properties
    "jacket_volume": 200.0,        # [L]
    "jacket_rho_cp": 4200.0,       # [J/(L.K)]
    "jacket_inlet_temperature": 290.0, # Coolant inlet temp [K]
}

solver_options = {"method": "LSODA"}


readable_system = create_system(
    system_class=EnergyBalanceSystem,
    initial_states=initial_states,
    initial_controls=initial_controls,
    initial_algebraic=initial_algebraic,
    measurement_definitions=measurement_definitions,
    calculation_definitions=calculation_definitions,
    control_definitions=control_definitions,
    system_constants=system_constants,
    solver_options=solver_options,
)

fast_system = create_system(
    system_class=EnergyBalanceFastSystem,
    initial_states=initial_states,
    initial_controls=initial_controls,
    initial_algebraic=initial_algebraic,
    measurement_definitions=measurement_definitions,
    calculation_definitions=calculation_definitions,
    control_definitions=control_definitions,
    system_constants=system_constants,
    solver_options=solver_options,
)


# --- 3. Run the Simulation ---
dt = 30
systems = {"readable": readable_system, "fast": fast_system}

if __name__ == "__main__":
    plt.figure(figsize=(14, 10))
    linestyles = ["-", "--"]

    for j, system in enumerate(systems.values()):
        for _ in range(5000):
            system.step(dt)  # type: ignore

        # not ideal - I want to change sp for B, but I have to refer to the F_J_in controller instead.
        pid_controller = system.controllable_quantities.control_definitions["F_J_in"]  # type: ignore
        pid_controller.sp_trajectory.change(0.2)

        for _ in range(5000):
            system.step(dt)  # type: ignore

        history = system.measured_history  # type: ignore
        t = history["time"]

        plt.subplot(4, 2, 1)
        plt.step(t, history["B"], linestyle=linestyles[j])
        plt.title("Concentration of B")
        plt.xlabel("Time Step")
        plt.ylabel("[B] (mol/L)")
        plt.grid(True)

        plt.subplot(4, 2, 2)
        plt.step(t, history["F_in"], linestyle=linestyles[j])
        plt.title("Inlet Flow Rate (F_in)")
        plt.xlabel("Time Step")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        plt.subplot(4, 2, 3)
        plt.step(t, history["V"], linestyle=linestyles[j])
        plt.title("Reactor Volume (V)")
        plt.xlabel("Time Step")
        plt.ylabel("Volume (L)")
        plt.grid(True)

        plt.subplot(4, 2, 4)
        plt.step(t, history["F_out"], linestyle=linestyles[j])
        plt.title("Outlet Flow Rate (F_out)")
        plt.xlabel("Time Step")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        plt.subplot(4, 2, 5)
        plt.step(t, history["T"], linestyle=linestyles[j])
        plt.title("Reactor Temperature (T)")
        plt.xlabel("Time Step")
        plt.ylabel("Temperature (K)")
        plt.grid(True)

        plt.subplot(4, 2, 6)
        plt.step(t, history["T_J"], linestyle=linestyles[j])
        plt.title("Jacket Temperature (T_J)")
        plt.xlabel("Time Step")
        plt.ylabel("Temperature (K)")
        plt.grid(True)

        plt.subplot(4, 2, 8)
        plt.step(t, history["F_J_in"], linestyle=linestyles[j])
        plt.title("Jacket inflow (F_J_in)")
        plt.xlabel("Time Step")
        plt.ylabel("flow (L/s)")
        plt.grid(True)

    for i in range(8):
        plt.subplot(4, 2, i + 1)
        plt.legend(systems.keys())

    plt.tight_layout()
    plt.show()
