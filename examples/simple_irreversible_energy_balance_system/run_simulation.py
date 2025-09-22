import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, List


from system_definitions import (
    EnergyBalanceStates,
    EnergyBalanceControlElements,
    EnergyBalanceAlgebraicStates,
    PIDController,
    EnergyBalanceSystem,
    EnergyBalanceFastSystem,
)
from modular_simulation.usables import SampledDelayedSensor
from modular_simulation.system import create_system
from modular_simulation.control_system import Trajectory

if TYPE_CHECKING:
    from modular_simulation.usables import Calculation


# 1. Set up the initial conditions and system components.
# =======================================================

initial_states = EnergyBalanceStates(V=100.0, A=1.0, B=0.0, T=350.0, T_J=300.0)
initial_controls = EnergyBalanceControlElements(F_in=1.0, F_J_in = 1.)
initial_algebraic = EnergyBalanceAlgebraicStates(F_out=1.0)

sensors = [
    SampledDelayedSensor(
        measurement_tag="F_out",
    ),
    SampledDelayedSensor(
        measurement_tag="F_in",
        coefficient_of_variance=0.05,
    ),
    SampledDelayedSensor(
        measurement_tag="B",
        coefficient_of_variance=0.05,
        sampling_period=900,
        deadtime=900,
    ),
    SampledDelayedSensor(
        measurement_tag="V",
    ),
    SampledDelayedSensor(
        measurement_tag="T",
    ),
    SampledDelayedSensor(
        measurement_tag="T_J",
    ),
    SampledDelayedSensor(
        measurement_tag="F_J_in",
    ),
]

calculations: List["Calculation"] = []

controllers = [
    PIDController(
        cv_tag = "V",
        mv_tag = "F_in",
        sp_trajectory=Trajectory(1.e3),
        Kp=1.0e-2,
        Ti=100.0,
        mv_range = (0., 1.e6)
    ),
    PIDController(
        cv_tag = "B",
        mv_tag = "F_J_in",
        sp_trajectory = Trajectory(0.5),
        Kp = 1.0e6,
        Ti = 1.0,
        mv_range = (274., 350.),
        inverted = True,
    ),
]

system_constants = {
    # --- Parameters We Just Derived ---
    "k0": 1.5e9,                   # Pre-exponential factor [1/min]
    "activation_energy": 72500.0,  # Activation energy [J/mol]
    "reaction_enthalpy": 825000.0,# Reaction enthalpy [J/mol] (exothermic)

    # --- Assumed Physical & Design Constants ---
    "Cv": 2.,                    # Outlet valve constant [L^0.5/min]
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
    sensors=sensors,
    calculations=calculations,
    controllers=controllers,
    system_constants=system_constants,
    solver_options=solver_options,
)

fast_system = create_system(
    system_class=EnergyBalanceFastSystem,
    initial_states=initial_states,
    initial_controls=initial_controls,
    initial_algebraic=initial_algebraic,
    sensors=sensors,
    calculations=calculations,
    controllers=controllers,
    system_constants=system_constants,
    solver_options=solver_options,
)


# --- 3. Run the Simulation ---
dt = 30
systems = {"readable": readable_system}

if __name__ == "__main__":
    plt.figure(figsize=(14, 10))
    linestyles = ["-", "--"]

    for j, system in enumerate(systems.values()):
        for _ in range(5000):
            system.step(dt)  # type: ignore

        system.extend_controller_trajectory(cv_tag = "B", value = 0.2)

        for _ in range(5000):
            system.step(dt)  # type: ignore

        history = system.measured_history  # type: ignore
        sensor_hist = history["sensors"]
        calc_hist = history["calculations"]

        plt.subplot(4, 2, 1)
        hist = sensor_hist["B"]
        plt.step(hist['time'], hist["value"], linestyle=linestyles[j])
        plt.title("Concentration of B")
        plt.xlabel("Time Step")
        plt.ylabel("[B] (mol/L)")
        plt.grid(True)

        plt.subplot(4, 2, 2)
        hist = sensor_hist["F_in"]
        plt.step(hist['time'], hist["value"], linestyle=linestyles[j])
        plt.title("Inlet Flow Rate (F_in)")
        plt.xlabel("Time Step")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        plt.subplot(4, 2, 3)
        hist = sensor_hist["V"]
        plt.step(hist['time'], hist["value"], linestyle=linestyles[j])
        plt.title("Reactor Volume (V)")
        plt.xlabel("Time Step")
        plt.ylabel("Volume (L)")
        plt.grid(True)

        plt.subplot(4, 2, 4)
        hist = sensor_hist["F_out"]
        plt.step(hist['time'], hist["value"], linestyle=linestyles[j])
        plt.title("Outlet Flow Rate (F_out)")
        plt.xlabel("Time Step")
        plt.ylabel("Flow (L/s)")
        plt.grid(True)

        plt.subplot(4, 2, 5)
        hist = sensor_hist["T"]
        plt.step(hist['time'], hist["value"], linestyle=linestyles[j])
        plt.title("Reactor Temperature (T)")
        plt.xlabel("Time Step")
        plt.ylabel("Temperature (K)")
        plt.grid(True)

        plt.subplot(4, 2, 6)
        hist = sensor_hist["T_J"]
        plt.step(hist['time'], hist["value"], linestyle=linestyles[j])
        plt.title("Jacket Temperature (T_J)")
        plt.xlabel("Time Step")
        plt.ylabel("Temperature (K)")
        plt.grid(True)

        plt.subplot(4, 2, 8)
        hist = sensor_hist["F_J_in"]
        plt.step(hist['time'], hist["value"], linestyle=linestyles[j])
        plt.title("Jacket inflow (F_J_in)")
        plt.xlabel("Time Step")
        plt.ylabel("flow (L/s)")
        plt.grid(True)

    for i in range(8):
        plt.subplot(4, 2, i + 1)
        plt.legend(systems.keys())

    plt.tight_layout()
    plt.show()
