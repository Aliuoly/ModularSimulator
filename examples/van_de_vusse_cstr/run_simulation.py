import matplotlib.pyplot as plt
from typing import Dict, TYPE_CHECKING

from modular_simulation.system import create_system
from modular_simulation.usables import SampledDelayedSensor

from system_definitions import (
    ConstantTrajectory,
    HeatDutyCalculation,
    PIController,
    VanDeVusseAlgebraicStates,
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


system_constants = {
    "F": 14.19,  # L/h
    "Ca0": 5.1,  # mol/L
    "T0": 104.9,  # °C
    "k10": 1.287e10,  # 1/h
    "E1": 9758.3,  # K
    "dHr1": 4.2,  # kJ/mol
    "rho": 0.9342,  # kg/L
    "Cp": 3.01,  # kJ/(kg·K)
    "kw": 4032.0,  # kJ/(h·K·m^2)
    "AR": 0.215,  # m^2
    "VR": 10.0,  # L
    "mK": 5.0,  # kg
    "CpK": 2.0,  # kJ/(kg·K)
    "Fj": 10.0,  # kg/h of jacket fluid
}

measurement_definitions = {
    "Ca": SampledDelayedSensor("Ca", sampling_period=0.1),
    "Cb": SampledDelayedSensor("Cb", sampling_period=0.1),
    "T": SampledDelayedSensor("T", sampling_period=0.1),
    "Tk": SampledDelayedSensor("Tk", sampling_period=0.1),
    "Tj_in": SampledDelayedSensor("Tj_in", sampling_period=0.1),
}

calculation_definitions: Dict[str, "Calculation"] = {
    "Qk": HeatDutyCalculation(kw=system_constants["kw"], area=system_constants["AR"])
}

control_definitions = {
    "Tj_in": PIController(
        pv_tag="T",
        sp_trajectory=ConstantTrajectory(80.0),
        Kp=2.0,
        Ti=0.5,
        u_min=10.0,
        u_max=120.0,
    )
}

solver_options = {"method": "LSODA"}

readable_system = create_system(
    system_class=VanDeVusseSystem,
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
    system_class=VanDeVusseFastSystem,
    initial_states=initial_states,
    initial_controls=initial_controls,
    initial_algebraic=initial_algebraic,
    measurement_definitions=measurement_definitions,
    calculation_definitions=calculation_definitions,
    control_definitions=control_definitions,
    system_constants=system_constants,
    solver_options=solver_options,
)

systems = {"readable": readable_system, "fast": fast_system}

dt = 0.01  # hours (~36 seconds)

if __name__ == "__main__":
    plt.figure(figsize=(12, 10))
    linestyles = ["-", "--"]

    for j, (label, system) in enumerate(systems.items()):
        for _ in range(6000):
            system.step(dt)  # type: ignore[arg-type]

        controller = system.controllable_quantities.control_definitions["Tj_in"]
        controller.sp_trajectory.change(85.0)

        for _ in range(6000):
            system.step(dt)  # type: ignore[arg-type]

        history = system.measured_history  # type: ignore[attr-defined]
        time = history["time"]

        plt.subplot(3, 2, 1)
        plt.step(time, history["T"], linestyle=linestyles[j], label=label)
        plt.title("Reactor Temperature")
        plt.xlabel("Time [h]")
        plt.ylabel("Temperature [°C]")
        plt.grid(True)

        plt.subplot(3, 2, 2)
        plt.step(time, history["Qk"], linestyle=linestyles[j], label=label)
        plt.title("Jacket Heat Duty")
        plt.xlabel("Time [h]")
        plt.ylabel("Qk [kJ/h]")
        plt.grid(True)

        plt.subplot(3, 2, 3)
        plt.step(time, history["Ca"], linestyle=linestyles[j], label=label)
        plt.title("Concentration of A")
        plt.xlabel("Time [h]")
        plt.ylabel("[A] [mol/L]")
        plt.grid(True)

        plt.subplot(3, 2, 4)
        plt.step(time, history["Cb"], linestyle=linestyles[j], label=label)
        plt.title("Concentration of B")
        plt.xlabel("Time [h]")
        plt.ylabel("[B] [mol/L]")
        plt.grid(True)

        plt.subplot(3, 2, 5)
        plt.step(time, history["Tk"], linestyle=linestyles[j], label=label)
        plt.title("Jacket Temperature")
        plt.xlabel("Time [h]")
        plt.ylabel("Temperature [°C]")
        plt.grid(True)
        
        plt.subplot(3, 2, 6)
        plt.step(time, history["Tj_in"], linestyle=linestyles[j], label=label)
        plt.title("Jacket Inlet Temperature")
        plt.xlabel("Time [h]")
        plt.ylabel("Temperature [°C]")
        plt.grid(True)

    for idx in range(6):
        plt.subplot(3, 2, idx + 1)
        plt.legend()

    plt.tight_layout()
    plt.show()
