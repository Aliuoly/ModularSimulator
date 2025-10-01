import logging
from typing import List, TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

from modular_simulation.framework import create_system
from modular_simulation.usables import SampledDelayedSensor
from modular_simulation.plotting import plot_triplet_series
from modular_simulation.control_system import PIDController, Trajectory

from system_definitions import (
    HeatDutyCalculation,
    VanDeVusseAlgebraicStates,
    VanDeVusseConstants,
    VanDeVusseControlElements,
    VanDeVusseStates,
    VanDeVusseSystem,
)

if TYPE_CHECKING:
    from modular_simulation.usables import Calculation


def make_systems():
    # Initial conditions based on the literature steady-state
    initial_states = VanDeVusseStates(
        Ca=2.2291,
        Cb=1.0417,
        T=79.591,
        Tk=77.69,
    )
    initial_controls = VanDeVusseControlElements(Tj_in=77.69)
    initial_algebraic = VanDeVusseAlgebraicStates()

    system_constants = VanDeVusseConstants(
        F=14.19,  # L/h
        Ca0=5.1,  # mol/L
        T0=104.9,  # °C
        k10=1.287e10,  # 1/h
        E1=9758.3,  # K
        dHr1=4.2,  # kJ/mol
        rho=0.9342,  # kg/L
        Cp=3.01,  # kJ/(kg·K)
        kw=4032.0,  # kJ/(h·K·m^2)
        AR=0.215,  # m^2
        VR=10.0,  # L
        mK=5.0,  # kg
        CpK=2.0,  # kJ/(kg·K)
        Fj=10.0,  # kg/h of jacket fluid
    )

    sensors = [
        SampledDelayedSensor(measurement_tag="Ca", sampling_period=0.1),
        SampledDelayedSensor(measurement_tag="Cb", sampling_period=0.1),
        SampledDelayedSensor(measurement_tag="T", sampling_period=0.1),
        SampledDelayedSensor(measurement_tag="Tk", sampling_period=0.1),
        SampledDelayedSensor(measurement_tag="Tj_in", sampling_period=0.1),
    ]

    calculations: List["Calculation"] = [ #type:ignore
        HeatDutyCalculation(
            output_tags=["Qk"],
            measured_input_tags=["Tk", "T"],
            # just gonna use real system constants here.
            # however, you can really make them anything - fit from data, etc. 
            area = system_constants.AR,
            kw = system_constants.kw,
        )
    ]

    controllers = [
        PIDController(
            mv_tag="Tj_in",
            cv_tag="T",
            sp_trajectory=Trajectory(80.0).hold(30.0).step(10).hold(30.0).step(-5),
            mv_range=(10.0, 110.0),
            Kp=2.0,
            Ti=0.5,
            cascade_controller = PIDController(
                mv_tag = "T",
                cv_tag = "Cb",
                sp_trajectory = Trajectory(0.20).hold(40.).step(0.10).hold(40).step(-0.2),
                mv_range = (50,120),
                Kp = 20.0,
                Ti = 1.,
            )
        )
    ]

    # Assemble the systems
    dt = 0.01  # hours (~36 seconds)
    normal_system = create_system(
        dt=dt,
        system_class=VanDeVusseSystem,
        initial_states=initial_states,
        initial_controls=initial_controls,
        initial_algebraic=initial_algebraic,
        sensors=sensors,
        calculations=calculations,
        controllers=controllers,
        system_constants=system_constants,
        use_numba=False
    )

    fast_system = create_system(
        dt=dt,
        system_class=VanDeVusseSystem,
        initial_states=initial_states,
        initial_controls=initial_controls,
        initial_algebraic=initial_algebraic,
        sensors=sensors,
        calculations=calculations,
        controllers=controllers,
        system_constants=system_constants,
        use_numba=True
    )
    return {"normal": normal_system, "fast": fast_system}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mpl.set_loglevel("warning")
    
    pv_kwargs = {'linestyle': '-'}
    sp_kwargs = {'linestyle': '--', "alpha": 0.5}
    systems = make_systems()
    for j, (label, system) in enumerate(systems.items()):
        system.step(nsteps = 12000)

        history = system.measured_history
        sensor_hist = history["sensors"]
        calc_hist = history["calculations"]
        sp_hist = system.setpoint_history
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(3, 2, 1)
        plot_triplet_series(
            ax,
            sensor_hist["T"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plot_triplet_series(
            ax,
            sp_hist["T"],
            style="step",
            line_kwargs=sp_kwargs,
            label=label,
        )
        plt.title("Reactor Temperature")
        plt.xlabel("Time [h]")
        plt.ylabel("Temperature [°C]")
        plt.grid(True)

        ax = plt.subplot(3, 2, 2)
        plot_triplet_series(
            ax,
            calc_hist["Qk"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plt.title("Jacket Heat Duty")
        plt.xlabel("Time [h]")
        plt.ylabel("Qk [kJ/h]")
        plt.grid(True)

        ax = plt.subplot(3, 2, 3)
        plot_triplet_series(
            ax,
            sensor_hist["Ca"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plt.title("Concentration of A")
        plt.xlabel("Time [h]")
        plt.ylabel("[A] [mol/L]")
        plt.grid(True)

        ax = plt.subplot(3, 2, 4)
        plot_triplet_series(
            ax,
            sensor_hist["Cb"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plot_triplet_series(
            ax,
            sp_hist["Cb"],
            style="step",
            line_kwargs=sp_kwargs,
            label=label,
        )
        plt.title("Concentration of B")
        plt.xlabel("Time [h]")
        plt.ylabel("[B] [mol/L]")
        plt.grid(True)

        ax = plt.subplot(3, 2, 5)
        plot_triplet_series(
            ax,
            sensor_hist["Tk"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plt.title("Jacket Temperature")
        plt.xlabel("Time [h]")
        plt.ylabel("Temperature [°C]")
        plt.grid(True)

        ax = plt.subplot(3, 2, 6)
        plot_triplet_series(
            ax,
            sensor_hist["Tj_in"],
            style="step",
            line_kwargs=pv_kwargs,
            label=label,
        )
        plt.title("Jacket Inlet Temperature")
        plt.xlabel("Time [h]")
        plt.ylabel("Temperature [°C]")
        plt.grid(True)
        plt.tight_layout()

    for idx in range(6):
        plt.subplot(3, 2, idx + 1)
        plt.legend()

    
    plt.show()
