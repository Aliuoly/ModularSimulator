from __future__ import annotations
from modular_simulation.usables import (
    CalculationBase,
    ControlElement,
    PIDController,
    SampledDelayedSensor,
    Trajectory,
)

sensors = [
    SampledDelayedSensor(measurement_tag="F_out", unit="L/s"),
    SampledDelayedSensor(
        measurement_tag="F_in",
        unit="L/s",
        coefficient_of_variance=0.05,
    ),
    SampledDelayedSensor(
        measurement_tag="B",
        unit="mol/L",
        coefficient_of_variance=0.05,
        sampling_period=900.0,
        deadtime=900.0,
    ),
    SampledDelayedSensor(measurement_tag="V", unit="L"),
    SampledDelayedSensor(measurement_tag="T", unit="K"),
    SampledDelayedSensor(measurement_tag="T_J", unit="K"),
    SampledDelayedSensor(measurement_tag="T_J_in", unit="K"),
    SampledDelayedSensor(measurement_tag="jacket_flow", unit="L/s"),
]

calculations: list[CalculationBase] = []

control_elements = [
    ControlElement(
        mv_tag="F_in",
        mv_range=(0.0, 1.0e6),
        mv_trajectory=Trajectory(y0=0.0),
        controller=PIDController(
            cv_tag="V",
            cv_range=(0.0, 2000.0),
            sp_trajectory=Trajectory(y0=1.0e3),
            Kp=1.0e-2,
            Ti=100.0,
        ),
    ),
    ControlElement(
        mv_tag="T_J_in",
        mv_range=(200.0, 350.0),
        mv_trajectory=Trajectory(y0=280.0),
        controller=PIDController(
            cv_tag="T_J",
            cv_range=(200.0, 400.0),
            Kp=1.0e-1,
            Ti=50.0,
            cascade_controller=PIDController(
                cv_tag="T",
                cv_range=(200.0, 400.0),
                Kp=1.0e-1,
                Ti=100.0,
                cascade_controller=PIDController(
                    cv_tag="B",
                    cv_range=(0.0, 1.0),
                    sp_trajectory=Trajectory(y0=0.02),
                    Kp=2.0e-1,
                    Ti=5.0,
                ),
            ),
        ),
    ),
]
