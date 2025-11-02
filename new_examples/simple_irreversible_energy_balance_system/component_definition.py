"""Sensors, calculations, and controllers for the energy balance example."""

from __future__ import annotations

from astropy.units import Unit

from modular_simulation.usables import (
    CalculationBase,
    PIDController,
    SampledDelayedSensor,
    Trajectory,
)


sensors = [
    SampledDelayedSensor(measurement_tag="F_out", unit=Unit("L/s")),
    SampledDelayedSensor(
        measurement_tag="F_in",
        unit=Unit("L/s"),
        coefficient_of_variance=0.05,
    ),
    SampledDelayedSensor(
        measurement_tag="B",
        unit=Unit("mol/L"),
        coefficient_of_variance=0.05,
        sampling_period=900.0,
        deadtime=900.0,
    ),
    SampledDelayedSensor(measurement_tag="V", unit=Unit("L")),
    SampledDelayedSensor(measurement_tag="T", unit=Unit("K")),
    SampledDelayedSensor(measurement_tag="T_J", unit=Unit("K")),
    SampledDelayedSensor(measurement_tag="T_J_in", unit=Unit("K")),
    SampledDelayedSensor(measurement_tag="jacket_flow", unit=Unit("L/s")),
]

calculations: list[CalculationBase] = []

controllers = [
    PIDController(
        mv_tag="F_in",
        cv_tag="V",
        sp_trajectory=Trajectory(1.0e3, unit=Unit("L")),
        Kp=1.0e-2,
        Ti=100.0,
        mv_range=(0.0 * Unit("L/s"), 1.0e6 * Unit("L/s")),
    ),
    PIDController(
        mv_tag="T_J_in",
        cv_tag="T_J",
        sp_trajectory=Trajectory(300.0, unit=Unit("K")),
        Kp=1.0e-1,
        Ti=50.0,
        mv_range=(200.0 * Unit("K"), 350.0 * Unit("K")),
        cascade_controller=PIDController(
            mv_tag="T_J",
            cv_tag="T",
            sp_trajectory=Trajectory(300.0, Unit("K")),
            Kp=1.0e-1,
            Ti=100.0,
            mv_range=(200.0 * Unit("K"), 350.0 * Unit("K")),
            cascade_controller=PIDController(
                mv_tag="T",
                cv_tag="B",
                sp_trajectory=(
                    Trajectory(0.02, unit=Unit("mol/L"))
                    .hold(duration=15_000.0)
                    .hold(duration=15_000.0, value=0.05)
                    .hold(duration=15_000.0, value=0.1)
                    .hold(duration=15_000.0, value=0.01)
                ),
                Kp=2.0e-1,
                Ti=5.0,
                mv_range=(250.0 * Unit("K"), 350.0 * Unit("K")),
            ),
        ),
    ),
]
