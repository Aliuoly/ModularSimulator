from modular_simulation.usables import (
    SampledDelayedSensor,
    Trajectory,
    CalculationBase,
    PIDController,
    ControlElement,
)
from modular_simulation.utils.wrappers import minute

sensors = [
    SampledDelayedSensor(
        measurement_tag="F_out",
        unit="L/s",
    ),
    SampledDelayedSensor(
        measurement_tag="F_in",
        unit="L/minute",
        coefficient_of_variance=0.05,
    ),
    SampledDelayedSensor(
        measurement_tag="B",
        unit="mol/L",
        coefficient_of_variance=0.05,
        sampling_period=minute(10),
        deadtime=minute(10),
    ),
    SampledDelayedSensor(
        measurement_tag="V",
        unit="L",
        faulty_probability=0.01,
        faulty_aware=True,
    ),
]
calculations: list[CalculationBase] = []

# Define the control elements and their controllers
control_elements = [
    ControlElement(
        mv_tag="F_in",
        mv_range=(0, 100 * 60),
        controller=PIDController(
            cv_tag="B",
            sp_trajectory=Trajectory(0.5),
            cv_range=(0, 2),
            Kp=1.0e-1 * 60,
            Ti=minute(2),
        ),
    )
]
