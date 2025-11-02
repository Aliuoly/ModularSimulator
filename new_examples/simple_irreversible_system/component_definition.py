
from modular_simulation.usables import (
    SampledDelayedSensor,
    Trajectory,
    CalculationBase,
    PIDController
)
from modular_simulation.utils.wrappers import minute

sensors=[
    SampledDelayedSensor(
        measurement_tag = "F_out",
        unit = "L/s",
    ),
    SampledDelayedSensor(
        measurement_tag = "F_in",
        unit = "L/s",
        coefficient_of_variance=0.05
    ),
    SampledDelayedSensor(
        measurement_tag = "B",
        unit = "mol/L",
        coefficient_of_variance=0.05,
        sampling_period = minute(10),
        deadtime = minute(10),
    ),
    SampledDelayedSensor(
        measurement_tag = "V",
        unit = "L",
        faulty_probability = 0.01,
        faulty_aware = True
    ),
]
calculations: list[CalculationBase] = []

# Define the controllers that will manipulate the control elements.
controllers=[
    PIDController(
        cv_tag="B",
        mv_tag="F_in",
        sp_trajectory=Trajectory(0.5, t0=0.0),
        Kp=1.0e-1,
        Ti=minute(2),
        mv_range=(0, 100),
    )
]