from functools import partial
from modular_simulation.usables import (
    SampledDelayedSensor,
    FirstOrderFilter,
    ControlElement,
    PIDController,
    Trajectory,
    ControllerMode,
)
from modular_simulation.utils.wrappers import minute
from calculation_definition import MoleRatioCalculation, Monomer1PartialPressure

analyzer_partial = partial(
    SampledDelayedSensor,
    deadtime=minute(2),
    sampling_period=minute(2),
) # 2 min delay and sampling period

sensors = [
    SampledDelayedSensor(measurement_tag = "F_m1", unit = "kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "F_m2", unit = "kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "F_cat", unit = "kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "pressure", unit = "kPa", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "yM1", unit = "", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "yM2", unit = "", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "effective_cat", unit = "kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "mass_prod_rate", unit = "tonne/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "monomer_rates", unit = "kmol/s", coefficient_of_variance=0.002),
]
calculations = [
    FirstOrderFilter(
        name = 'mass_prod_rate_filter',
        raw_signal_tag = 'mass_prod_rate',
        filtered_signal_tag = 'filtered_mass_prod_rate',
        time_constant = minute(10), 
    ),
    FirstOrderFilter(
        name = 'pressure_filter',
        raw_signal_tag = 'pressure',
        filtered_signal_tag = 'filtered_pressure',
        time_constant = minute(10), 
    ),
    MoleRatioCalculation(
        rM2_tag = "rM2",
        yM1_tag = "yM1",
        yM2_tag = "yM2",
    ),
    Monomer1PartialPressure(
        pM1_tag = "pM1", 
        yM1_tag = "yM1",
        pressure_tag = "filtered_pressure"
    ),
]
control_elements = [
    ControlElement(
        mv_tag="F_m1",
        mv_range=(0, 80000.),
        controller=PIDController(
            cv_tag="pM1",
            cv_range = (0, 1000),
            sp_trajectory=Trajectory(y0=700.0),
            Kp=1000 / 2000 * 28,
            Ti=minute(12),
            setpoint_weight=0.0,
            velocity_form=True,
        ),
    ),
    ControlElement(
        mv_tag="F_m2",
        mv_range=(0, 8000.),
        controller=PIDController(
            cv_tag="rM2",
            cv_range = (0, 1),
            sp_trajectory=Trajectory(y0=0.3),
            Kp=300 * 56,
            Ti=minute(30),
            setpoint_weight=0.0,
            velocity_form=True,
        ),
    ),
    ControlElement(
        mv_tag="F_cat",
        mv_range=(0, 20.),
        controller=PIDController(
            cv_tag="filtered_mass_prod_rate",
            cv_range = (0, 100),
            sp_trajectory=Trajectory(y0=50.0),
            Kp=1,
            Ti=minute(60),
            mode=ControllerMode.TRACKING,
        ),
    ),
]