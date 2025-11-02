from functools import partial
from calculation_definition import (
    MoleRatioCalculation, 
    Monomer1PartialPressure
)
from astropy.units import Unit
from modular_simulation.usables.calculations.first_order_filter import FirstOrderFilter
from modular_simulation.usables import (
    PIDController,
    Trajectory,
    ControllerMode,
    MVController,
    SampledDelayedSensor
)
from modular_simulation.utils.wrappers import minute

analyzer_partial = partial(
    SampledDelayedSensor, 
    deadtime = minute(2), 
    sampling_period = minute(2),
    coefficient_of_variance = 0.002, # 1% rel std
    ) # 2 min delay and sampling period
sensors = [
    analyzer_partial(measurement_tag="yM1", unit='', random_seed=0),
    analyzer_partial(measurement_tag="yM2", unit='', random_seed=1),
    SampledDelayedSensor(measurement_tag = "pressure", unit="kPa", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "mass_prod_rate", unit="kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "F_m1", unit = "kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "F_m2", unit = "kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "F_cat", unit = "kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "effective_cat", unit = "kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "monomer_rates", unit = "kmol/hr", coefficient_of_variance=0.002),
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
        signal_unit = Unit("kPa"),
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
controllers = [
    PIDController(
        mv_tag = "F_m1",
        cv_tag = "pM1",
        sp_trajectory = Trajectory(y0 = 700.0),
        Kp = 1000/2000*28,
        Ti = minute(12), 
        mv_range = (0, 80000.),
        setpoint_weight = 0.0,
        velocity_form=True
    ),
    PIDController(
        mv_tag = "F_m2",
        cv_tag = "rM2",
        sp_trajectory = Trajectory(y0 = 0.3),
        Kp = 300 * 56,
        Ti = minute(30),
        mv_range = (0, 8000.),
        setpoint_weight = 0.0,
        velocity_form=True
    ),
    MVController(
        mv_tag = "F_cat",
        cv_tag = "F_cat",
        sp_trajectory = Trajectory(y0=6.0),
        mv_range = (0, 20.),
        mode = ControllerMode.AUTO,
        cascade_controller = PIDController(
            mv_tag = "F_cat",
            cv_tag = "filtered_mass_prod_rate",
            sp_trajectory = Trajectory(y0=50.0),
            Kp = 1,
            Ti = minute(60), 
            mv_range = (0, 20.),
            mode = ControllerMode.TRACKING,
        )
    )
]