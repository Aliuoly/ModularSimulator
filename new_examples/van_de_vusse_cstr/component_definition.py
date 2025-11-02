"""Sensors, calculations, and controllers for the Van de Vusse CSTR example."""
from functools import partial

from modular_simulation.usables import (
    PIDController,
    Trajectory,
    SampledDelayedSensor,
)
from modular_simulation.utils.wrappers import hour

from calculation_definition import HeatDutyCalculation

SAMPLE_SENSOR = partial(
    SampledDelayedSensor,
    sampling_period=hour(0.1),
    deadtime=0.0,
    coefficient_of_variance=0.002,
)

sensors = [
    SAMPLE_SENSOR(measurement_tag="Ca", unit="mol/L"),
    SAMPLE_SENSOR(measurement_tag="Cb", unit="mol/L"),
    SAMPLE_SENSOR(measurement_tag="T", unit="deg_C"),
    SAMPLE_SENSOR(measurement_tag="Tk", unit="deg_C"),
    SAMPLE_SENSOR(measurement_tag="Tj_in", unit="deg_C"),
]

calculations = [
    HeatDutyCalculation(
        heat_duty_tag="Qk",
        Tk_tag="Tk",
        T_tag="T",
        kw=4032.0 / 3600.0,
        area=0.215,
    ),
]

controllers = [
    PIDController(
        mv_tag="Tj_in",
        cv_tag="T",
        sp_trajectory=Trajectory(80.0).hold(hour(30)).step(10.0).hold(hour(30)).step(-5.0),
        mv_range=(10.0, 110.0),
        Kp=2.0,
        Ti=hour(0.5),
        cascade_controller=PIDController(
            mv_tag="T",
            cv_tag="Cb",
            sp_trajectory=Trajectory(0.20).hold(hour(40)).step(0.10).hold(hour(40)).step(-0.2),
            mv_range=(50.0, 120.0),
            Kp=20.0,
            Ti=hour(1.0),
        ),
    ),
]
