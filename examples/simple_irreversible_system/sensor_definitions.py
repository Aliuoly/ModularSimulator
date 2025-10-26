
from system_definitions import (
    IrreversibleStates,
    IrreversibleControlElements,
    IrreversibleAlgebraicStates,
    IrreversibleConstants,
    IrreversibleSystem,
)

from modular_simulation.usables import SampledDelayedSensor
from modular_simulation.plotting import plot_triplet_series
from modular_simulation.framework import create_system
from modular_simulation.usables import Trajectory, PIDController
, TYPE_CHECKING
import logging
from astropy.units import Unit

sensors=[
    SampledDelayedSensor(
        measurement_tag = "F_out",
        unit = Unit("L/s"),
    ),
    SampledDelayedSensor(
        measurement_tag = "F_in",
        unit = Unit("L/s"),
        coefficient_of_variance=0.05
    ),
    SampledDelayedSensor(
        measurement_tag = "B",
        unit = Unit("mol/L"),
        coefficient_of_variance=0.05,
        sampling_period = 900,
        deadtime = 900,
    ),
    SampledDelayedSensor(
        measurement_tag = "V",
        unit = Unit("L"),
        faulty_probability = 0.01,
        faulty_aware = True
    ),
]