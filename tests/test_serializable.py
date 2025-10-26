import math
import pytest
from astropy.units import Unit

from modular_simulation.usables.calculations.first_order_filter import FirstOrderFilter
from modular_simulation.usables.controllers.controller_base import ControllerBase
from modular_simulation.usables.controllers.trajectory import Trajectory
from modular_simulation.usables.sensors.sampled_delayed_sensor import SampledDelayedSensor
from modular_simulation.usables.usable_quantities import UsableQuantities
from modular_simulation.validation.exceptions import (
    CalculationConfigurationError,
    ControllerConfigurationError,
    SensorConfigurationError,
)    
from modular_simulation.measurables import (
    MeasurableQuantities,
    States,
    ControlElements,
    AlgebraicStates,
    Constants,
)
import pytest
from astropy.units import Unit, Quantity
from typing import Annotated

from modular_simulation.measurables import (
    MeasurableQuantities,
    States,
    ControlElements,
    AlgebraicStates,
    Constants,
)


UNIT_TEMPERATURE = Unit("K")
UNIT_POWER = Unit("W")
UNIT_TIME = Unit("s")
UNIT_DIMENSIONLESS = Unit(1)


class ThermalStates(States):
    temperature: Annotated[float, UNIT_TEMPERATURE] = 300.0


class ThermalAlgebraic(AlgebraicStates):
    heat_flux: Annotated[float, UNIT_POWER] = 0.0


class ThermalControlElements(ControlElements):
    heater_power: Annotated[float, UNIT_POWER] = 0.0


class ThermalConstants(Constants):
    ambient_temperature: Annotated[float, UNIT_TEMPERATURE] = 295.0
    time_constant: Annotated[float, UNIT_TIME] = 5.0
sensor = SampledDelayedSensor(
    measurement_tag="temperature",
    alias_tag="temp_meas",
    sampling_period=1.0,
    deadtime=0.0,
    coefficient_of_variance=0.0,
    random_seed=1,
)
mv_sensor = SampledDelayedSensor(
    measurement_tag="heater_power",
    sampling_period=1.0,
    deadtime=0.0,
    coefficient_of_variance=0.0,
    random_seed=1,
)
filter = FirstOrderFilter(
    filtered_signal_tag="temp_filtered",
    raw_signal_tag="temp_meas",
    time_constant=1.0,
)
class ProportionalController(ControllerBase):
    gain: float = 1.0

    def _control_algorithm(self, t, cv, sp):  # type: ignore[override]
        return self._u0 + self.gain * (sp - cv)
controller = ProportionalController(
    mv_tag="heater_power",
    cv_tag="temp_filtered",
    sp_trajectory=Trajectory(y0=300.0, unit=Unit("K")).step(5).hold(10, 5),
    mv_range=(0, 100),
    ramp_rate=50.0,
)
measurable = MeasurableQuantities(
    states=ThermalStates(),
    algebraic_states=ThermalAlgebraic(),
    control_elements=ThermalControlElements(),
    constants=ThermalConstants(),
)
def test_quantity_serializable(thermal_measurables, base_sensor, base_filter, base_controller):
    mv_sensor = SampledDelayedSensor(
        measurement_tag="heater_power",
        alias_tag="heater_power",
        sampling_period=0.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
    )
    usable = UsableQuantities(
        sensors=[base_sensor, mv_sensor],
        calculations=[base_filter],
        controllers=[base_controller],
        measurable_quantities=thermal_measurables,
    )
    usable._initialize()

usable = UsableQuantities(
    sensors=[sensor, mv_sensor],
    calculations=[filter],
    controllers=[controller],
    measurable_quantities=measurable,
)
usable._initialize()

print(controller)
print(controller.model_dump_json())
print(ProportionalController(**controller.model_dump()))