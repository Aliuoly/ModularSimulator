import math
import pytest
from astropy.units import Unit

from modular_simulation.usables.calculations.first_order_filter import FirstOrderFilter
from modular_simulation.usables.controllers.controller import Controller
from modular_simulation.usables.controllers.trajectory import Trajectory
from modular_simulation.usables.sensors.sampled_delayed_sensor import SampledDelayedSensor
from modular_simulation.usables.usable_quantities import UsableQuantities
from modular_simulation.validation.exceptions import (
    CalculationConfigurationError,
    ControllerConfigurationError,
    SensorConfigurationError,
)


class ProportionalController(Controller):
    gain: float = 1.0

    def _control_algorithm(self, t, cv, sp):  # type: ignore[override]
        return self._u0 + self.gain * (sp - cv)


@pytest.fixture()
def base_sensor():
    return SampledDelayedSensor(
        measurement_tag="temperature",
        alias_tag="temp_meas",
        sampling_period=0.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
    )


@pytest.fixture()
def base_filter():
    return FirstOrderFilter(
        filtered_signal_tag="temp_filtered",
        raw_signal_tag="temp_meas",
        time_constant=1.0,
    )


@pytest.fixture()
def base_controller(heater_mv_range):
    return ProportionalController(
        mv_tag="heater_power",
        cv_tag="temp_filtered",
        sp_trajectory=Trajectory(y0=300.0, unit=Unit("K")),
        mv_range=heater_mv_range,
        ramp_rate=50.0,
    )


def test_duplicate_tags_raise_exception(thermal_measurables):
    sensors = [
        SampledDelayedSensor(measurement_tag="temperature", alias_tag="duplicate"),
        SampledDelayedSensor(measurement_tag="temperature", alias_tag="duplicate"),
    ]
    with pytest.raises(ExceptionGroup) as excinfo:
        UsableQuantities(
            sensors=sensors,
            calculations=[],
            controllers=[],
            measurable_quantities=thermal_measurables,
        )
    assert any(isinstance(err, SensorConfigurationError) for err in excinfo.value.exceptions)


def test_missing_sensor_source_raises(thermal_measurables):
    sensors = [SampledDelayedSensor(measurement_tag="unknown_tag", unit=Unit("K"))]
    with pytest.raises(ExceptionGroup) as excinfo:
        UsableQuantities(
            sensors=sensors,
            calculations=[],
            controllers=[],
            measurable_quantities=thermal_measurables,
        )
    assert any(isinstance(err, SensorConfigurationError) for err in excinfo.value.exceptions)


def test_missing_calculation_input_raises(thermal_measurables, base_sensor):
    calc = FirstOrderFilter(
        filtered_signal_tag="temp_filtered",
        raw_signal_tag="missing_sensor",
        time_constant=1.0,
    )
    with pytest.raises(ExceptionGroup) as excinfo:
        UsableQuantities(
            sensors=[base_sensor],
            calculations=[calc],
            controllers=[],
            measurable_quantities=thermal_measurables,
        )
    assert any(isinstance(err, CalculationConfigurationError) for err in excinfo.value.exceptions)


def test_missing_controller_references_raise(thermal_measurables, base_sensor, base_filter, heater_mv_range):
    controller = ProportionalController(
        mv_tag="nonexistent",
        cv_tag="temp_filtered",
        sp_trajectory=Trajectory(y0=300.0, unit=Unit("K")),
        mv_range=heater_mv_range,
    )
    with pytest.raises(ExceptionGroup) as excinfo:
        UsableQuantities(
            sensors=[base_sensor],
            calculations=[base_filter],
            controllers=[controller],
            measurable_quantities=thermal_measurables,
        )
    assert any(isinstance(err, ControllerConfigurationError) for err in excinfo.value.exceptions)


def test_update_executes_all_components(thermal_measurables, base_sensor, base_filter, base_controller):
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

    thermal_measurables.states.temperature = 305.0
    usable.update(1.0)
    usable.update(2.0)

    sensor_history = base_sensor.measurement_history
    assert len(sensor_history) >= 3
    assert sensor_history[-1].value == pytest.approx(305.0)

    filtered_info = next(info for info in usable.tag_infos if info.tag == "temp_filtered")
    assert filtered_info.history[-1].time == pytest.approx(2.0)

    control_value = thermal_measurables.control_elements.heater_power
    lower, upper = (bound.value for bound in base_controller.mv_range)
    assert lower <= control_value <= upper
    assert not math.isnan(control_value)
