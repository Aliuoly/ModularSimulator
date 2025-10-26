import math

import pytest
from astropy.units import Quantity, Unit

from modular_simulation.core import create_system
from modular_simulation.interfaces import (
    ControllerBase,
    ControllerMode,
    SampledDelayedSensor,
    Trajectory,
)
from modular_simulation.interfaces.calculations.first_order_filter import FirstOrderFilter
from modular_simulation.validation.exceptions import (
    CalculationConfigurationError,
    ControllerConfigurationError,
    SensorConfigurationError,
)


class ProportionalController(ControllerBase):
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
        unit=Unit("K"),
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


def test_duplicate_tags_raise_exception(thermal_model):
    sensors = [
        SampledDelayedSensor(measurement_tag="temperature", alias_tag="duplicate", unit=Unit("K")),
        SampledDelayedSensor(measurement_tag="temperature", alias_tag="duplicate", unit=Unit("K")),
    ]
    with pytest.raises(ExceptionGroup) as excinfo:
        create_system(
            dt=Quantity(1.0, Unit("s")),
            dynamic_model=thermal_model,
            sensors=sensors,
        )
    assert any(isinstance(err, SensorConfigurationError) for err in excinfo.value.exceptions)


def test_missing_sensor_source_raises(thermal_model):
    with pytest.raises(ExceptionGroup) as excinfo:
        create_system(
            dt=Quantity(1.0, Unit("s")),
            dynamic_model=thermal_model,
            sensors=[SampledDelayedSensor(measurement_tag="unknown_tag", unit=Unit("K"))],
        )
    assert any(isinstance(err, SensorConfigurationError) for err in excinfo.value.exceptions)


def test_missing_calculation_input_raises(thermal_model, base_sensor):
    calc = FirstOrderFilter(
        filtered_signal_tag="temp_filtered",
        raw_signal_tag="missing_sensor",
        time_constant=1.0,
    )
    with pytest.raises(ExceptionGroup) as excinfo:
        create_system(
            dt=Quantity(1.0, Unit("s")),
            dynamic_model=thermal_model,
            sensors=[base_sensor],
            calculations=[calc],
        )
    assert any(isinstance(err, CalculationConfigurationError) for err in excinfo.value.exceptions)


def test_missing_controller_references_raise(thermal_model, base_sensor, base_filter, heater_mv_range):
    controller = ProportionalController(
        mv_tag="nonexistent",
        cv_tag="temp_filtered",
        sp_trajectory=Trajectory(y0=300.0, unit=Unit("K")),
        mv_range=heater_mv_range,
    )
    with pytest.raises(ExceptionGroup) as excinfo:
        create_system(
            dt=Quantity(1.0, Unit("s")),
            dynamic_model=thermal_model,
            sensors=[base_sensor],
            calculations=[base_filter],
            controllers=[controller],
        )
    assert any(isinstance(err, ControllerConfigurationError) for err in excinfo.value.exceptions)


def test_update_executes_all_components(thermal_model, base_sensor, base_filter, base_controller):
    mv_sensor = SampledDelayedSensor(
        measurement_tag="heater_power",
        alias_tag="heater_power",
        sampling_period=0.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
        unit=Unit("W"),
    )
    system = create_system(
        dt=Quantity(1.0, Unit("s")),
        dynamic_model=thermal_model,
        sensors=[base_sensor, mv_sensor],
        calculations=[base_filter],
        controllers=[base_controller],
        use_numba=False,
        record_history=False,
    )

    system.dynamic_model.temperature = 305.0
    system.update(1.0)
    system.dynamic_model.temperature = 305.0
    system.update(2.0)

    sensor_history = base_sensor._tag_info.history
    assert len(sensor_history) >= 3
    assert sensor_history[-1].value == pytest.approx(305.0)

    filtered_info = system.tag_infos[[info.tag for info in system.tag_infos].index("temp_filtered")]
    assert filtered_info.history[-1].time == pytest.approx(2.0)

    control_value = system.dynamic_model.heater_power
    lower, upper = base_controller.mv_range
    assert lower <= control_value <= upper
    assert not math.isnan(control_value)
