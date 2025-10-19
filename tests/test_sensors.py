import pytest

from modular_simulation.usables.sensors.sampled_delayed_sensor import SampledDelayedSensor
from modular_simulation.usables.usable_quantities import UsableQuantities


@pytest.fixture()
def sampled_sensor(thermal_measurables):
    sensor = SampledDelayedSensor(
        measurement_tag="temperature",
        alias_tag="temp_meas",
        sampling_period=1.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
        random_seed=1,
    )
    usable = UsableQuantities(
        sensors=[sensor],
        calculations=[],
        controllers=[],
        measurable_quantities=thermal_measurables,
    )
    usable._initialize()
    return sensor, thermal_measurables


def test_sensor_initial_measurement_records_history(sampled_sensor):
    sensor, measurables = sampled_sensor
    history = sensor.measurement_history
    assert len(history) == 1
    recorded = history[0]
    assert recorded.time == pytest.approx(0.0)
    assert recorded.value == pytest.approx(measurables.states.temperature)
    assert recorded.ok is True


def test_sensor_sampling_period_prevents_fast_updates(sampled_sensor):
    sensor, measurables = sampled_sensor

    measurables.states.temperature = 320.0
    initial_history_len = len(sensor.measurement_history)
    no_update = sensor.measure(0.5)
    assert no_update.value == pytest.approx(sensor.measurement_history[-1].value)
    assert len(sensor.measurement_history) == initial_history_len

    measurables.states.temperature = 325.0
    sensor.measure(1.0)
    sensor.measure(2.0)
    assert len(sensor.measurement_history) == initial_history_len + 2
    assert sensor.measurement_history[-1].value == pytest.approx(325.0)


def test_sensor_deadtime_returns_delayed_samples(thermal_measurables):
    sensor = SampledDelayedSensor(
        measurement_tag="temperature",
        alias_tag="temp_meas",
        sampling_period=0.5,
        deadtime=1.0,
        coefficient_of_variance=0.0,
        random_seed=0,
    )
    usable = UsableQuantities(
        sensors=[sensor],
        calculations=[],
        controllers=[],
        measurable_quantities=thermal_measurables,
    )
    usable._initialize()

    true_history = []
    samples = []
    for step in range(6):
        t = 0.5 * step
        value = 300.0 + 5.0 * step
        thermal_measurables.states.temperature = value
        true_history.append((t, value))
        sensor.measure(t)
        samples.append((t, sensor.measurement_history[-1].value))

    def true_value_at(time: float) -> float:
        candidates = [val for ts, val in true_history if ts <= time + 1e-12]
        return candidates[-1] if candidates else true_history[0][1]

    for t, measured in samples:
        expected_time = max(0.0, t - 1.0)
        expected_value = true_value_at(expected_time)
        assert measured == pytest.approx(expected_value, rel=1e-6, abs=1e-6)
