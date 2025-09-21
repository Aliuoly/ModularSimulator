import pytest

pytest.importorskip("numpy")

from modular_simulation.usables.sensors.sensor import Sensor


class InstantSensor(Sensor):
    """Sensor that always updates and returns the provided measurement."""

    def _should_update(self, t: float) -> bool:  # type: ignore[override]
        return True

    def _get_processed_value(self, raw_value, t):  # type: ignore[override]
        return raw_value


def test_sensor_measurement_history_tracks_samples():
    sensor = InstantSensor(measurement_tag="X")

    state = {"value": 0.0}
    sensor._measurement_function = lambda: state["value"]
    sensor._initialized = True

    samples = []
    for i in range(5):
        state["value"] = float(i)
        measurement = sensor.measure(float(i))
        samples.append(measurement.value)

    history = sensor.measurement_history()
    assert [h.t for h in history] == [float(i) for i in range(5)]
    assert [h.value for h in history] == samples
    assert [h.ok for h in history] == [True] * 5
