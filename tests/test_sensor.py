import pytest

from types import SimpleNamespace

pytest.importorskip("numpy")

from modular_simulation.quantities import UsableQuantities
from modular_simulation.usables.sensors.sensor import Sensor


class InstantSensor(Sensor):
    """Sensor that always updates and returns the provided measurement."""

    def _should_update(self, t: float) -> bool:  # type: ignore[override]
        return True

    def _get_processed_value(self, raw_value, t):  # type: ignore[override]
        return raw_value


def test_sensor_measurement_history_tracks_samples():
    sensor = InstantSensor(measurement_tag="X")

    measurement_owner = SimpleNamespace(X=0.0)
    sensor._measurement_owner = measurement_owner
    sensor._initialized = True

    usable = UsableQuantities(sensors=[sensor], calculations=[])

    samples = []
    for i in range(5):
        measurement_owner.X = float(i)
        measurement = usable.update(float(i))["X"]
        samples.append(measurement.value)

    history = usable.history["sensors"]["X"]
    assert [h.t for h in history] == [float(i) for i in range(5)]
    assert [h.value for h in history] == samples
    assert [h.ok for h in history] == [True] * 5
