import numpy as np
import pytest
from typing import Annotated, override
from pydantic import Field

from modular_simulation.framework.system import System
from modular_simulation.measurables.process_model import (
    ProcessModel,
    StateMetadata,
    StateType,
)
from modular_simulation.components.sensors import AbstractSensor as SensorBase
from modular_simulation.components.abstract_component import ComponentUpdateResult


class MinimalProcessModel(ProcessModel):
    x: Annotated[float, StateMetadata(StateType.DIFFERENTIAL, "1", "simple state")] = Field(1.0)

    @staticmethod
    def calculate_algebraic_values(
        y: np.ndarray,
        u: np.ndarray,
        k: np.ndarray,
        y_map: dict[str, slice | int],
        u_map: dict[str, slice | int],
        k_map: dict[str, slice | int],
        algebraic_map: dict[str, slice | int],
        algebraic_size: int,
    ) -> np.ndarray:
        return np.zeros(algebraic_size, dtype=float)

    @staticmethod
    def differential_rhs(
        t: float,
        y: np.ndarray,
        u: np.ndarray,
        k: np.ndarray,
        algebraic: np.ndarray,
        y_map: dict[str, slice | int],
        u_map: dict[str, slice | int],
        k_map: dict[str, slice | int],
        algebraic_map: dict[str, slice | int],
    ) -> np.ndarray:
        return -y


class MinimalSensor(SensorBase):
    @override
    def _update(self, t: float) -> ComponentUpdateResult:
        measurement = self._measurement_getter()
        self._point.data = measurement
        return ComponentUpdateResult(data_value=measurement, exceptions=[])


@pytest.fixture()
def minimal_system() -> System:
    model = MinimalProcessModel(x=2.0)
    sensor = MinimalSensor(measurement_tag="x", alias_tag="x_sensor", unit="1")
    return System(
        dt=0.1,
        process_model=model,
        sensors=[sensor],
        calculations=[],
        control_elements=[],
        record_history=False,
        show_progress=False,
    )


def test_system_save_load_round_trip(minimal_system: System) -> None:
    minimal_system.step(duration=0.3)
    payload = minimal_system.save()

    restored = System.load(payload)

    assert isinstance(restored.process_model, MinimalProcessModel)
    assert restored.dt == pytest.approx(minimal_system.dt)
    assert restored.process_model.x == pytest.approx(minimal_system.process_model.x)
    assert restored.time == pytest.approx(minimal_system.time)
    assert isinstance(restored.sensors[0], MinimalSensor)
    assert restored.tag_store[restored.sensors[0].alias_tag].tag == "x_sensor"

    starting_value = restored.process_model.x
    restored.step(duration=restored.dt)
    expected = starting_value * np.exp(-restored.dt)
    assert restored.process_model.x == pytest.approx(expected, rel=1e-3)

    update_result = restored.sensors[0].update(restored.time)
    assert update_result.data_value.ok
