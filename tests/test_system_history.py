import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np
from enum import Enum
from pydantic import ConfigDict, Field

from modular_simulation.measurables import ControlElements, States
from modular_simulation.quantities import ControllableQuantities, MeasurableQuantities, UsableQuantities
from modular_simulation.system import System


class DummyStateMap(Enum):
    X = 0


class DummyStates(States):
    model_config = ConfigDict(extra="forbid")
    StateMap = DummyStateMap
    X: float = Field(0.0)


class DummyControlElements(ControlElements):
    u: float = 0.0


class DummySystem(System):
    @staticmethod
    def _calculate_algebraic_values(y, StateMap, control_elements, system_constants):  # type: ignore[override]
        return {}

    @staticmethod
    def rhs(t, y, StateMap, algebraic_values_dict, control_elements, system_constants):  # type: ignore[override]
        return np.zeros_like(y)


def test_system_history_flags_disable_recording():
    measurables = MeasurableQuantities(
        states=DummyStates(X=1.0),
        control_elements=DummyControlElements(u=0.0),
    )
    usables = UsableQuantities(sensors=[], calculations=[])
    controllables = ControllableQuantities(controllers=[])

    system = DummySystem(
        measurable_quantities=measurables,
        usable_quantities=usables,
        controllable_quantities=controllables,
        system_constants={},
        solver_options={},
        record_history=False,
        record_measured_history=False,
    )

    assert system.history == {}
    assert system.measured_history == {}
