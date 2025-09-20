import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np
from enum import Enum
from pydantic import ConfigDict, Field

from modular_simulation.measurables import ControlElements, States
from modular_simulation.quantities import ControllableQuantities, MeasurableQuantities, UsableQuantities
from modular_simulation.system import System
from modular_simulation.usables import Calculation, Sensor, TimeValueQualityTriplet


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


class DummySensor(Sensor):
    def _should_update(self, t: float) -> bool:  # type: ignore[override]
        return True

    def _get_processed_value(self, raw_value, t):  # type: ignore[override]
        return raw_value


class TimeEchoCalculation(Calculation):
    def _calculation_algorithm(self, t, inputs_dict):  # type: ignore[override]
        return TimeValueQualityTriplet(t=t, value=t, ok=True)


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


def test_system_measured_history_includes_calculations():
    measurables = MeasurableQuantities(
        states=DummyStates(X=0.0),
        control_elements=DummyControlElements(u=0.0),
    )
    sensor = DummySensor(measurement_tag="X")
    calculation = TimeEchoCalculation(
        output_tag="calc",
        measured_input_tags=[],
        calculated_input_tags=[],
        constants={},
    )
    usables = UsableQuantities(sensors=[sensor], calculations=[calculation])
    controllables = ControllableQuantities(controllers=[])

    system = DummySystem(
        measurable_quantities=measurables,
        usable_quantities=usables,
        controllable_quantities=controllables,
        system_constants={},
        solver_options={},
    )

    system.measurable_quantities.states.X = 1.0
    system.step(1.0)
    system.measurable_quantities.states.X = 2.0
    system.step(1.0)

    history = system.measured_history

    assert history["X"].tolist() == [0.0, 1.0, 2.0]
    assert history["calc"].tolist() == [0.0, 0.0, 1.0]
    assert history["calc_ok"].tolist() == [True, True, True]
    assert "_sensors" in history and "_calculations" in history
    assert np.array_equal(history["X"], history["_sensors"]["X"]["value"])
    assert np.array_equal(history["calc"], history["_calculations"]["calc"]["value"])
    assert np.array_equal(history["time"], history["_sensors"]["X"]["time"])
