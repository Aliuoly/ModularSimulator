import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np
from enum import Enum
from pydantic import ConfigDict, Field
from typing import ClassVar, Type

from modular_simulation.control_system.controllers.cascade_controller import (
    CascadeController,
    ControllerMode,
)
from modular_simulation.control_system.controllers.controller import Controller
from modular_simulation.control_system.trajectory import Trajectory
from modular_simulation.measurables import ControlElements, States
from modular_simulation.quantities import ControllableQuantities, MeasurableQuantities, UsableQuantities
from modular_simulation.framework.system import System
from modular_simulation.usables import Calculation, Sensor, TimeValueQualityTriplet


class DummyStateMap(Enum):
    X = 0


class DummyStates(States):
    model_config = ConfigDict(extra="forbid")
    StateMap: ClassVar[Type[Enum]] = DummyStateMap
    X: float = Field(0.0)


class DummyControlElements(ControlElements):
    u: float = 0.0


class DummySystem(System):
    @staticmethod
    def calculate_algebraic_values(
        y,
        u,
        k,
        y_map,
        u_map,
        k_map,
        algebraic_map,
    ) -> np.ndarray:  # type: ignore[override]
        del y, u, k, y_map, u_map, k_map, algebraic_map
        return np.zeros(0, dtype=float)

    @staticmethod
    def rhs(
        t,
        y,
        u,
        k,
        algebraic,
        y_map,
        u_map,
        k_map,
        algebraic_map,
    ) -> np.ndarray:  # type: ignore[override]
        del t, u, k, algebraic, y_map, u_map, k_map, algebraic_map
        return np.zeros_like(y)


class DummySensor(Sensor):
    def _should_update(self, t: float) -> bool:  # type: ignore[override]
        return True

    def _get_processed_value(self, raw_value, t):  # type: ignore[override]
        return raw_value


class TimeEchoCalculation(Calculation):
    def _calculation_algorithm(self, t, inputs_dict):  # type: ignore[override]
        return TimeValueQualityTriplet(t=t, value=t, ok=True)


class SimpleController(Controller):
    def _control_algorithm(self, cv_value, sp_value, t):  # type: ignore[override]
        if isinstance(sp_value, TimeValueQualityTriplet):
            target = float(sp_value.value)
        else:
            target = float(sp_value)
        return TimeValueQualityTriplet(t=t, value=target, ok=True)


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
    )

    assert system.history == {}
    assert system.measured_history != {}


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

    
    assert "sensors" in history and "calculations" in history
    assert [x.value for x in history['sensors']["X"]] == [0.0, 1.0, 2.0]
    assert [calc.value for calc in history['calculations']["calc"]] == [0.0, 0.0, 1.0]
    assert [calc.ok for calc in history['calculations']["calc"]] == [True, True, True]


class CascadeStateMap(Enum):
    inner_cv = 0
    bridge_cv = 1
    outer_cv = 2


class CascadeStates(States):
    model_config = ConfigDict(extra="forbid")
    StateMap: ClassVar[Type[Enum]] = CascadeStateMap
    inner_cv: float = Field(0.0)
    bridge_cv: float = Field(0.0)
    outer_cv: float = Field(0.0)


class CascadeControlElements(ControlElements):
    inner_mv: float = 0.0
    bridge_mv: float = 0.0
    outer_mv: float = 0.0


def _make_simple_controller(mv_tag: str, cv_tag: str) -> SimpleController:
    return SimpleController(
        mv_tag=mv_tag,
        cv_tag=cv_tag,
        sp_trajectory=Trajectory(y0=0.0, t0=0.0),
        mv_range=(-10.0, 10.0),
    )


def test_extend_controller_trajectory_targets_active_cascade_loop():
    sensors = [
        DummySensor(measurement_tag="inner_cv"),
        DummySensor(measurement_tag="bridge_cv"),
        DummySensor(measurement_tag="outer_cv"),
    ]

    inner = _make_simple_controller(mv_tag="inner_mv", cv_tag="inner_cv")
    bridge = _make_simple_controller(mv_tag="bridge_mv", cv_tag="bridge_cv")
    outer = _make_simple_controller(mv_tag="outer_mv", cv_tag="outer_cv")

    outer_cascade = CascadeController(inner_loop=bridge, outer_loop=outer)
    cascade = CascadeController(inner_loop=inner, outer_loop=outer_cascade)

    measurables = MeasurableQuantities(
        states=CascadeStates(),
        control_elements=CascadeControlElements(),
    )
    usables = UsableQuantities(sensors=sensors, calculations=[])
    controllables = ControllableQuantities(controllers=[cascade])

    system = DummySystem(
        measurable_quantities=measurables,
        usable_quantities=usables,
        controllable_quantities=controllables,
        system_constants={},
        solver_options={},
    )

    assert cascade.active_sp_trajectory() is outer.sp_trajectory
    system.extend_controller_trajectory(cv_tag=cascade.cv_tag, value=1.5)
    assert outer.sp_trajectory.current_value(system._t) == pytest.approx(1.5)

    cascade.mode = ControllerMode.auto
    system.extend_controller_trajectory(cv_tag=cascade.cv_tag, value=0.5)
    assert inner.sp_trajectory.current_value(system._t) == pytest.approx(0.5)
