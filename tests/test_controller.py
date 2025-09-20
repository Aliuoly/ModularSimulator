import pytest

pytest.importorskip("numpy")

from modular_simulation.control_system.controllers.cascade_controller import (
    CascadeController,
    ControllerMode,
)
from modular_simulation.control_system.controllers.controller import Controller
from modular_simulation.control_system.trajectory import Trajectory
from modular_simulation.usables import TimeValueQualityTriplet


class DummyController(Controller):
    """Minimal concrete controller for exercising base-class behaviour."""

    def model_post_init(self, __context):  # type: ignore[override]
        super().model_post_init(__context)
        self._observed_setpoints: list[float] = []

    def _control_algorithm(self, cv_value, sp_value, t):  # type: ignore[override]
        if isinstance(sp_value, TimeValueQualityTriplet):
            sp = float(sp_value.value)
        else:
            sp = float(sp_value)
        self._observed_setpoints.append(sp)
        return TimeValueQualityTriplet(t, 0.0, ok=True)


class EchoController(Controller):
    """Controller that echoes its setpoint as the control output."""

    def model_post_init(self, __context):  # type: ignore[override]
        super().model_post_init(__context)
        self._observed_setpoints: list[float] = []

    def _control_algorithm(self, cv_value, sp_value, t):  # type: ignore[override]
        if isinstance(sp_value, TimeValueQualityTriplet):
            sp = float(sp_value.value)
        else:
            sp = float(sp_value)
        self._observed_setpoints.append(sp)
        return TimeValueQualityTriplet(t, sp, ok=True)


def _prepare_controller(ramp_rate: float | None) -> tuple[DummyController, dict[str, float]]:
    traj = Trajectory(y0=0.0, t0=0.0)
    controller = DummyController(
        mv_tag="mv",
        cv_tag="cv",
        sp_trajectory=traj,
        mv_range=(-10.0, 10.0),
        ramp_rate=ramp_rate,
    )
    controller._mv_setter = lambda value: None
    controller._last_value = TimeValueQualityTriplet(0.0, 0.0, ok=True)
    state = {"t": 0.0, "value": 0.0}

    def get_pv() -> TimeValueQualityTriplet:
        return TimeValueQualityTriplet(state["t"], state["value"], ok=True)

    controller._cv_getter = get_pv
    return controller, state


def _prepare_echo_controller() -> tuple[EchoController, dict[str, float], list[float]]:
    traj = Trajectory(y0=0.0, t0=0.0)
    controller = EchoController(
        mv_tag="mv_echo",
        cv_tag="cv_echo",
        sp_trajectory=traj,
        mv_range=(-10.0, 10.0),
    )
    controller._last_value = TimeValueQualityTriplet(0.0, 0.0, ok=True)
    state = {"t": 0.0, "value": 0.0}

    def get_pv() -> TimeValueQualityTriplet:
        return TimeValueQualityTriplet(state["t"], state["value"], ok=True)

    controller._cv_getter = get_pv
    applied: list[float] = []
    controller._mv_setter = lambda value: applied.append(value)
    return controller, state, applied


def test_controller_ramp_rate_limits_setpoint_progression():
    controller, state = _prepare_controller(ramp_rate=1.0)

    state.update({"t": 0.0, "value": 0.0})
    controller.update(0.0)
    assert controller._observed_setpoints[-1] == pytest.approx(0.0)

    controller.sp_trajectory.set_now(1.0, 5.0)
    for t, expected in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]:
        state.update({"t": t, "value": 0.0})
        controller.update(t)
        assert controller._observed_setpoints[-1] == pytest.approx(expected)

    state.update({"t": 2.0, "value": 10.0})
    controller.track_cv(2.0)
    state.update({"t": 3.0, "value": 0.0})
    controller.update(3.0)
    assert controller._observed_setpoints[-1] == pytest.approx(10.0)


def test_controller_without_ramp_tracks_raw_setpoint():
    controller, state = _prepare_controller(ramp_rate=None)

    state.update({"t": 0.0, "value": 0.0})
    controller.update(0.0)

    controller.sp_trajectory.set_now(1.0, 7.5)
    state.update({"t": 1.0, "value": 0.0})
    controller.update(1.0)
    assert controller._observed_setpoints[-1] == pytest.approx(7.5)


def test_cascade_requires_non_cascade_inner_loop():
    inner, _ = _prepare_controller(ramp_rate=None)
    outer, _, _ = _prepare_echo_controller()

    inner_cascade = CascadeController(
        inner_loop=inner,
        outer_loop=outer,
    )

    with pytest.raises(TypeError):
        CascadeController(
            inner_loop=inner_cascade,  # type: ignore[arg-type]
            outer_loop=outer,
        )


def test_cascade_updates_inner_setpoint_from_outer_output():
    inner, inner_state = _prepare_controller(ramp_rate=None)
    inner_actions: list[float] = []
    inner._mv_setter = lambda value: inner_actions.append(value)
    inner._last_value = TimeValueQualityTriplet(0.0, 0.0, ok=True)

    outer, outer_state, outer_actions = _prepare_echo_controller()

    cascade = CascadeController(
        inner_loop=inner,
        outer_loop=outer,
    )
    cascade.mode = ControllerMode.cascade

    inner_state.update({"t": 1.0, "value": 0.0})
    outer_state.update({"t": 1.0, "value": 0.0})
    outer.sp_trajectory.set_now(0.0, 3.0)

    result = cascade.update(1.0)

    assert outer_actions == []
    assert inner_actions != []
    assert inner._observed_setpoints[-1] == pytest.approx(3.0)
    assert result is inner._last_value
    assert cascade.inner_loop.sp_trajectory is cascade.outer_loop


def test_cascade_auto_mode_tracks_outer_measurement():
    inner, inner_state = _prepare_controller(ramp_rate=None)
    outer, outer_state, outer_actions = _prepare_echo_controller()

    cascade = CascadeController(
        inner_loop=inner,
        outer_loop=outer,
    )
    cascade.mode = ControllerMode.auto

    inner_state.update({"t": 2.0, "value": 0.0})
    outer_state.update({"t": 2.0, "value": 5.0})

    cascade.update(2.0)

    assert outer.sp_trajectory.current_value(2.0) == pytest.approx(5.0)
    assert outer_actions == []


def test_cascade_update_trajectory_targets_active_source():
    inner, _ = _prepare_controller(ramp_rate=None)
    outer, _, _ = _prepare_echo_controller()

    cascade = CascadeController(inner_loop=inner, outer_loop=outer)

    cascade.mode = ControllerMode.cascade
    cascade.update_trajectory(0.0, 7.0)
    assert cascade.active_sp_trajectory().current_value(0.0) == pytest.approx(7.0)

    cascade.mode = ControllerMode.auto
    cascade.update_trajectory(0.0, 1.5)
    assert cascade.active_sp_trajectory().current_value(0.0) == pytest.approx(1.5)
