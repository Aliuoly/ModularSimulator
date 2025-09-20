import pytest

pytest.importorskip("numpy")

from modular_simulation.control_system.controller import Controller
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
