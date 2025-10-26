import pytest
import pytest
from astropy.units import Quantity, Unit

from modular_simulation.core import create_system
from modular_simulation.interfaces import (
    ControllerBase,
    ControllerMode,
    SampledDelayedSensor,
    Trajectory,
)


class RampController(ControllerBase):
    gain: float = 1.0

    def _control_algorithm(self, t, cv, sp):  # type: ignore[override]
        return self._u0 + self.gain * (sp - cv)


@pytest.fixture()
def controller_setup(thermal_model, heater_mv_range):
    temp_sensor = SampledDelayedSensor(
        measurement_tag="temperature",
        alias_tag="temp_meas",
        sampling_period=0.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
        unit=Unit("K"),
    )
    mv_sensor = SampledDelayedSensor(
        measurement_tag="heater_power",
        alias_tag="heater_power",
        sampling_period=0.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
        unit=Unit("W"),
    )
    controller = RampController(
        mv_tag="heater_power",
        cv_tag="temp_meas",
        sp_trajectory=Trajectory(y0=300.0, unit=Unit("K")),
        mv_range=heater_mv_range,
        ramp_rate=10.0,
        gain=0.5,
    )
    system = create_system(
        dt=Quantity(1.0, Unit("s")),
        dynamic_model=thermal_model,
        sensors=[temp_sensor, mv_sensor],
        controllers=[controller],
        use_numba=False,
        record_history=False,
    )
    return controller, system


def test_controller_initial_mode_is_auto(controller_setup):
    controller, system = controller_setup
    assert controller.mode == ControllerMode.AUTO
    assert controller._sp_getter == controller.sp_trajectory


def test_controller_update_applies_ramp(controller_setup):
    controller, system = controller_setup

    system.dynamic_model.temperature = 290.0
    controller.sp_trajectory.set_now(0.0, 310.0)
    system.update(1.0)
    system.dynamic_model.temperature = 290.0
    system.update(2.0)
    assert controller._last_output.ok is True

    allowed_delta = controller.ramp_rate * 1.0
    expected = controller._last_output.value
    assert expected == pytest.approx(controller._u0 + allowed_delta, rel=1e-6)
    assert system.dynamic_model.heater_power == expected


def test_controller_mode_switching(controller_setup):
    controller, system = controller_setup

    controller.change_control_mode(ControllerMode.TRACKING)
    assert controller.mode == ControllerMode.TRACKING

    system.dynamic_model.temperature = 305.0
    system.update(1.0)
    system.dynamic_model.temperature = 305.0
    system.update(2.0)
    assert controller.sp_trajectory(2.0) == pytest.approx(305.0)

    controller.change_control_mode("auto")
    assert controller.mode == ControllerMode.AUTO

    controller.sp_trajectory.set_now(3.0, 315.0)
    system.dynamic_model.temperature = 305.0
    system.update(3.0)
    assert controller._last_output.time == pytest.approx(3.0)
