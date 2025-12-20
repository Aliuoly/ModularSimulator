import pytest

from modular_simulation.usables import (
    SampledDelayedSensor,
    PIDController,
    Trajectory,
    ControllerMode,
)
from modular_simulation.framework.system import System


def test_sampled_delayed_sensor_measures_and_histories(thermal_process_model):
    sensor = SampledDelayedSensor(
        measurement_tag="temperature",
        alias_tag="reactor_temperature",
        unit="K",
        sampling_period=0.5,
        deadtime=0.0,
    )

    sensor.initialize(0.0, thermal_process_model)
    first_sample = sensor.measure(0.0)
    assert first_sample.value == pytest.approx(300.0)

    thermal_process_model.temperature = 320.0
    held_sample = sensor.measure(0.25)
    assert held_sample.time == pytest.approx(0.0)

    delayed_sample = sensor.measure(1.0)
    assert delayed_sample.value == pytest.approx(320.0)
    assert len(sensor.measurement_history) == 2
    assert sensor.alias_tag == "reactor_temperature"


def test_pid_controller_updates_manipulated_variable(thermal_process_model):
    sensor = SampledDelayedSensor(
        measurement_tag="temperature",
        unit="K",
        sampling_period=0.0,
    )
    mv_sensor = SampledDelayedSensor(
        measurement_tag="heater_power",
        unit="K/s",
        sampling_period=0.0,
    )
    controller = PIDController(
        cv_tag="temperature",
        mv_tag="heater_power",
        sp_trajectory=Trajectory(310.0),
        Kp=2.0,
        Ti=5.0,
        mv_range=(0.0, 20.0),
        period=0.0,
    )

    system = System(
        dt=1.0,
        process_model=thermal_process_model,
        sensors=[sensor, mv_sensor],
        controllers=[controller],
        calculations=[],
        show_progress=False,
    )

    system.set_controller_mode("temperature", ControllerMode.AUTO)
    thermal_process_model.t = system.dt
    system._update_components()

    assert thermal_process_model.heater_power > 0.0
    assert controller._control_action.value == pytest.approx(thermal_process_model.heater_power)
    assert controller._sp_tag_info.tag == "temperature.sp"
