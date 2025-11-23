import pytest

from modular_simulation.framework.system import System
from modular_simulation.usables import SampledDelayedSensor, PIDController, Trajectory, ControllerMode
from modular_simulation.validation.exceptions import SensorConfigurationError, ControllerConfigurationError


def _make_basic_sensor():
    return SampledDelayedSensor(
        measurement_tag="temperature",
        unit="K",
    )


def _make_pid_controller(setpoint: float = 310.0):
    return PIDController(
        cv_tag="temperature",
        mv_tag="heater_power",
        sp_trajectory=Trajectory(setpoint),
        Kp=1.5,
        Ti=4.0,
        mv_range=(0.0, 25.0),
        period=0.0,
    )


def test_system_step_updates_measurements_and_state(thermal_process_model):
    sensor = _make_basic_sensor()
    mv_sensor = SampledDelayedSensor(measurement_tag="heater_power", unit="K/s")
    controller = _make_pid_controller(315.0)
    system = System(
        dt=1.0,
        process_model=thermal_process_model,
        sensors=[sensor, mv_sensor],
        controllers=[controller],
        calculations=[],
        record_history=False,
        show_progress=False,
    )

    system.set_controller_mode("temperature", ControllerMode.AUTO)
    thermal_process_model.t = system.dt
    system.step()

    assert thermal_process_model.temperature > 300.0
    assert system.history == {}
    measured_history = system.measured_history
    assert "sensors" in measured_history
    assert sensor.alias_tag in measured_history["sensors"]
    assert len(measured_history["sensors"][sensor.alias_tag]) >= 1


def test_system_validation_catches_bad_sensor(thermal_process_model):
    bad_sensor = SampledDelayedSensor(measurement_tag="invalid", unit="K")

    with pytest.raises(ExceptionGroup) as excinfo:
        System(
            dt=1.0,
            process_model=thermal_process_model,
            sensors=[bad_sensor],
            controllers=[],
            calculations=[],
            show_progress=False,
        )

    assert any(isinstance(err, SensorConfigurationError) for err in excinfo.value.exceptions)


def test_system_validation_catches_bad_controller(thermal_process_model):
    sensor = _make_basic_sensor()
    bad_controller = PIDController(
        cv_tag="temperature",
        mv_tag="invalid_mv",
        sp_trajectory=Trajectory(310.0),
        Kp=1.0,
        Ti=3.0,
        mv_range=(0.0, 10.0),
    )

    with pytest.raises(ExceptionGroup) as excinfo:
        System(
            dt=1.0,
            process_model=thermal_process_model,
            sensors=[sensor],
            controllers=[bad_controller],
            calculations=[],
            show_progress=False,
        )

    assert any(isinstance(err, ControllerConfigurationError) for err in excinfo.value.exceptions)
