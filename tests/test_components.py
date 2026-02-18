import pytest

from modular_simulation.components import (
    SampledDelayedSensor,
    PIDController,
    Trajectory,
    ControllerMode,
    ControlElement,
)
from modular_simulation.framework.system import System


def test_sampled_delayed_sensor_measures_and_histories(thermal_process_model):
    sensor = SampledDelayedSensor(
        measurement_tag="temperature",
        alias_tag="reactor_temperature",
        unit="K",
        sampling_period=0.5,
        deadtime=0.2,
    )

    # Sensors now require a System for initialization logic (e.g. tag resolution)
    # We create a dummy system using the fixture model
    system = System(
        dt=0.05,
        process_model=thermal_process_model,
        sensors=[sensor],
        calculations=[],
        control_elements=[],
        show_progress=False,
    )

    # Initialization happens in System.__init__ now if we pass components there
    # But let's verify it happened
    assert sensor._initialized

    # First sample at t=0
    # System init calls install() but doesn't run first update() automatically
    # (Wait, yes it does? System.__init__ calls _update_components(0.0)?)
    # Let's check system definition. Usually System.step() drives it.

    # Manually check initial state
    assert sensor.point.data.value == pytest.approx(300.0)

    # Step to 0.25 (half sample period) - should hold value
    # But wait, we need to update process model first to see a change IF it sampled
    system.process_model.temperature = 320.0
    system.step(duration=0.25)

    # Sensor logic: if t < last + period, return holding value
    # Last sample was at t=0 (300K). Next sample at 0.5.
    # So at 0.25, it should still be 300K (holding), ignoring the 320K change
    assert sensor.point.data.value == pytest.approx(300.0)
    assert sensor.point.data.time == pytest.approx(0.0)  # Timestamp of the SAMPLE

    # Step to 0.5 - should sample now
    # Step to 0.5 - simulation reaches 0.5, but components updated at 0.45 last
    system.step(duration=0.25)  # t=0.5

    # Trigger one more step so components see t=0.5 and sample
    system.step(duration=system.dt)

    # Now it should HAVE sampled (internally), but because of DEADTIME=0.2...
    # Sample taken at t=0.5. Available at t=0.7.
    # Current t=0.55.
    # So visible measurement should STILL be Sample(t=0).
    assert sensor.point.data.time == pytest.approx(0.0)
    assert sensor.point.data.value == pytest.approx(300.0)

    # Now step past deadtime (target 0.75)
    remaining = 0.75 - system.time
    system.step(remaining)

    # Now t=0.75. Sample(t=0.5) should be visible.
    assert sensor.point.data.time == pytest.approx(0.5)
    # The value will be whatever the physics evolved to, definitely NOT 300.0
    assert sensor.point.data.value != pytest.approx(300.0)

    assert sensor.alias_tag == "reactor_temperature"


def test_pid_controller_updates_manipulated_variable(thermal_process_model):
    # Setup: Sensor for PV, Sensor for MV (optional, but good for tracking),
    # ControlElement with PIDController

    pv_sensor = SampledDelayedSensor(
        measurement_tag="temperature",
        unit="K",
        sampling_period=0.0,  # Continuous
    )

    # In new arch, controller sits INSIDE ControlElement
    controller = PIDController(
        cv_tag="temperature",
        cv_range=(200.0, 400.0),
        sp_trajectory=Trajectory(y0=310.0),  # SP = 310
        Kp=2.0,
        Ti=5.0,
    )

    control_element = ControlElement(
        mv_tag="heater_power",
        mv_range=(0.0, 20.0),
        mv_trajectory=Trajectory(y0=0.0),
        controller=controller,
        mode=ControllerMode.AUTO,  # Start in AUTO
    )

    system = System(
        dt=1.0,
        process_model=thermal_process_model,
        sensors=[pv_sensor],
        calculations=[],
        control_elements=[control_element],
        show_progress=False,
    )

    # Initial state:
    # PV = 300, SP = 310, Error = 10
    # PID starts with 0 integral if commissioned well?
    # PID update at t=0? usually update() checks period.

    # Run one step
    system.step()
    # t=1.0.
    # PV is read from t=0 state? System update order matters:
    # 1. Sensors update (read T=300)
    # 2. Control logic updates (uses T=300)
    # 3. Process model step (uses MV calculated in 2)

    # Expected Control Action (Velocity form default is False? Position form?)
    # PIDController defaults: Kp=2, Ti=5, velocity_form=False
    # Error = 10
    # P = 2 * 10 = 20
    # I = 2/5 * Integral(e*dt). First step dt? or 0?
    # If using positional form, typically integral starts at previous MV equivalent?
    # Or just 0 accumulator.
    # At t=0 update: error=10.
    # If we step, integration happens.

    # Let's just verify it MOVES in the right direction
    # Heater power should increase because T(300) < SP(310)

    assert system.process_model.heater_power > 0.0

    # Check that control element updated
    assert control_element._mv_point.data.value == pytest.approx(system.process_model.heater_power)

    # Check controller internals (public access has changed to PrivateAttr for state)
    # We can check the point data in tag_store
    assert system.point_registry["temperature.sp"].data.value == 310.0
