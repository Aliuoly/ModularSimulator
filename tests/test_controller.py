import pytest


pytest.importorskip("numpy")

from modular_simulation.control_system.controllers.controller import Controller, ControllerMode
from modular_simulation.control_system.trajectory import Trajectory
from modular_simulation.measurables import ControlElements, States
from modular_simulation.quantities import MeasurableQuantities, UsableQuantities
from modular_simulation.usables import Sensor
from modular_simulation.validation.system_validation import _initialize_sensors_and_calculations


class ErrorEchoController(Controller):
    """
    Minimal concrete controller for exercising base-class behaviour.
    Returns the error defined as sp_value - cv_value as the control output.
    """

    def _control_algorithm(self, t, cv_value, sp_value):  
        return sp_value - cv_value

class DummyStates(States):
    
    mv2: float = 10.0
    mv3: float = 100.0
    cv1: float = -5.0
    cv2: float = 5.0
    cv3: float = 15.0

class DummyControlElements(ControlElements):
    mv1: float = -5.0

class DummySensor(Sensor):
    def _get_processed_value(self, t, raw_value):
        return raw_value # do nothing
    def _should_update(self, t):
        return True # always update
    
    
def prepare_normal_controller():
    return ErrorEchoController(
        mv_tag = 'mv',
        cv_tag = 'cv',
        sp_trajectory = Trajectory(1.0),
        mv_range = (-100,100),
    )

def prepare_cascade_controller_1():
    return ErrorEchoController(
        mv_tag = 'mv1',
        cv_tag = 'cv1',
        sp_trajectory = Trajectory(1.0),
        mv_range = (-50,50),
        cascade_controller = ErrorEchoController(
            mv_tag = 'mv2',
            cv_tag = 'cv2',
            sp_trajectory = Trajectory(2.0),
            mv_range = (-50,50)
        )
    )

def prepare_cascade_controller_2():
    return ErrorEchoController(
        mv_tag = 'mv1',
        cv_tag = 'cv1',
        sp_trajectory = Trajectory(1.0),
        mv_range = (-50,50),
        mode = ControllerMode.TRACKING,
        cascade_controller = ErrorEchoController(
            mv_tag = 'mv2',
            cv_tag = 'cv2',
            sp_trajectory = Trajectory(10.0),
            mv_range = (-50,50),
            cascade_controller = ErrorEchoController(
                mv_tag = 'mv3',
                cv_tag = 'cv3',
                sp_trajectory = Trajectory(100.0),
                mv_range = (-50,50)
            )
        )
    )

def prepare_usable():
    return UsableQuantities(
        sensors = [
            DummySensor(measurement_tag = 'mv1'),
            DummySensor(measurement_tag = 'mv2'),
            DummySensor(measurement_tag = 'mv3'),
            DummySensor(measurement_tag = 'cv1'),
            DummySensor(measurement_tag = 'cv2'),
            DummySensor(measurement_tag = 'cv3'),
        ]
    )

def prepare_measurable():
    return MeasurableQuantities(
        states = DummyStates(),
        control_elements = DummyControlElements(),
    )

def test_normal_controller():

    usable = prepare_usable()
    measurable = prepare_measurable()
    controller = prepare_normal_controller()
    # simulate initialization logic during system creation
    controller._initialize(usable, measurable.control_elements)

    output = controller.update(t = 0)
    assert output.t == 0.0

def test_cascade_controller_1():
    usable = prepare_usable()
    measurable = prepare_measurable()
    controller = prepare_cascade_controller_1()

    _initialize_sensors_and_calculations(
        measurable, usable
    )
    controller._initialize(usable, measurable.control_elements)
    first_output = controller.update(t = 0).value

    # expected behavior:
    # currently the inner loop is in TRACKING mode
    # thus, the output at t = 0 (old trajectory)
    # and the output at t = 1 (new trajecty from .set_now)
    # should be the same, since the setpoint should internally get
    # set to the CV value and ignore any other setpoint source. 
    controller.sp_trajectory.set_now(t = 1, value = 10)

    # even though we just set it to 10, the controller should NOT 
    # see 10, but see the cv value instead
    assert controller.sp_trajectory(t = 1) == 10
    assert controller.sp_trajectory(t = 1) != controller._sp_getter(1).value
    assert controller._sp_getter(1).value == controller._cv_getter().value
    # and as such, the output from t = 1 and t = 0 should be identical (since same sp - the cv value itself)
    second_output = controller.update(t = 1).value
    assert first_output == second_output

    # now let's change to AUTO mode
    # expected behavior:
    # when the sp_trajectory's return changes, the output should change as well. 
    # if it doesn't change, the output should not change either 
    #   (ONLY because of our dummy controller implementation. A PID e.g., would not do this due to time dynamics)
    controller._change_control_mode(mode = 'AUTO')
    third_output = controller.update(t = 2).value
    # since we JUST changed the mode, the sp should be tracking until someone sets it, hence:
    assert second_output == third_output
    fourth_output = controller.update(t = 3).value
    # since sp did not change between t = 2 and t = 3, output should be the same:
    assert third_output == fourth_output
    # now we change the setpoint and assert consistency of _sp_getter:
    controller.sp_trajectory.set_now(t = 4, value = 10)
    assert controller.sp_trajectory(t = 4) == controller._sp_getter(4)
    # and now check that control output has changed:
    fifth_output = controller.update(t = 5)
    assert fourth_output != fifth_output

    # moving on to CASCADE mode
    # since inner loop was on AUTO, cascade controller should be in TRACKING mode:
    assert controller.cascade_controller.mode == ControllerMode.TRACKING
    # and the sp should be equal to the cv
    assert controller.cascade_controller._sp_getter(5) == controller.cascade_controller._cv_getter()
    # now we change mode to CASCADE and verify that the cascade controller is not in AUTO mode
    # and the inner loop is in CASCADE mode:
    controller._change_control_mode(mode = 'CASCADE')
    assert controller.cascade_controller.mode == ControllerMode.AUTO
    assert controller.mode == ControllerMode.CASCADE
    # and we once again verify that the sp_getter of the inner loop correctly 
    # grabs the output of the cascade controller
    assert controller._sp_getter(6).value == controller.cascade_controller.update(t = 6).value
    

test_cascade_controller_1()