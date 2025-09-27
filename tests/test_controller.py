import pytest


pytest.importorskip("numpy")

from modular_simulation.control_system.controller import Controller, ControllerMode
from modular_simulation.control_system.trajectory import Trajectory
from modular_simulation.measurables import ControlElements, States
from modular_simulation.quantities import MeasurableQuantities, UsableQuantities, ControllableQuantities
from modular_simulation.usables import Sensor
from modular_simulation.validation.exceptions import ControllerConfigurationError


class ErrorEchoController(Controller):
    """
    Minimal concrete controller for exercising base-class behaviour.
    Returns the error defined as sp_value - cv_value as the control output.
    """

    def _control_algorithm(self, t, cv, sp):  
        return sp - cv

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
        mv_tag = 'mv1',
        cv_tag = 'cv1',
        sp_trajectory = Trajectory(1.0),
        mv_range = (-100,100),
    )

def prepare_improper_normal_controller():
    return ErrorEchoController(
        mv_tag = 'mv6',
        cv_tag = 'cv6',
        sp_trajectory = Trajectory(0),
        mv_range = (-100,100)
    )

def prepare_improper_cascade_controller_1():
    return ErrorEchoController(
        mv_tag = 'mv4',
        cv_tag = 'cv4',
        sp_trajectory = Trajectory(1.0),
        mv_range = (-50,50),
        cascade_controller = ErrorEchoController(
            mv_tag = 'mv5',
            cv_tag = 'cv5',
            sp_trajectory = Trajectory(2.0),
            mv_range = (-50,50)
        )
    )

def prepare_proper_cascade_controller_1():
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


def prepare_quantities():
    measurable = MeasurableQuantities(
        states = DummyStates(),
        control_elements = DummyControlElements(),
    )
    usable = UsableQuantities(
        sensors = [
            DummySensor(measurement_tag = 'mv1'),
            DummySensor(measurement_tag = 'mv2'),
            DummySensor(measurement_tag = 'mv3'),
            DummySensor(measurement_tag = 'cv1'),
            DummySensor(measurement_tag = 'cv2'),
            DummySensor(measurement_tag = 'cv3'),
        ],
        measurable_quantities = measurable
    )
    return measurable, usable

def test_normal_controller_instantiation():

    measurable, usable = prepare_quantities()
    controller = prepare_normal_controller()
    controllable = ControllableQuantities(
        controllers = [controller],
        control_elements = measurable.control_elements,
        usable_quantities = usable
    )

def test_controllable_quantities_construction():

    # case 1 - proper normal controller
    measurable, usable = prepare_quantities()
    controller = prepare_normal_controller()
    controllable = ControllableQuantities(
        controllers = [controller],
        control_elements = measurable.control_elements,
        usable_quantities = usable
    )

    # case 2 - improper normal controller
    measurable, usable = prepare_quantities()
    controller = prepare_improper_normal_controller()
    with pytest.raises(ExceptionGroup) as ex_info:
        controllable = ControllableQuantities(
            controllers = [controller],
            control_elements = measurable.control_elements,
            usable_quantities = usable
        )
    controller_errors = [ex for ex in ex_info.value.exceptions if isinstance(ex, ControllerConfigurationError)]
    assert len(controller_errors) > 0
    
    # case 3 - proper cascade controller
    measurable, usable = prepare_quantities()
    controller = prepare_proper_cascade_controller_1()
    controllable = ControllableQuantities(
        controllers = [controller],
        control_elements = measurable.control_elements,
        usable_quantities = usable
    )

    # case 4 - improper cascade controller
    measurable, usable = prepare_quantities()
    controller = prepare_improper_cascade_controller_1()
    with pytest.raises(ExceptionGroup) as ex_info:
        controllable = ControllableQuantities(
            controllers = [controller],
            control_elements = measurable.control_elements,
            usable_quantities = usable
        )
    controller_errors = [ex for ex in ex_info.value.exceptions if isinstance(ex, ControllerConfigurationError)]
    assert len(controller_errors) > 0



def test_cascade_controller_1():
    measurable, usable = prepare_quantities()
    controller = prepare_proper_cascade_controller_1()

    controllable = ControllableQuantities(
        controllers = [controller],
        control_elements = measurable.control_elements,
        usable_quantities = usable
    )
    controller = controllable.controllers[0]
    first_output = controller.update(t = 0.1).value

    # by default, things are in cascade
    # except the most-outer loop in AUTO. lets check that
    # only 1 layer of cascade here, hence:
    assert controller.mode == ControllerMode.CASCADE
    assert controller.cascade_controller.mode == ControllerMode.AUTO
    assert controller._sp_getter == controller.cascade_controller.update

    # now lets change inner loop to TRACKING
    controller._change_control_mode(mode = ControllerMode.TRACKING)
    assert controller.mode == ControllerMode.TRACKING
    # this should automatically change the outerloop to tracking as well
    assert controller.cascade_controller.mode == ControllerMode.TRACKING

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
    controller._change_control_mode(mode = 'AUTO')
    assert controller.mode == ControllerMode.AUTO
    assert controller.cascade_controller.mode == ControllerMode.TRACKING
    # since just changed from tracking mode, the setpoint should be unchanged
    # until something else changes it. Also, the sp_getter should
    # now point towards the trajectory
    assert controller._sp_getter(2) == controller.sp_trajectory(2)
    third_output = controller.update(t = 2).value
    fourth_output = controller.update(t = 3).value
    # since sp did not change between t = 2 and t = 3, output should be the same:
    assert third_output == fourth_output
    # now we change the setpoint and assert consistency of _sp_getter:
    controller.sp_trajectory.set_now(t = 4, value = 10)
    assert controller.sp_trajectory(t = 4) == controller._sp_getter(4)
    # and now check that control output has changed:
    fifth_output = controller.update(t = 5)
    assert fourth_output != fifth_output
    

if __name__ == '__main__':
    test_controllable_quantities_construction()
    test_cascade_controller_1()