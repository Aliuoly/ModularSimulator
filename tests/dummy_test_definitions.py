
from modular_simulation.control_system.controller import Controller
from modular_simulation.measurables import ControlElements, States
from modular_simulation.usables import (
    Sensor,
    Calculation,
    MeasuredTag,
    OutputTag,
)


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
    
class AddAllCalculation(Calculation):
    output_tag: OutputTag

    mv1_tag: MeasuredTag
    mv2_tag: MeasuredTag
    cv1_tag: MeasuredTag
    cv2_tag: MeasuredTag

    def _calculation_algorithm(self, t, inputs_dict):
        result = sum(inputs_dict.values())
        return {self.output_tag: result}

    