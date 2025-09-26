
from modular_simulation.control_system.controllers.controller import Controller
from modular_simulation.measurables import ControlElements, States
from modular_simulation.usables import Sensor, Calculation


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
    
class AddAllCalculation(Calculation):
    def _calculation_algorithm(self, t, inputs_dict):
        return sum(inputs_dict.values())

    