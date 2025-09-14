import numpy as np
from numpy.typing import NDArray
from typing import Dict, Union, Any, List, Optional
from pydantic import BaseModel, PrivateAttr

from modular_simulation.usables import Calculation, Sensor
from modular_simulation.measurables import States, ControlElements, AlgebraicStates
from modular_simulation.control_system import Controller



class MeasurableQuantities(BaseModel):
    states: States
    control_elements: ControlElements
    algebraic_states: Optional[AlgebraicStates] = None


UsableResults = Dict[str, Any]
class UsableQuantities:
    """
    1. Defines how measurements and calculations are obtained through
        - measurement_definition: Dict[str, Sensor]
        - calculation_definition: Dict[str, Calculation]
    2. Saves the current snapshot of measurements and calculations
        - results: Dict[str, Any]
    """
    measurement_definitions: Dict[str, Sensor]
    calculation_definitions: Dict[str, Calculation]

    _tag_list: List[str] = PrivateAttr(default_factory=list)
    _usable_results: UsableResults = PrivateAttr(default_factory=dict)
    
    #model_config = dict(arbitrary_types_allowed=True)

    def __init__(self,
                 measurement_definitions: Dict[str, Sensor],
                 calculation_definitions: Dict[str, Calculation]):
        self.measurement_definitions = measurement_definitions
        self.calculation_definitions = calculation_definitions
        self._tag_list = list(self.measurement_definitions.keys()) \
                      + list(self.calculation_definitions.keys())
        self._usable_results = {tag: None for tag in self._tag_list}

    def update(
            self, 
            measurable_quantities: MeasurableQuantities
            ) -> UsableResults:
        

        for tag, sensor in self.measurement_definitions.items():
            self._usable_results[tag] = sensor.measure(measurable_quantities)
    
        for tag, calculation in self.calculation_definitions.items():
            self._usable_results[tag] = calculation.calculate(self._usable_results)

        return self._usable_results



ControlOutputs = Dict[str, Union[NDArray[np.float64], float]]
class ControllableQuantities:

    control_definitions: Dict[str, Controller]

    _tag_list: List[str] = PrivateAttr(default_factory=list)
    _control_outputs: ControlOutputs = PrivateAttr(default_factory=dict)

    #model_config = dict(arbitrary_types_allowed=True)

    def __init__(self, control_definitions: Dict[str, Controller]):
        self.control_definitions = control_definitions
        self._tag_list = [output_name for output_name in self.control_definitions.keys()]
        self._control_outputs = {tag: None for tag in self._tag_list}   

        # Enforce all Controlled tags are specified in Trajectories
   
    def update(
            self, 
            usable_results: UsableResults,
            t: float
            ) -> ControlOutputs:
        
        for output_name, controller in self.control_definitions.items():
            self._control_outputs[output_name] = controller.update(usable_results, t)

        return self._control_outputs
