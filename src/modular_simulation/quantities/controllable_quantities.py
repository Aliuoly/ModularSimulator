import numpy as np
from numpy.typing import NDArray
from typing import Dict, Union
from modular_simulation.control_system import Controller
from modular_simulation.quantities import UsableResults


ControlOutputs = Dict[str, Union[NDArray[np.float64], float]]


class ControllableQuantities:
    """Container for the controllers acting on the system's control elements."""

    def __init__(
        self,
        control_definitions: Dict[str, Controller],
    ) -> None:
        self.control_definitions = control_definitions
    
    def update(
            self, 
            usable_results: "UsableResults",
            t: float
            ) -> ControlOutputs:
        
        # Validation occurs during system setup, so this method can focus solely
        # on delegating to the controllers.
        control_outputs: ControlOutputs = {}
        for tag, controller in self.control_definitions.items():
            control_outputs[tag] = controller.update(usable_results, t)

        return control_outputs
