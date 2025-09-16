import numpy as np
from numpy.typing import NDArray
from typing import Dict, Union, TYPE_CHECKING
from modular_simulation.control_system import Controller
from modular_simulation.quantities import UsableResults, UsableQuantities



ControlOutputs = Dict[str, Union[NDArray[np.float64], float]]
class ControllableQuantities:
    """
    Manages controllers and pre-links their sensor dependencies at initialization.
    """
    def __init__(
            self, 
            control_definitions: Dict[str, Controller],
            usable_quantities: UsableQuantities # Explicit dependency
            ):
        
        self.control_definitions = control_definitions
        
        # --- Validation and Linking Logic is now here in __init__ ---
        # This code runs only ONCE.
        for tag, controller in self.control_definitions.items():
            pv_tag = controller.pv_tag
            
            # Check if the required sensor tag exists in the provided UsableQuantities
            if pv_tag not in usable_quantities.measurement_definitions:
                raise ValueError(
                    f"Controller '{tag}' has a pv_tag ('{pv_tag}') that does not exist "
                    "in the UsableQuantities measurement definitions."
                )
            
            # If it exists, link the actual sensor object to the controller
            pv_sensor = usable_quantities.measurement_definitions[pv_tag]
            controller.link_pv_sensor(pv_sensor)
    
    def update(
            self, 
            usable_results: "UsableResults",
            t: float
            ) -> ControlOutputs:
        
        # The update method is now lean. It doesn't need validation flags
        # because the validation already happened in __init__.
        control_outputs: ControlOutputs = {}
        for tag, controller in self.control_definitions.items():
            control_outputs[tag] = controller.update(usable_results, t)

        return control_outputs
