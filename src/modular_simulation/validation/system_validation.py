"""Utilities for validating system configurations before simulation."""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modular_simulation.framework.system import System
    from modular_simulation.quantities import (
        MeasurableQuantities,
        UsableQuantities,
        ControllableQuantities,
    )
import warnings 
class ConfigurationError(Exception):
    """
    A custom error that arises from mistakes in measurable, usable, or controllable quantity definition. 
    """
    def __init__(self, message = "An validation error occured"):
        self.message = message
        super().__init__(self.message)

def validate_and_link(system: "System") -> None:
    """Validate the system configuration and link dependent components.

    This ensures that:
        * every control element has a corresponding controller.
        * each controller's process-variable tag exists and is linked to a sensor.
        * measurements and calculations can be evaluated successfully.
    """
    measurable = system.measurable_quantities
    usable = system.usable_quantities
    controllable = system.controllable_quantities

    _initialize_controllers(measurable, usable, controllable)
    _warn_uncontrolled_control_elements(measurable, controllable)
    usable.update(0.0)




def _initialize_controllers(
        measurable: "MeasurableQuantities", 
        usable: "UsableQuantities", 
        controllable:"ControllableQuantities"
        ) -> None:
    for controller in controllable.controllers:
        controller._initialize(
            usable_quantities = usable,
            control_elements = measurable.control_elements
            )

def _warn_uncontrolled_control_elements(
        measurable: "MeasurableQuantities",
        controllable: "ControllableQuantities"
        ) -> None:
    for ce_name in measurable.control_elements.__class__.model_fields:
        found = False
        for controller in controllable.controllers:
            if controller.mv_tag == ce_name:
                found = True
        
        if not found:
            warnings.warn(
                f"Control element {ce_name} is defined but not controlled; "
                "as such, it will remain at the value initialized. "
                "If this is not intended, ensure a controller with "
                "cv_tag equal to the name of the control element is defined in controllable quantities. "
                f"Currently defined controllers control {", ".join([controller.mv_tag for controller in controllable.controllers])}"
            )