"""Utilities for validating system configurations before simulation."""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modular_simulation.system import System
    from modular_simulation.quantities import (
        MeasurableQuantities,
        UsableQuantities,
        ControllableQuantities,
    )
import warnings 

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

    _check_duplicate_tags(usable, controllable)

    _initialize_sensors_and_calculations(measurable, usable)
    _initialize_controllers(measurable, usable, controllable)
    _warn_uncontrolled_control_elements(measurable, controllable)
    usable.update(0.0)

def _check_duplicate_tags(
        usable: "UsableQuantities",
        controllable: "ControllableQuantities"
    ) -> None:
    """checks if any duplicate sensors or controllers are defined.
    
    sensors are duplicate if they share the same measurament_tag 
    calculations are duplicate if they share the same output_tag
    controllers are duplicate if they share the same mv_tag (manipulated variable)
    """
    seen_measurement_tags = []
    for sensor in usable.sensors:
        tag = sensor.measurement_tag
        if tag in seen_measurement_tags:
            raise RuntimeError(
                f"Duplicate measurement tag '{tag}' found. Make sure each sensor has a unique measurement_tag."
            )
        seen_measurement_tags += [tag]

    seen_calculation_tags = []
    for calculation in usable.calculations:
        tag = calculation.output_tag
        if tag in seen_measurement_tags:    
            raise RuntimeError(
                f"Duplicate calculation tag '{tag}' found - overlapped with an existing sensor's measurement_tag. "
            )
        if tag in seen_calculation_tags:
            raise RuntimeError(
                f"Duplicate calculation tag '{tag}' found - overlapped with another calculation's output_tag. "
            )
        seen_calculation_tags += [tag]
    
    seen_mv_tag = []
    for controller in controllable.controllers:
        tag = controller.mv_tag
        if tag in seen_mv_tag:    
            raise RuntimeError(
                f"Duplicate controller '{controller.cv_tag}' found - mv '{tag}' in conflict with an another controller's manipulated variable. "
            )
        seen_mv_tag += [tag]
        

def _initialize_sensors_and_calculations(
        measurable: "MeasurableQuantities", 
        usable: "UsableQuantities"
        ) -> None:
    
    for sensor in usable.sensors:
        sensor._initialize(measurable)

    for calculation in usable.calculations:
        calculation._initialize(
            usable.sensors,
            usable.calculations
            )

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