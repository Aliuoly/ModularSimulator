from pydantic import BaseModel, ConfigDict, Field, model_validator, PrivateAttr
from modular_simulation.validation.exceptions import MeasurableConfigurationError
from modular_simulation.measurables.base_classes import AlgebraicStates, States, ControlElements, Constants
from functools import cached_property
from typing import Iterable, Dict
from astropy.units import UnitBase # type:ignore




class MeasurableQuantities(BaseModel):
    constants: Constants = Field(
        default_factory = Constants,
        description = (
            "Constants of the system to which this MeasurableQuantities definition applies. "
            "e.g., geometric properties of vessels, rate constants for reactions, ..."
        )
    )
    states: States = Field(
        default_factory = States,
        description = (
            "Differential states of the system to which this MeasurableQuantities definition applies. "
            "e.g., concentration of a species or temperature in the reaction medium."
        )
    )
    control_elements: ControlElements = Field(
        default_factory = ControlElements,
        description = (
            "Final control elements of the system to which this MeasurableQuantities definition applies."
            "e.g., the feed rate of a stream to the system."
        )
    )
    algebraic_states: AlgebraicStates = Field(
        default_factory = AlgebraicStates,
        description = (
            "Algebraic states of the system to which this MeasurableQuantities definition applies. "
            "e.g., the outlet flow of a pressure driven stream dictated by the valve equation, "
                "the consumption rate of a species due to multiple reactions taking place."
        )
    )
    _unit_info: Dict[str, UnitBase] = PrivateAttr()
    model_config = ConfigDict(extra = 'forbid')
    
    @model_validator(mode = 'after')
    def validate_tag_list(self):
        """ensures no tag shows up in multiple places."""
        duplicate_tag_list = []
        seen_tag_list = []
        for tag in self.tag_list:
            if tag in seen_tag_list:
                duplicate_tag_list.append(tag)
            seen_tag_list.append(tag)
        if len(duplicate_tag_list) > 0:
            raise MeasurableConfigurationError(
                "The following duplicate tag(s) were detected in the measurable quantity definition: "
                f"{', '.join(duplicate_tag_list)}"
            )
        if len(seen_tag_list) == 0:
            raise MeasurableConfigurationError(
                "No measurable quantities defined. Aborting."
            )
        return self
    
    @cached_property
    def tag_list(self) -> Iterable[str]:
        return_list = list(self.states.tag_list)
        return_list.extend(list(self.algebraic_states.tag_list))
        return_list.extend(list(self.control_elements.tag_list))
        return_list.extend(list(self.constants.tag_list))
        return return_list
    
    @cached_property
    def tag_unit_info(self) -> Dict[str, UnitBase]:
        return_dict = {}
        return_dict.update(self.algebraic_states.tag_unit_info)
        return_dict.update(self.states.tag_unit_info)
        return_dict.update(self.constants.tag_unit_info)
        return_dict.update(self.control_elements.tag_unit_info)
        return return_dict
