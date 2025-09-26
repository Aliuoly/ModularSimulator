from pydantic import BaseModel, ConfigDict, Field, model_validator
from modular_simulation.quantities.utils import ConfigurationError
from modular_simulation.measurables.base_classes import AlgebraicStates, States, ControlElements, Constants
from functools import cached_property
from typing import Iterable



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
            raise ConfigurationError(
                "The following duplicate tag(s) were detected in the measurable quantity definition: "
                f"{', '.join(duplicate_tag_list)}"
            )
        if len(seen_tag_list) == 0:
            raise ConfigurationError(
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
