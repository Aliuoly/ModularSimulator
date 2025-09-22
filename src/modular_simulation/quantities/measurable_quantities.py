from pydantic import BaseModel, ConfigDict, Field
from modular_simulation.measurables.base_classes import AlgebraicStates, States, ControlElements, Constants
from functools import cached_property

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
        default_factory = AlgebraicStates(),
        description = (
            "Algebraic states of the system to which this MeasurableQuantities definition applies. "
            "e.g., the outlet flow of a pressure driven stream dictated by the valve equation, "
                "the consumption rate of a species due to multiple reactions taking place."
        )
    )

    model_config = ConfigDict(extra = 'forbid')

    @cached_property
    def available_tags(self) -> list[str]:
        tags = []
        for category in [self.states, self.control_elements, self.algebraic_states, self.constants]:
            tags.extend(category.__class__.model_fields.keys())
        return tags
