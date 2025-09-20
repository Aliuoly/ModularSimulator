from typing import Optional, Any
from pydantic import BaseModel
from modular_simulation.measurables.algebraic_states import AlgebraicStates
from modular_simulation.measurables.states import States
from modular_simulation.measurables.control_elements import ControlElements



class MeasurableQuantities(BaseModel):
    states: States
    control_elements: ControlElements
    algebraic_states: Optional[AlgebraicStates] = None


    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.algebraic_states is None:
            self.algebraic_states = AlgebraicStates()

    @property
    def available_tags(self) -> list[str]:
        tags = []
        for category in [self.states, self.control_elements, self.algebraic_states]:
            if category:
                tags.extend(category.model_fields.keys())
        return tags
