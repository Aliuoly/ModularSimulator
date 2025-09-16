from typing import Optional, Any, TYPE_CHECKING
from pydantic import BaseModel
from modular_simulation.measurables import AlgebraicStates
from modular_simulation.measurables import States, ControlElements



class MeasurableQuantities(BaseModel):
    states: States
    control_elements: ControlElements
    algebraic_states: Optional[AlgebraicStates] = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.algebraic_states is None:
            self.algebraic_states = AlgebraicStates()