from pydantic import BaseModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modular_simulation.quantities import ControlOutputs

class ControlElements(BaseModel):
    """
    A Pydantic-based abstract class for representing control elements.
    Directly interacts with the States, but is externally controlled.
    Subclasses should define control variables as fields.
    """
    class Config:
        arbitrary_types_allowed = True

    def update(self, control_outputs: "ControlOutputs") -> None:
        """Updates the control element values from controller outputs."""
        update_field_names = self.__class__.model_fields.keys()
        for field_name, value in control_outputs.items():
            if field_name in update_field_names:
                setattr(self, field_name, value)