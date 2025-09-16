from pydantic import BaseModel, ConfigDict
from abc import ABC
from numpy.typing import NDArray
import numpy as np
from enum import Enum
from typing import get_origin, ClassVar

class States(BaseModel, ABC):
    """
    Abstract base class for differential state variables in a simulation.

    This class uses Pydantic for data validation and structure, and includes
    robust, generic logic to ensure that any subclass is correctly defined
    with a matching StateMap enum for NumPy array conversion.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    def __init_subclass__(cls, **kwargs):
        """
        Performs validation on subclasses when they are defined.
        Ensures a valid StateMap exists and perfectly matches the model fields.
        """
        super().__init_subclass__(**kwargs)
        
        # Don't run validation on this base class itself.
        if ABC in cls.__bases__:
            return

        if not hasattr(cls, 'StateMap') or not issubclass(cls.StateMap, Enum):
            raise TypeError(f"{cls.__name__} must define a 'StateMap' class attribute that is an Enum.")

        # Use the annotations, which are required by Pydantic
        # to check non StateMap fields, which are to be defined in the StateMap Enum
        model_field_names = set()
        for name, type_hint in getattr(cls, '__annotations__', {}).items():
            # A more robust check to see if the annotation is a ClassVar.
            is_class_var = (get_origin(type_hint) is ClassVar) or (type_hint is ClassVar)

            # Exclude ClassVars and private attributes from the field check.
            if not is_class_var and not name.startswith('_'):
                model_field_names.add(name)
        
        enum_member_names = {member.name for member in cls.StateMap}
        
        # Now, compare the sets and provide a detailed error message if they don't match.
        if model_field_names != enum_member_names:
            missing_in_model = enum_member_names - model_field_names
            missing_in_enum = model_field_names - enum_member_names
            error_message = (
                f"The fields in {cls.__name__} and its StateMap Enum do not match.\n"
            )
            if missing_in_model:
                error_message += f"  - Fields in enum but not in model: {missing_in_model}\n"
            if missing_in_enum:
                error_message += f"  - Fields in model but not in enum: {missing_in_enum}\n"
            raise TypeError(error_message)


    @classmethod
    def get_total_size(cls) -> int:
        """Calculates the total size of the flat NumPy array representation."""
        max_val = 0
        for member in cls.StateMap: #type: ignore
            if isinstance(member.value, int):
                max_val = max(max_val, member.value + 1)
            elif isinstance(member.value, slice):
                max_val = max(max_val, member.value.stop)
        return max_val

    def to_array(self) -> NDArray[np.float64]:
        """Converts the Pydantic model instance to a flat NumPy array."""
        array = np.zeros(self.get_total_size(), dtype=np.float64)
        for member in self.StateMap: #type: ignore
            array[member.value] = getattr(self, member.name)
        return array
    
    @classmethod
    def from_array(cls, array: NDArray) -> "States":
        """Creates a Pydantic model instance from a flat NumPy array."""
        kwargs = {member.name: array[member.value] for member in cls.StateMap} #type: ignore
        return cls(**kwargs)

