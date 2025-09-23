from pydantic import BaseModel, PrivateAttr, ConfigDict
from typing import Dict, Type
import numpy as np
from numpy.typing import NDArray
from enum import Enum

class BaseIndexedModel(BaseModel):
    """
    Base class for the measurable data containers which are
    indexible via an internally constructed Enum. 
    Can be converted to numpy arrays via said indexing, 
    and can be updating from numpy arrays via the same indexing. 
    """

    _index_map: Type[Enum] = PrivateAttr()
    _index_dict: Dict[str, slice] = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    
    def model_post_init(self, context):
        # generate _index_map based on the defined fields
        enum_members = {}
        slice_start = 0
        for field_name in self.__class__.model_fields:
            try:
                value = np.asarray(getattr(self, field_name))
            except Exception as ex:
                raise TypeError(
                    f"Field '{field_name}' of class '{self.__class__.__name__}' is not coercable into a numpy array. "
                    f"coercion exception: {ex}"
                )
            slice_end = slice_start + value.size
            enum_members[field_name] = slice(slice_start, slice_end)
            slice_start = slice_end
        self._index_map = Enum("index_map", enum_members)
        self._index_dict = {
            name: member.value for name, member in self._index_map.__members__.items()  # type: ignore[attr-defined]
        }


    def get_total_size(self) -> int:
        """Calculates the total size of the flat NumPy array representation."""
        max_val = 0
        for member in self._index_map:
            max_val = max(max_val, member.value.stop)
        return max_val

    def to_array(self) -> NDArray[np.float64]:
        """Converts the Pydantic model instance to a flat NumPy array."""
        array = np.zeros(self.get_total_size(), dtype=np.float64)
        for member in self._index_map: 
            array[member.value] = getattr(self, member.name)
        return array
    
    def update_from_array(self, array: NDArray[np.float64]) -> None:
        """updates the class in place using the provided array."""
        for member in self._index_map:
            setattr(self, member.name, array[member.value])

    def index_map_dict(self) -> Dict[str, slice]:
        """Returns a dictionary mapping field names to their array slices."""
        return self._index_dict

class ControlElements(BaseIndexedModel):
    """
    Base container class for control elements.
    Directly interacts with the States, but is externally controlled.
    Subclasses should define control variables as fields.
    """
    # e.g. user defined fields:
    # flow_rate: float = 0.0
    # inlet_temperature: float = 273.

class States(BaseIndexedModel):
    """
    Base container class for differential state variables in a simulation.
    """

class AlgebraicStates(BaseIndexedModel):

    """
    Base container class for quantities that are algebraic functions of states. 
    Calculation is defined in the system. It may be calculated from any method
    that does not involve integration.
    """
    # e.g. user defined fields:
    # outlet_flow: float = 0.0
        # e.g. calculated as a pressure driven flow = Cv * f(dP)

class Constants(BaseIndexedModel):
    """
    Base container class for constants in a simulation.
    """
    

