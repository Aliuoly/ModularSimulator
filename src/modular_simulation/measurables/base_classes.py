from pydantic import BaseModel, PrivateAttr, ConfigDict
from typing import Dict, List
import numpy as np
from numpy.typing import NDArray
from functools import cached_property

class BaseIndexedModel(BaseModel):
    """
    Base class for the measurable data containers which are
    indexible via an internally constructed Enum. 
    Can be converted to numpy arrays via said indexing, 
    and can be updating from numpy arrays via the same indexing. 
    """

    _index_map: Dict[str, slice] = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    
    def model_post_init(self, context):
        # generate _index_map based on the defined fields
        index_dict = {}
        slice_start = 0
        for field_name in self.model_dump():
            try:
                value = np.asarray(getattr(self, field_name))
            except Exception as ex:
                raise TypeError(
                    f"Field '{field_name}' of class '{self.__class__.__name__}' is not coercable into a numpy array. "
                    f"coercion exception: {ex}"
                )
            slice_end = slice_start + value.size
            index_dict[field_name] = slice(slice_start, slice_end)
            slice_start = slice_end
        self._index_map = index_dict

        # get array size
        array_size = 0
        for slice_of_field in self._index_map.values():
            array_size = max(array_size, slice_of_field.stop)
        self._array_size = array_size

    @cached_property
    def tag_list(self) -> List[str]:
        return list(self._index_map.keys())

    def to_array(self) -> NDArray[np.float64]:
        # the following combination, from testing, gave the best times
        # use np.zeros(...) to remake array each time
        #   instead of np.empty(...) and alike
        #   using a preallocated array and updating it
        #   was no faster and flowed things down due to
        #   requirement of additional checking logics. 
        # use dictionary for array indexing instead of 
        #   an enum. 
        array = np.zeros(self._array_size, dtype=np.float64)
        for attr_name, slice in self._index_map.items(): 
            array[slice] = getattr(self, attr_name)
        return array

    def update_from_array(self, array: NDArray[np.float64]) -> None:
        """updates the class in place using the provided array."""
        for field_name, field_index in self._index_map.items():
            if field_index.start == field_index.stop - 1:
                setattr(self, field_name, array[field_index][0])
                continue
            setattr(self, field_name, array[field_index])
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
    

