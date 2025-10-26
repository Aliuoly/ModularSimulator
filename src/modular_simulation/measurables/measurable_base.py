from pydantic import BaseModel, PrivateAttr, ConfigDict, model_validator
import numpy as np
from numpy.typing import NDArray
from functools import cached_property
from modular_simulation.validation.exceptions import MeasurableConfigurationError
from astropy.units import UnitBase, Unit
class MeasurableBase(BaseModel):
    """
    Base class for the measurable data containers which are
    indexible via an internally constructed Enum. 
    Can be converted to numpy arrays via said indexing, 
    and can be updating from numpy arrays via the same indexing. 
    """
    _index_map: dict[str, slice] = PrivateAttr()
    _tag_unit_info: dict[str, UnitBase] = PrivateAttr(default_factory = dict)
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

    @model_validator(mode='after')
    def _ensure_unit_annotated(self):
        for field_name, field_info in self.__class__.model_fields.items():
            metadata = field_info.metadata
            if not metadata:
                raise MeasurableConfigurationError(
                    f"Field '{field_name}' of '{self.__class__.__name__}' is missing a unit annotation."
                )
            unit = metadata[0]
            if isinstance(unit, str):
                try:
                    unit = Unit(unit, parse_strict = "raise")
                except ValueError as parsing_error:
                    raise MeasurableConfigurationError(
                        f"Error parsing unit annotation for measurable '{self.__class__.__name__}"
                        f": {parsing_error}"
                    )
            elif not isinstance(unit, UnitBase):
                raise MeasurableConfigurationError(
                    f"Field '{field_name}' of '{self.__class__.__name__}' must be annotated with a Unit in the first metadata slot."
                )
            self._tag_unit_info[field_name] = unit
        return self
    
    @cached_property
    def tag_unit_info(self) -> dict[str, UnitBase]:
        return self._tag_unit_info
    
    @cached_property
    def tag_list(self) -> list[str]:
        return list(self._index_map.keys())

    def to_array(self) -> NDArray:
        # the following combination, from testing, gave the best times
        # use np.zeros(...) to remake array each time
        #   instead of np.empty(...) and alike
        #   using a preallocated array and updating it
        #   was no faster and slowed things down due to
        #   requirement of additional checking logics. 
        # use dictionary for array indexing instead of 
        #   an enum. 
        array = np.zeros(self._array_size, dtype=float)
        for attr_name, slice in self._index_map.items(): 
            array[slice] = getattr(self, attr_name)
        return array

    def update_from_array(self, array: NDArray) -> None:
        """updates the class in place using the provided array."""
        for field_name, field_index in self._index_map.items():
            if field_index.start == field_index.stop - 1:
                setattr(self, field_name, array[field_index][0])
                continue
            setattr(self, field_name, array[field_index])

    
class ControlElements(MeasurableBase):
    """
    Base container class for control elements.
    Directly interacts with the States, but is externally controlled.
    Subclasses should define control variables as fields.
    Example ControlElements subclass:
    ```
    class ExampleControlElements(ControlElements):
        inlet_flow: Annotated[float, "L/s"]                          # annotated using with str
        inlet_temperature: Annotated[float, astropy.units.Unit("K")] # annotated using with astropy.units.Unit. Both valid.
    ```
    """

class States(MeasurableBase):
    """
    Base container class for differential state variables in a simulation.
    Example States subclass:
    ```
    class ExampleStates(States):
        concentration: Annotated[float, "mol/L"]               # annotated using with str
        temperature: Annotated[float, astropy.units.Unit("K")] # annotated using with astropy.units.Unit. Both valid.
    ```
    """

class AlgebraicStates(MeasurableBase):

    """
    Base container class for quantities that are algebraic functions of states. 
    CalculationBase is defined in the system. It may be calculated from any method
    that does not involve integration.
    Example AlgebraicStates subclass:
    ```
    class ExampleAlgebraicStates(AlgebraicStates):
        reaction_rate: Annotated[float, "mol/s"]               # annotated using with str
        density: Annotated[float, astropy.units.Unit("g/L")]   # annotated using with astropy.units.Unit. Both valid.
    ```
    """

class Constants(MeasurableBase):
    """
    Base container class for constants in a simulation.
    Example Constants subclass:
    ```
    class ExampleConstants(Constants):
        rate_constant: Annotated[float, "1/s"]                 # annotated using with str
        volume: Annotated[float, astropy.units.Unit("m3")]     # annotated using with astropy.units.Unit as cubic meter
        area: Annotated[float, astropy.units.Unit("m")**2]     # base units raised to a power can be done like this as well
    ```
    """
    

