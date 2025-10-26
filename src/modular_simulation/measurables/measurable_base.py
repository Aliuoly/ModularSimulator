from pydantic import BaseModel, PrivateAttr, ConfigDict, dataclasses, model_validator
import numpy as np
from numpy.typing import NDArray
from functools import cached_property
from modular_simulation.validation.exceptions import MeasurableConfigurationError
from astropy.units import UnitBase, Unit
from enum import IntEnum

class CategorizedMeasurables:
    def __init__(self, diff_states, alg_states, control_elements, constants):
        self.differential_states = diff_states
        self.alg_states = alg_states
        self.control_elements = control_elements
        self.constants = constants


class MeasurableType(IntEnum):
    DifferentialState = 0 
    AlgebraicState = 1
    ControlElement = 2
    Constant = 3

@dataclasses.dataclass
class MeasurableMetadata:
    type: MeasurableType
    unit: str

def update_map(map, slice_start, field_name, field_value):
    try:
        value = np.asarray(value)
    except Exception as ex:
        raise TypeError(
            f"measurable field '{field_name}' is not coercable into a numpy array. "
            f"coercion exception: {ex}"
        )
    slice_end = slice_start + value.size
    map[field_name] = slice(slice_start, slice_end)
    slice_start = slice_end
    return map, slice_end

class MeasurableBase(BaseModel):
    """
    Base class for the measurable data containers which are
    indexible via an internally constructed Enum. 
    Can be converted to numpy arrays via said indexing, 
    and can be updating from numpy arrays via the same indexing. 
    """
    _diff_state_map: dict[str, slice] = PrivateAttr()
    _diff_array_size: int = PrivateAttr()
    _alg_state_map: dict[str, slice] = PrivateAttr()
    _alg_array_size: int = PrivateAttr()
    _control_element_map: dict[str, slice] = PrivateAttr()
    _control_array_size: int = PrivateAttr()
    _constant_map: dict[str, slice] = PrivateAttr()
    _constant_array_size: int = PrivateAttr()
    _tag_unit_info: dict[str, UnitBase] = PrivateAttr(default_factory = dict)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    
    @model_validator(mode = 'after')
    def _validate_annotation_and_categorize(self):
        diff_slice_start = 0
        alg_slice_start = 0
        ce_slice_start = 0
        k_slice_start = 0
        for field_name, field_info in self.__class__.model_fields.items():
            try:
                metadata = field_info.metadata[0]
            except Exception as ex:
                raise MeasurableConfigurationError(
                    f"Field '{field_name}' in Measurable '{self.__class__.__name__}' is not properly annotated. "
                    "Be sure to use the 'Annotated' type hint with 'TagMetadata' as the metadata. "
                ) from ex
            validated = False
            for metadatum in metadata: # Lol datum
                if isinstance(metadatum, MeasurableMetadata):
                    match metadatum.type:
                        case MeasurableType.DifferentialState:
                            self._diff_state_map, self._diff_array_size = update_map(
                                self._diff_state_map, self._diff_array_size, 
                                field_name, getattr(self, field_name)
                                )
                        case MeasurableType.AlgebraicState:
                            self._alg_state_map, self._alg_array_size = update_map(
                                self._alg_state_map, self._alg_array_size, field_name, getattr(self, field_name)
                                )
                        case MeasurableType.ControlElement:
                            self._control_element_map, self._control_array_size = update_map(
                                self._control_element_map, self._control_array_size, field_name, getattr(self, field_name)
                                )
                        case MeasurableType.Constant:
                            self._constant_map, self._constant_array_size = update_map(
                                self._constant_map, self._constant_array_size, field_name, getattr(self, field_name)
                                )
                        case _:
                            raise MeasurableConfigurationError(
                                f"Measurable type '{metadatum.type}' of '{field_name}' in "
                                f"Measurable '{self.__class__.__name__}' unrecognized"
                            )
                    try:
                        unit = Unit(metadatum.unit)
                        self._tag_unit_info[field_name] = unit
                    except Exception as ex:
                        raise MeasurableConfigurationError(
                            f"Failed to parse unit '{unit}' of '{field_name}' of '{self.__class__.__name__}'."
                        )
                    validated = True
            if not validated:
                raise MeasurableConfigurationError(
                    f"Field '{field_name}' in Measurable '{self.__class__.__name__}' is not properly annotated. "
                    "Be sure to use the 'Annotated' type hint with 'MeasurableMetadata' instance as the metadata. "
                )

    def model_post_init(self, context):
        self._validate_annotation_and_categorize()
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

        

    def _ensure_unit_annotated(self):
        for field_name, field_info in self.__class__.model_fields.items():
            metadata = field_info.metadata
            if not metadata:
                raise MeasurableConfigurationError(
                    f"Field '{field_name}' of '{self.__class__.__name__}' is missing a unit annotation."
                )
            unit = metadata[0]
            
        return self
    
    @cached_property
    def tag_unit_info(self) -> dict[str, UnitBase]:
        return self._tag_unit_info
    
    @cached_property
    def tag_list(self) -> list[str]:
        return list(self._index_map.keys())
    
    @cached_property
    def categorized_tags(self) -> dict[str, list[str]]:
        return CategorizedMeasurables(
            list(self._diff_state_map.keys()),
            list(self._alg_state_map.keys()),
            list(self._control_element_map.keys()),
            list(self._constant_map.keys())
        )

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
    

