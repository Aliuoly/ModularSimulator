from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Any, Annotated, TypeAlias
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from modular_simulation.validation.exceptions import CalculationDefinitionError
from modular_simulation.usables.tag_info import TagInfo, TagData
from astropy.units import UnitBase #type: ignore
from enum import IntEnum

def ensure_list(value: Any) -> Any:  
    if not isinstance(value, list):  
        return [value]
    else:
        return value


def make_converted_input_getter(raw_tag_info: TagInfo, desired_tag_info: TagInfo):
    if raw_tag_info.unit.is_equivalent(desired_tag_info.unit):
        converter = raw_tag_info.unit.get_converter(desired_tag_info.unit)

        def input_getter() -> TagData:
            return TagData(
                raw_tag_info.data.time,
                converter(raw_tag_info.data.value),
                raw_tag_info.data.ok,
            )

        return input_getter
    raise CalculationDefinitionError(
        f"Tried to convert tag '{raw_tag_info.tag}' from '{raw_tag_info.unit}' to '{desired_tag_info.unit}' and failed. "
        "Make sure these units are compatible. "
    )
    
class TagType(IntEnum):
    INPUT = 1
    OUTPUT = 2
    CONSTANT = 3

class TagMetadata:
    def __init__(self, type: TagType, unit: UnitBase, description: str | None = None):
        self.unit = unit
        self.description = description
        self.type = type

TagAnnotation: TypeAlias = Annotated[str, TagMetadata]
ConstantAnnotation: TypeAlias = Annotated[float | NDArray, TagMetadata]

# Default tag aliases used throughout the examples.  Users can always opt to
# specify their own :class:`Annotated` types with custom units and descriptions
# when more detail is required.

class Calculation(ABC, BaseModel):
    """
    inputs and outputs tag names are expected to be annotated with the following info
    1. unit
    2. description - optional
    3. input/output/constant type
    e.g., input_one_tag: Annotated[str, TagAnnotation(TagType.INPUT), Unit('m'), 'this is input one']
    """
    name: str | None = Field(
        default = None,
        description = "Name of the calculation - optional."
    )
    _last_results: Dict[str, TagData] = PrivateAttr(default_factory = dict)
    _input_data_getters: Dict[str, Callable[[], TagData]] = PrivateAttr(default_factory=dict)
    _last_input_data_dict: Dict[str, TagData] = PrivateAttr(default_factory=dict)
    _last_input_value_dict: Dict[str, float | NDArray] = PrivateAttr(default_factory=dict)
    _output_tag_info_dict: Dict[str, TagInfo] = PrivateAttr(default_factory=dict)
    _input_tag_info_dict: Dict[str, TagInfo] = PrivateAttr(default_factory=dict)
    _initialized: bool = PrivateAttr(default = False)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


    @model_validator(mode = 'after')
    def _categorize_fields(self):
        for field_name, field_info in self.__class__.model_fields.items():
            if field_name == 'name':
                continue
            try:
                metadata = field_info.metadata[0]
            except Exception as ex:
                raise CalculationDefinitionError(
                    f"Field '{field_name}' in calculation '{self.__class__.__name__}' is not properly annotated. "
                    "Be sure to use the 'Annotated' type hint with 'TagMetadata' as the metadata. "
                ) from ex
            
            if isinstance(metadata, TagMetadata):
                tag_info = TagInfo(
                    tag = getattr(self, field_name),
                    unit = metadata.unit,
                    description = metadata.description
                )
                if metadata.type == TagType.OUTPUT:
                    self._output_tag_info_dict[tag_info.tag] = tag_info
                elif metadata.type == TagType.INPUT:
                    self._input_tag_info_dict[tag_info.tag] = tag_info
                elif metadata.type == TagType.CONSTANT:
                    pass # constants don't need to be stored in a special way
                else:
                    raise CalculationDefinitionError(
                        f"Field '{field_name}' in calculation '{self.__class__.__name__}' has an invalid TagType."
                    )
            else:
                raise CalculationDefinitionError(
                    f"Field '{field_name}' in calculation '{self.__class__.__name__}' is missing TagMetadata annotation."
                )
        return self
    
    @property
    def calculation_name(self):
        if self.name is None:
            return self.__class__.__name__
        return self.name
    
    def _initialize(
            self,
            tag_infos: List[TagInfo],
            ) -> None:
        """
        Links calculation inputs to tag info instances
        and creates a simple callable for it.
        Also calls the .calculate method once to initialize the results 
        so they are not NANs.
        Validation is already done so no error handling is placed here. 
        """
        for input_tag, input_tag_info in self._input_tag_info_dict.items():
            for tag_info in tag_infos:
                if tag_info.tag == input_tag: 
                    self._input_data_getters[input_tag] = make_converted_input_getter(
                        raw_tag_info = tag_info, 
                        desired_tag_info = input_tag_info
                    )
                    break
        self._initialized = True
        self.calculate(t = 0)
        
    def _update_input_triplets(self) -> None:
        tag_data_dict = self._last_input_data_dict
        for tag_name, tag_data_getter in self._input_data_getters.items():
            tag_data_dict[tag_name] = tag_data_getter()

    def _update_input_values(self) -> Dict[str, float | NDArray]:
        value_dict = self._last_input_value_dict
        for tag_name, tag_data in self._last_input_data_dict.items():
            value_dict[tag_name] = tag_data.value
        return self._last_input_value_dict

    @property
    def ok(self) -> bool:
        """
        whether or not the calculation quality is ok. If any of the inputs are not ok,  
            the calculationis also not ok.
        """
        possible_faulty_inputs_oks = [input_value.ok for input_value in self._last_input_data_dict.values()]
        return any(possible_faulty_inputs_oks) if possible_faulty_inputs_oks else True

    @abstractmethod
    def _calculation_algorithm(
        self, 
        t: float, 
        inputs_dict: Dict[str, float | NDArray]
        ) -> Dict[str, float | NDArray]:
        pass    

    def calculate(self, t: float) -> TagData:
        """public facing method to get the calculation result"""
        if not self._initialized:
            raise RuntimeError(
                "Tried to call 'calculate' before the system orchestrated the various quantities. "
                "Make sure this calculation is part of a system and the system has been constructed."
            )
        self._update_input_triplets()
        self._update_input_values()
        outputs_dict = self._calculation_algorithm(
            t=t,
            inputs_dict=self._last_input_value_dict,
        )
        
        for output_tag, output_value in outputs_dict.items():
            tag_data = TagData(time = t, value = output_value, ok = self.ok)
            self._output_tag_info_dict[output_tag].data = tag_data