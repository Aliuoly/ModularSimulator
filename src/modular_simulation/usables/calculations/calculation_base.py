from abc import ABC, abstractmethod
from typing import Annotated, TypeAlias
from numpy.typing import NDArray
from collections.abc import Callable
from enum import IntEnum
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, PlainSerializer, BeforeValidator
from astropy.units import UnitBase, Unit
from modular_simulation.validation.exceptions import CalculationDefinitionError, CalculationConfigurationError
from modular_simulation.usables.tag_info import TagInfo, TagData
from modular_simulation.utils.typing import StateValue, TimeValue
from modular_simulation.utils import extract_unique_metadata

    
class TagType(IntEnum):
    INPUT = 1
    OUTPUT = 2
    CONSTANT = 3



class TagMetadata(BaseModel):
    """
    Represents information about a model state, including its type, unit, and description

    :var type: The type of the tag (e.g., input, output, constant)
    :vartype type: TagType
    :var unit: The unit associated with the tag's value. Defaults to unitless ("").
    :vartype unit: SerializableUnit
    :var description: A brief description of the tag. Use this rather than inline comment where applicable.
                        Defaults to empty string ("").
    :vartype description: str = ""
    """
    type: TagType
    unit: Annotated[
        UnitBase,
        BeforeValidator(lambda u: u if isinstance(u, UnitBase) else Unit(u)),
        PlainSerializer(lambda u: str(u)),
    ] = ""
    description: str = ""
    model_config = ConfigDict(extra='allow',arbitrary_types_allowed=True)


TagAnnotation: TypeAlias = Annotated[str, TagMetadata]
ConstantAnnotation: TypeAlias = Annotated[StateValue, TagMetadata]

# Default tag aliases used throughout the examples.  Users can always opt to
# specify their own :class:`Annotated` types with custom units and descriptions
# when more detail is required.

class CalculationBase(BaseModel, ABC):
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

    #-----construction time defined-----
    _tag_metadata_dict: dict[str, TagMetadata] = PrivateAttr()
    _output_tag_info_dict: dict[str, TagInfo] = PrivateAttr()

    #-----initialization (wiring) time defined-----
    _t: TimeValue = PrivateAttr() 
    _input_data_getters: dict[str, Callable[[], TagData]] = PrivateAttr()
    _input_data_dict: dict[str, TagData] = PrivateAttr()
    _input_value_dict: dict[str, StateValue] = PrivateAttr()
    _initialized: bool = PrivateAttr(default = False)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, context):
        if self.name is None:
            self.name = self.__class__.__name__

        self._tag_metadata_dict = {
            field_name: extract_unique_metadata(field, TagInfo, field_name, CalculationDefinitionError)
            for field_name, field in self.__class__.model_fields.items() 
            if field_name != "name"
        }
        self._output_tag_info_dict = {
            tag: TagInfo(tag = tag, unit = metadata.unit, description = metadata.description)
            for tag, metadata in self._tag_metadata_dict.items()
            if metadata.type == TagType.OUTPUT
        }

    
    def wire_inputs(self, t: TimeValue, available_tag_info_dict: dict[str, TagInfo]) -> None:
        """
        Links calculation inputs to tag info instances
        and creates a simple callable for it.
        Also calls the .calculate method once to initialize the results 
        so they are not NANs.
        Validation is already done so no error handling is placed here. 
        """
        input_tag_metadata_dict = {
            tag: metadata for tag, metadata in self._tag_metadata_dict.items()
            if metadata.type == TagType.INPUT
        }
        for input_field, input_tag_metadata in input_tag_metadata_dict.items():
            input_tag = getattr(self, input_field)
            found_tag_info = available_tag_info_dict.get(input_tag)
            if found_tag_info is None:
                raise CalculationConfigurationError(
                    f"'{self.name}' calculation's input field {input_field} = '{input_tag}' "
                    "was not found amongst the available tags. Double check the tag spelling "
                    "and that it corresponds to either a sensor measurement or a calculation output."
                )
            self._input_data_getters[input_tag] = found_tag_info.make_converted_data_getter(
                target_unit = input_tag_metadata.unit
            )
        self._t = t
        self._initialized = True
        self.calculate(t = t)
        
    def _update_input_triplets(self) -> None:
        tag_data_dict = self._input_data_dict
        for tag_name, tag_data_getter in self._input_data_getters.items():
            tag_data_dict[tag_name] = tag_data_getter()

    def _update_input_values(self) -> dict[str, float | NDArray]:
        value_dict = self._input_value_dict
        for tag_name, tag_data in self._input_data_dict.items():
            value_dict[tag_name] = tag_data.value
        return self._input_value_dict

    @property
    def ok(self) -> bool:
        """
        whether or not the calculation quality is ok. If any of the inputs are not ok,  
            the calculationis also not ok.
        """
        possible_faulty_inputs_oks = [input_value.ok for input_value in self._input_data_dict.values()]
        return any(possible_faulty_inputs_oks) if possible_faulty_inputs_oks else True

    @abstractmethod
    def _calculation_algorithm(self, t: TimeValue, inputs_dict: dict[str, StateValue]) -> dict[str, StateValue]:
        pass    

    def calculate(self, t: TimeValue) -> TagData:
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
            inputs_dict=self._input_value_dict,
        )
        
        for output_tag, output_value in outputs_dict.items():
            tag_data = TagData(time = t, value = output_value, ok = self.ok)
            self._output_tag_info_dict[output_tag].data = tag_data

    @property
    def outputs(self) -> dict[str, TagData]:
        return {tag: data for tag, data in self._output_tag_info_dict.items()}