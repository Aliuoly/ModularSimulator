from abc import ABC, abstractmethod
from typing import Callable, Dict, List, TYPE_CHECKING, Any, Tuple, NewType, get_type_hints, Union
from numpy.typing import NDArray
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, BeforeValidator, model_validator
from modular_simulation.usables.time_value_quality_triplet import TimeValueQualityTriplet
from modular_simulation.validation.exceptions import CalculationConfigurationError
if TYPE_CHECKING:
    from modular_simulation.quantities.usable_quantities import UsableQuantities
    from modular_simulation.usables.sensor import Sensor

OutputTag = NewType("OutputTag",str)
MeasuredTag = NewType("MeasuredTag", str)
CalculatedTag = NewType("CalculatedTag", str)
ScalarConstant = NewType("ScalarConstant", float)
ArrayConstant = NewType("ArrayConstant", NDArray)
Constant = Union[ScalarConstant, ArrayConstant]

def ensure_list(value: Any) -> Any:  
    if not isinstance(value, list):  
        return [value]
    else:
        return value
def make_measured_input_getter(sensor: "Sensor") -> Callable[[], TimeValueQualityTriplet]:
    def input_getter() -> TimeValueQualityTriplet:
        return sensor._last_value
    return input_getter
def make_calculated_input_getter(calculation: "Calculation", tag: str) -> Callable[[], TimeValueQualityTriplet]:
    def input_getter() -> TimeValueQualityTriplet:
        return calculation._last_results[tag]
    return input_getter
    
class Calculation(ABC, BaseModel):
    #TODO: change it such that fields are defined for inputs rather than
    #    relying on a list of inputs. 
    """
    constants are to be defined as subclass attributes so as to be accessible by attribute 
    lookup in the _calculation_algorithm implementation. 
    tags are expected to match exactly in the calculation algorithm and in the input tags list. 
    """
    name: str | None = Field(
        default = None,
        description = "Name of the calculation - optional."
    )
    _last_results: Dict[str, TimeValueQualityTriplet] = PrivateAttr(default_factory = dict)
    _input_getters: Dict[str, Callable[[], TimeValueQualityTriplet]] = PrivateAttr(default_factory=dict)
    _last_input_triplet_dict: Dict[str, TimeValueQualityTriplet] = PrivateAttr(default_factory=dict)
    _last_input_value_dict: Dict[str, float | NDArray] = PrivateAttr(default_factory=dict)
    _history: Dict[str, List[TimeValueQualityTriplet]] = PrivateAttr(default_factory = dict)
    _output_tags: Tuple[str] = PrivateAttr()
    _measured_input_tags: Tuple[str] = PrivateAttr()
    _calculated_input_tags: Tuple[str] = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode = "after")
    def validate_input_fields_and_make_tuples(self):
        output_tags = []
        measured_input_tags = []
        calculated_input_tags = []

        allowed_types = (OutputTag, MeasuredTag, CalculatedTag, Constant)
        invalid_field_names = []
            # Iterate over the model's defined fields to access their annotations
        for field_name, field_info in self.__class__.model_fields.items():
        # Skip fields that don't need this specific validation
            if field_name == "name":
                continue
            # field_info.annotation holds the actual type hint (e.g., OutputTag)
            annotation = field_info.annotation
            if  annotation not in allowed_types:
                invalid_field_names.append(field_name)
            else:
                tag_name = getattr(self, field_name)
                if annotation == OutputTag:
                    output_tags.append(tag_name)
                elif annotation == MeasuredTag:
                    measured_input_tags.append(tag_name)
                elif annotation == CalculatedTag:
                    calculated_input_tags.append(tag_name)
                    
        if len(invalid_field_names) > 0:
            raise CalculationConfigurationError(
                f"The following fields in calculation '{self.__class__.__name__}' is not a constant, "
                f"measured input, calculated input, or output tag: {', '.join(invalid_field_names)}. \n"
                "When defining the calculation subclass, be sure to wrap the inputs in either 'MeasuredTag', "
                "'CalculatedTag', 'OutputTag' or 'Constant', which can be imported with \n" 
                "from modular_simulation.usables import MeasuredTag, CalculatedTag, OutputTag, Constant"
            )
        
        self._output_tags = output_tags
        self._measured_input_tags = measured_input_tags
        self._calculated_input_tags = calculated_input_tags

        return self

    @property
    def calculation_name(self):
        if self.name is None:
            return self.__class__.__name__
        return self.name
    
    def _initialize(
            self,
            usable_quantities: "UsableQuantities"
            ) -> None:
        """
        generates the input getting functions and save them as a dictionary of callables.
        Called once during system instantiation. Refers to the Singleton instance of
        sensors that are defined also at system instantiation. 
        By now, usable quantities class has already validated tags do exist, so we just look for it. 
        """
        sensors = usable_quantities.sensors
        calculations = usable_quantities.calculations
        self._input_getters = {}
        # 1. look in sensors for measured tags
        for input_tag in self._measured_input_tags:
            for sensor in sensors:
                if sensor.alias_tag == input_tag: # use the alias tag in case it is different from the raw measurement tag
                    self._input_getters[input_tag] = make_measured_input_getter(sensor)

        # 2. look in calculations for calculated tags
        for input_tag in self._calculated_input_tags:
            for calculation in calculations:
                for output_tag in calculation._output_tags:
                    if output_tag == input_tag:
                        self._input_getters[input_tag] = make_calculated_input_getter(calculation, input_tag)
        for tag in self._output_tags:
            self._history[tag] = []
            
    def _update_input_triplets(self) -> None:
        triplet_dict = self._last_input_triplet_dict
        for tag_name, tag_getter in self._input_getters.items():
            triplet_dict[tag_name] = tag_getter()
        if len(triplet_dict) == 0:
            raise RuntimeError(
                "Calculation is not initialized. Make sure you used the create_system function to define your system. "
            )
    

    def _update_input_values(self) -> Dict[str, float | NDArray]:
        value_dict = self._last_input_value_dict
        for tag_name, triplet in self._last_input_triplet_dict.items():
            value_dict[tag_name] = triplet.value
        return self._last_input_value_dict

    @property
    def ok(self) -> bool:
        """
        whether or not the calculation quality is ok. If any of the inputs are not ok,  
            the calculationis also not ok.
        """
        possible_faulty_inputs_oks = [input_value.ok for input_value in self._last_input_triplet_dict.values()]
        return any(possible_faulty_inputs_oks) if possible_faulty_inputs_oks else True

    @abstractmethod
    def _calculation_algorithm(
        self, 
        t: float, 
        inputs_dict: Dict[str, float | NDArray]
        ) -> Dict[str, float | NDArray]:
        pass    

    def calculate(self, t: float) -> TimeValueQualityTriplet:
        """public facing method to get the calculation result"""
        self._update_input_triplets()
        self._update_input_values()
        outputs_dict = self._calculation_algorithm(
            t=t,
            inputs_dict=self._last_input_value_dict,
        )
        result_triplets = {}
        
        for output_tag, output_value in outputs_dict.items():
            triplet = TimeValueQualityTriplet(t = t, value = output_value, ok = self.ok)
            result_triplets[output_tag] = triplet
            self._history[output_tag].append(triplet)
        self._last_results = result_triplets
        return result_triplets

    @property
    def history(self) -> Dict[str, List[TimeValueQualityTriplet]]:
        return self._history.copy()