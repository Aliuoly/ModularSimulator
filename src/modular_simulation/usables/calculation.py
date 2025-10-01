from abc import ABC, abstractmethod
from typing import Callable, Dict, List, TYPE_CHECKING, Any, Annotated
from numpy.typing import NDArray
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, BeforeValidator
from modular_simulation.usables.time_value_quality_triplet import TimeValueQualityTriplet
if TYPE_CHECKING:
    from modular_simulation.quantities.usable_quantities import UsableQuantities
    
def ensure_list(value: Any) -> Any:  

    if not isinstance(value, list):  
        return [value]
    else:
        return value
    
class Calculation(BaseModel, ABC):
    #TODO: change it such that fields are defined for inputs rather than
    #    relying on a list of inputs. 
    """
    constants are to be defined as subclass attributes so as to be accessible by attribute 
    lookup in the _calculation_algorithm implementation. 
    tags are expected to match exactly in the calculation algorithm and in the input tags list. 
    """
    output_tags: Annotated[List[str], BeforeValidator(ensure_list)] = Field(
        ...,
        description = "tags of the calculation's output. Must be unique."
    )
    measured_input_tags: List[str] = Field(
        default_factory = list,
        description = "list of tags corresponding to measurements that are inputs to this calculation."
    )
    calculated_input_tags: List[str] = Field(
        default_factory = list,
        description = "list of tags corresponding to calculations that are inputs to this calculation."
    )
    name: str | None = Field(
        default = None,
        description = "Name of the calculation - optional."
    )

    _last_results: Dict[str, TimeValueQualityTriplet] = PrivateAttr()
    _input_getters: Dict[str, Callable[[], TimeValueQualityTriplet]] = PrivateAttr(default_factory=dict)
    _last_input_triplet_dict: Dict[str, TimeValueQualityTriplet] = PrivateAttr(default_factory=dict)
    _last_input_value_dict: Dict[str, float | NDArray] = PrivateAttr(default_factory=dict)
    _history: Dict[str, List[TimeValueQualityTriplet]] = PrivateAttr(default_factory = dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, context:Any):
        if self.name is None:
            self.name = self.__class__.__name__
    
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
        for tag in self.measured_input_tags:
            for sensor in sensors:
                if sensor.alias_tag == tag: # use the alias tag in case it is different from the raw measurement tag
                    self._input_getters[tag] = lambda sensor=sensor: sensor._last_value #type: ignore

        # 2. look in calculations for calculated tags
        for tag in self.calculated_input_tags:
            for calculation in calculations:
                for output_tag in calculation.output_tags:
                    if output_tag == tag:
                        self._input_getters[tag] = (
                            lambda calculation=calculation, output_tag=output_tag: # type: ignore
                                calculation._last_results[output_tag]              
                        )
        for tag in self.output_tags:
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
        ) -> float | NDArray:
        pass    

    def calculate(self, t: float) -> TimeValueQualityTriplet:
        """public facing method to get the calculation result"""
        self._update_input_triplets()
        self._update_input_values()
        results = self._calculation_algorithm(
            t=t,
            inputs_dict=self._last_input_value_dict,
        )
        result_triplets = {}
        
        for i, result in enumerate(np.atleast_1d(results)):
            tag = self.output_tags[i]
            triplet = TimeValueQualityTriplet(t = t, value = result, ok = self.ok)
            result_triplets[tag] = triplet
            self._history[tag].append(triplet)
        self._last_results = result_triplets
        return result_triplets

    @property
    def history(self) -> Dict[str, List[TimeValueQualityTriplet]]:
        return self._history.copy()