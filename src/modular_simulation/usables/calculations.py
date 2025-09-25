from abc import ABC, abstractmethod
from typing import Callable, Dict, List
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from modular_simulation.usables.sensors.sensor import Sensor
from modular_simulation.usables.time_value_quality_triplet import TimeValueQualityTriplet
    



class Calculation(BaseModel, ABC):

    output_tag: str = Field(
        ...,
        description = "tag of the calculation's output. Must be unique."
    )
    measured_input_tags: List[str] = Field(
        ...,
        description = "list of tags corresponding to measurements that are inputs to this calculation."
    )
    calculated_input_tags: List[str] = Field(
        ...,
        description = "list of tags corresponding to calculations that are inputs to this calculation."
    )
    constants: Dict[str, float] = Field(
        ...,
        description = (
            "dictionary of constants that are used in this calculation. Do not get this confused with "
            "system constants defined in measurable_quantities - those are true system constants, while "
            "these are 'known' system constants to be used in user calculations, which may or may not be system-accurate."
        )
    )

    _last_value: TimeValueQualityTriplet = PrivateAttr()
    _input_getters: Dict[str, Callable[[], TimeValueQualityTriplet]] | None = PrivateAttr(default=None)
    _last_input_triplet_dict: Dict[str, TimeValueQualityTriplet] = PrivateAttr(default_factory=dict)
    _last_input_value_dict: Dict[str, float | NDArray] = PrivateAttr(default_factory=dict)
    _history: List[TimeValueQualityTriplet] = PrivateAttr(default_factory = list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def initialized(self) -> bool: return (self._input_getters is None)

    def _initialize(
            self,
            sensors: List["Sensor"],
            calculations: List["Calculation"],
            ) -> None:
        """
        generates the input getting functions and save them as a dictionary of callables.
        Called once during system instantiation. Refers to the Singleton instance of
        sensors that are defined also at system instantiation.
        """
        self._input_getters = {}
        # 1. look in sensors for measured tags
        for tag in self.measured_input_tags:
            found = False
            for sensor in sensors:
                if sensor.measurement_tag == tag:
                    self._input_getters[tag] = lambda sensor=sensor: sensor._last_value #type: ignore
                    found = True
            if not found:
                raise AttributeError(
                    f"no measurement tagged '{tag}' is defined. "
                    f"Available measurements are: {', '.join([s.measurement_tag for s in sensors])}"
                    )
            
        # 2. look in calculations for calculated tags
        for tag in self.calculated_input_tags:
            found = False
            for calculation in calculations:
                if calculation.output_tag == tag:
                    self._input_getters[tag] = lambda calculation=calculation: calculation._last_value #type: ignore
                    found = True
            if not found:
                raise AttributeError(
                    f"no calculation tagged '{tag}' is defined. "
                    f"Available calculations are: {', '.join([c.output_tag for c in calculations])}"
                    )
    
    def _update_input_triplets(self) -> None:
        triplet_dict = self._last_input_triplet_dict
        if self._input_getters is not None:
            for tag_name, tag_getter in self._input_getters.items():
                triplet_dict[tag_name] = tag_getter()
        else:
            raise RuntimeError(
                "Calculation is not initialized. Make sure you used the create_system function to define your system. "
            )
    

    def _update_input_values(self) -> Dict[str, float | NDArray]:
        value_dict = self._last_input_value_dict
        for tag_name, triplet in self._last_input_triplet_dict.items():
            value_dict[tag_name] = triplet.value
        for constant_name, constant in self.constants.items():
            value_dict[constant_name] = constant
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
        result = self._calculation_algorithm(
            t=t,
            inputs_dict=self._last_input_value_dict,
        )
        result_triplet = TimeValueQualityTriplet(t = t, value = result, ok = self.ok)
        self._last_value = result_triplet
        self._history.append(result_triplet)
        return result_triplet

    def history(self) -> List[TimeValueQualityTriplet]:
        return self._history.copy()