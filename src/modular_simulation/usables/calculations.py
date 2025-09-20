from abc import ABC, abstractmethod
from typing import Callable, Dict, List
from functools import lru_cache
import numpy as np
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
        description = "dictionary of constants that are used in this calculation."
    )

    _last_value: TimeValueQualityTriplet | None = PrivateAttr(default=None)
    _input_getters: Dict[str, Callable[[], TimeValueQualityTriplet]] | None = PrivateAttr(default=None)
    _buffer_size: int = PrivateAttr(default=10_000)
    _history_size: int = PrivateAttr(default=0)
    _history_times: np.ndarray | None = PrivateAttr(default=None)
    _history_values: np.ndarray | None = PrivateAttr(default=None)
    _history_ok: np.ndarray | None = PrivateAttr(default=None)
    _record_history: bool = PrivateAttr(default=True)

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
                    self._input_getters[tag] = lambda sensor=sensor: sensor._last_value
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
                    self._input_getters[tag] = lambda calculation=calculation: calculation._last_value
                    found = True
            if not found:
                raise AttributeError(
                    f"no calculation tagged '{tag}' is defined. "
                    f"Available calculations are: {', '.join([c.output_tag for c in calculations])}"
                    )
    @property
    @lru_cache(maxsize=1) # only remember the last call
    def inputs(self) -> Dict[str, TimeValueQualityTriplet | float | NDArray]:
        if self._input_getters is not None:
            return {tag_name: tag_getter() for tag_name, tag_getter in self._input_getters.items()}
        raise RuntimeError(
            "Calculation is not initialized. Make sure you used the create_system function to define your system. "
        )

    @property
    def ok(self) -> bool:
        """
        whether or not the calculation quality is ok. If any of the inputs are not ok,  
            the calculationis also not ok.
        """
        possible_faulty_inputs_oks = [
            input_value.ok
            for input_value in self.inputs.values()
            if isinstance(input_value, TimeValueQualityTriplet)
        ]
        return any(possible_faulty_inputs_oks) if possible_faulty_inputs_oks else True

    @abstractmethod
    def _calculation_algorithm(
        self, 
        t: float, 
        inputs_dict: Dict[str, float | NDArray]
        ) -> TimeValueQualityTriplet:
        pass    

    def calculate(self, t: float) -> TimeValueQualityTriplet:
        """public facing method to get the calculation result"""
        result = self._calculation_algorithm(
            t=t,
            inputs_dict=self.inputs,
        )
        result.ok = self.ok
        self._last_value = result
        self._record_result(result)
        return result

    def history(self) -> Dict[str, np.ndarray]:
        if (
            not self._record_history
            or self._history_times is None
            or self._history_values is None
            or self._history_ok is None
            or self._history_size == 0
        ):
            return {
                "time": np.asarray([], dtype=float),
                "value": np.asarray([], dtype=float),
                "ok": np.asarray([], dtype=bool),
            }

        times = self._history_times[: self._history_size].copy()
        values = self._history_values[: self._history_size].copy()
        ok = self._history_ok[: self._history_size].copy()
        return {"time": times, "value": values, "ok": ok}

    def set_history_enabled(self, enabled: bool) -> None:
        self._record_history = enabled
        if not enabled:
            self._history_times = None
            self._history_values = None
            self._history_ok = None
            self._history_size = 0

    def _record_result(self, result: TimeValueQualityTriplet) -> None:
        if not self._record_history:
            return

        value_array = np.asarray(result.value)

        if self._history_times is None:
            self._initialize_history_buffers(value_array)
        elif self._history_size >= self._history_times.shape[0]:
            self._expand_history_buffers()

        if (
            self._history_times is None
            or self._history_values is None
            or self._history_ok is None
        ):
            raise RuntimeError("History buffers must be initialized before storing results.")

        self._history_times[self._history_size] = result.t
        if value_array.shape == ():
            self._history_values[self._history_size] = value_array.item()
        else:
            self._history_values[self._history_size] = value_array
        self._history_ok[self._history_size] = result.ok
        self._history_size += 1

    def _initialize_history_buffers(self, value_array: np.ndarray) -> None:
        values_shape = (
            (self._buffer_size,) if value_array.shape == () else (self._buffer_size, *value_array.shape)
        )
        self._history_times = np.empty(self._buffer_size, dtype=float)
        self._history_ok = np.empty(self._buffer_size, dtype=bool)
        self._history_values = np.empty(values_shape, dtype=value_array.dtype)

    def _expand_history_buffers(self) -> None:
        if (
            self._history_times is None
            or self._history_values is None
            or self._history_ok is None
        ):
            raise RuntimeError("History buffers must be initialized before expansion.")

        new_length = self._history_times.shape[0] + self._buffer_size

        expanded_times = np.empty(new_length, dtype=float)
        expanded_times[: self._history_times.shape[0]] = self._history_times
        self._history_times = expanded_times

        expanded_ok = np.empty(new_length, dtype=bool)
        expanded_ok[: self._history_ok.shape[0]] = self._history_ok
        self._history_ok = expanded_ok

        value_shape = (new_length, *self._history_values.shape[1:])
        expanded_values = np.empty(value_shape, dtype=self._history_values.dtype)
        expanded_values[: self._history_values.shape[0]] = self._history_values
        self._history_values = expanded_values

        