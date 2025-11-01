import numpy as np
from abc import ABC, abstractmethod
from pydantic import (
    BaseModel, ConfigDict,
    Field, PrivateAttr, 
    PlainSerializer, BeforeValidator
)
from astropy.units import Unit, UnitBase
from typing import Any, Annotated
from collections.abc import Callable
from modular_simulation.usables.tag_info import TagData, TagInfo
from modular_simulation.measurables.process_model import ProcessModel
from modular_simulation.utils.typing import StateValue, TimeValue
from modular_simulation.validation.exceptions import SensorConfigurationError



class SensorBase(BaseModel, ABC):
    """Abstract base class for all sensors in the modular simulation framework.

    Subclasses implement the scheduling and signal conditioning steps while
    the base class wires the sensor into the orchestrated
    :class:`~modular_simulation.measurables.base_classes.MeasurableQuantities`
    instance.  Common responsibilities such as unit conversion, alias tag
    management, clipping to instrument ranges, first-order dynamics, and the
    injection of noise/fault models are handled here.  The latest
    :class:`~modular_simulation.usables.tag_info.TagData` sample is stored on
    the sensor's :class:`~modular_simulation.usables.tag_info.TagInfo`, which
    also historizes every emitted measurement for later analysis.
    """
    measurement_tag: str = Field(
        ...,
        description = "tag of the state or control element to measure."
    )
    alias_tag: str | None = Field(
        default = None,
        description = (
            "alias of the measurement to be used when the sensor returns results."
            "e.g., if measurement_tag was 'cumm_MI' and alias tag was 'lab_MI', "
            "then, a usable with tag 'lab_MI' will be available, while 'cumm_MI' would not be available."
        )
    )
    unit: Annotated[
        UnitBase,
        BeforeValidator(lambda u: u if isinstance(u, UnitBase) else Unit(u)),
        PlainSerializer(lambda u: str(u)),
    ] = Field(
        description = "Unit of the measured quantity. Will be parsed with Astropy if is a string. "
    )
    description: str = Field(
        default = "No description provided.",
        description = "Description of the sensor's measurement."
    )
    coefficient_of_variance: float = Field(
        default = 0.0,
        description = (
            "the standard deviation of the measurement noise, defined as "
            "a fraction of the true value of measurement."
        )
    )
    faulty_probability: float = Field(
        default = 0.0, 
        ge=0.0, lt=1.0,
        description = "probability for the sensor to err (stall or spike)"
    )
    faulty_aware: bool = Field(
        default = False,
        description = (
            "whether or not the sensor knows it has erred. "
            "if True, the .ok field of the measurement result will be set automatically. "
            "if False, the .ok field of the measurement will not be set by the sensor routine. "
        )
    )
    instrument_range: tuple[StateValue, StateValue] = Field(
        default_factory = lambda : (-float('inf'), float('inf')),
        description = (
            "the range of value this sensor can return. "
            "If true value is beyong the range, it is clipped and returned. "
        )
    )
    time_constant: float = Field(
        default = 0.0,
        description = (
            "The timeconstant associated with the dynamics of "
            "this sensor. The final measurement is then filtered "
            "based on this time constant to mimic sensor dynamics. "
        )
    )
    random_seed: int = Field(
        default = 0,
        description = "Random seed for signal noise injection"
    )

    #----construction time initialized----
    _rng: np.random.Generator = PrivateAttr()
    _tag_info: TagInfo = PrivateAttr()

    #----commission time initialized----
    _measurement_getter: Callable[[], StateValue] = PrivateAttr()
    # all internal calculation that requires the previous measurement and timestamp should use these
    # these may or may not be equivalent to _tag_info.data.value and .t
    _measurement: StateValue = PrivateAttr() 
    _t: TimeValue = PrivateAttr()                # all internal calculation that requires the previous timestamp should use this
    _initialized: bool = PrivateAttr(default = False)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def model_post_init(self, context: Any) -> None:
        self._rng = np.random.default_rng(self.random_seed)
        self._tag_info = TagInfo(
            tag = self.alias_tag, 
            unit = self.unit, 
            description = self.description,
            _raw_tag = self.measurement_tag)
        if self.alias_tag is None: 
            self.alias_tag = self.measurement_tag

    def commission(self, t: TimeValue, process: ProcessModel) -> None:
        """
        Commission the sensor for the process (lol)
        Resolve the measurement source and prime the sensor state.
        Note that all necessary sensor states (last measurements, times, etc.)
        are initialized here as well, so no subsequent operations
        will check for "if self.something is None"
        """
        desired_state = self.measurement_tag
        state_info = process.state_metadata_dict.get(desired_state)
        if state_info is None:
            raise SensorConfigurationError(
                f"'{desired_state}' sensor's desired measurement doesn't exist in the process model."
            )
        if not self.unit.is_equivalent(state_info.unit):
            raise SensorConfigurationError(
                f"'{desired_state}' sensor's desired unit '{self.unit}' is not compatible "
                f"with the measured state's unit '{state_info.unit}'."
            )

        self._measurement_getter = process.make_converted_getter(desired_state, self.unit)
        self._t = t
        self._measurement = self._measurement_getter()
        self._initialized = True
        self.measure(t = self._t, force=True) # flush the update logic
    
    def measure(self, t: TimeValue, *, force: bool = False) -> TagData:
        """Execute the measurement pipeline and return the resulting :class:`TagData`.

        The method enforces initialization, optionally reuses the most recent
        measurement when no refresh is required, gathers the raw value from the
        linked model, applies subclass-defined processing, and then runs the
        shared noise/fault machinery before historizing the result.
        """
        if not self._initialized:
            raise RuntimeError(
                "Tried to call 'measure' before the system orchestrated the various quantities. "
                "Make sure this sensor is part of a system and the system has been constructed."
            )
        if not self._should_update(t) and not force:
            return self._tag_info.data
        
        raw_value = self._measurement_getter()
        if self.time_constant > 1e-12:
            raw_value = self._sensor_dynamics(raw_value, t)
        processed_value, successful = np.clip(
            self._get_processed_value(t = t, raw_value = raw_value), 
            *self.instrument_range
        )
        if not successful:
            return self._tag_info.data
        
        final_value, is_faulty = self._apply_noise_and_faults(processed_value)

        ok = True if not self.faulty_aware else (not is_faulty)
        self._tag_info.data = TagData(time = t, value = final_value, ok = ok)
        return self._tag_info.data
    
    def _sensor_dynamics(self, new_value: StateValue, t: TimeValue):
        dt = t - self._t
        lamb = dt / (dt + self.time_constant) # approximation
        self._t = t
        self._measurement = lamb * new_value + (1-lamb) * self._measurement
        return self._measurement
    
    @abstractmethod
    def _should_update(self, t: TimeValue) -> bool:
        """
        Subclass hook that decides whether a fresh measurement is required.

        Typical implementations enforce sampling periods, latency, or
        dead-band logic.  Returning ``False`` short-circuits ``measure`` and
        reuses the previous :class:`TagData` sample.
        """
        pass

    @abstractmethod
    def _get_processed_value(self, t: TimeValue, raw_value: StateValue) -> tuple[StateValue, bool]:
        """
        Transform the true value into the sensor's processed output.

        Subclasses should perform any latency simulation, biasing, filtering,
        or other transformations here and return a value with the same shape as
        ``raw_value``. 
        Also must return a boolean indicating if processing was succesful.
            True = success, False = failed
        """
        pass

    def _apply_noise_and_faults(self, value: StateValue) -> tuple[StateValue, bool]:
        """
        Apply the configured stochastic noise and random faults.

        A fault either freezes the value at the last good sample or introduces
        a random spike.  When no fault occurs, zero-mean Gaussian noise with a
        standard deviation proportional to the true value is injected.
        Returns the perturbed value and a ``bool`` indicating whether a fault
        occurred.
        """
        # Check for a fault
        faulty = False
        if self._rng.random() < self.faulty_probability:
            faulty = True
            # A fault occurred, simulate what kind
            # 1. frozen value
            if self._rng.random() < 0.5:
                return self._measurement, faulty
            # 2. spike
            value *= (0.5 + self._rng.random())
            return value, faulty

        # If not faulty, apply noise
        noisy_value = value
        if self.coefficient_of_variance > 0:
            noise_std_dev = np.abs(value * self.coefficient_of_variance)
            noise = self._rng.normal(loc=0.0, scale=noise_std_dev)
            noisy_value = value + noise
        
        return noisy_value, False

    @property
    def measurement_history(self) -> list[TagData]:
        return list(self._tag_info.history)
