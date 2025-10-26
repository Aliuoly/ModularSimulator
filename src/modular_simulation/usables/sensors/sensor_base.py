from numpy.typing import NDArray
import numpy as np
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, field_serializer, field_validator
from typing import TYPE_CHECKING, Any
from collections.abc import Callable
from astropy.units import UnitBase, Unit
from modular_simulation.usables.tag_info import TagData, TagInfo

if TYPE_CHECKING:
    from modular_simulation.measurables import MeasurableQuantities, MeasurableBase

def make_measurement_getter(object: "MeasurableBase", tag: str, converter:Callable) -> Callable[[], float|NDArray]:
    def measurement_getter() -> float | NDArray:
        return converter(getattr(object, tag))
    return measurement_getter


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
    unit: str|UnitBase = Field(
        description = "Unit of the measured quantity. Will be parsed with Astropy if is a string. "
    )
    description: str|None = Field(
        default = None,
        description = "Description of the sensor's measurement."
    )
    coefficient_of_variance: float = Field(
        default = 0.0,
        description = (
            "the standard deviation of the measurement noise, defined as "
            "a fraction of the true value of measurement."
        ))
    
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
        ))
    instrument_range: tuple[float,float] = Field(
        default_factory = lambda : (-float('inf'), float('inf')),
        description = (
            "the range of value this sensor can return. "
            "If true value is beyong the range, it is clipped and returned."
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
    random_seed: int = Field(0)

    _rng: np.random.Generator = PrivateAttr()
    _tag_info: TagInfo = PrivateAttr()
    _last_measurement: float|NDArray|None = PrivateAttr(default = None)
    _last_t: float = PrivateAttr(default = 0)
    _initialized: bool = PrivateAttr(default = False)
    _measurement_getter: Callable[[], float | NDArray] = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("unit", mode = 'before')
    @classmethod
    def convert_unit(cls, unit: str|UnitBase) -> UnitBase:
        if isinstance(unit, str):
            return Unit(unit)
        return unit
    
    def model_post_init(self, context: Any) -> None:
        self._rng = np.random.default_rng(self.random_seed)
        if self.alias_tag is None:
            self.alias_tag = self.measurement_tag
        self._tag_info = TagInfo(
            tag = self.alias_tag,
            unit = self.unit,
            description = "" if self.description is None else self.description
        )

    def _initialize(self, measurable_quantities: "MeasurableQuantities") -> None:
        """Resolve the measurement source and prime the sensor state.

        The measurable quantities container has already validated that the
        requested ``measurement_tag`` exists.  This routine simply locates the
        owning model, builds a unit-aware getter callable, and performs an
        initial measurement so ``_tag_info`` contains a populated
        :class:`TagData` sample before the simulation loop starts.
        """
        search_order = list(measurable_quantities.model_dump())
        
        for category in search_order:
            owner = getattr(measurable_quantities, category)
            if owner is not None and hasattr(owner, self.measurement_tag):
                converter = measurable_quantities.tag_unit_info[self.measurement_tag].\
                    get_converter(self._tag_info.unit)
                self._measurement_getter = make_measurement_getter(owner, self.measurement_tag, converter)
                self._initialized = True
                self.measure(t = 0)
                
                return

    # --- Template Method ---
    def measure(self, t: float) -> TagData:
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
        # 1. Decide if a new measurement should be taken
        if not self._should_update(t) and self._last_value is not None:
            return self._last_value

        # 2. Get the true, raw value from the system. 
        raw_value = self._measurement_getter()

        # 3. Apply subclass-specific processing (e.g., time delay) to the true value
        processed_value = np.clip(
            self._get_processed_value(t = t, raw_value = raw_value), 
            *self.instrument_range
            )
        processed_value = self._sensor_dynamics(processed_value, t)
        
        # 4. Apply noise and faults to the processed value
        final_value, is_faulty = self._apply_noise_and_faults(processed_value)

        # 5. Create and store the new measurement. Notice is_faulty is not used. 
        #     This is to simulate the fact that the sensor itself doesnt know its faulty yet. 
        #     If you want to use it though, change it lol. 
        ok = True if not self.faulty_aware else (not is_faulty)
        self._tag_info.data = TagData(time = t, value = final_value, ok = ok)
        return self._tag_info.data
    
    def _sensor_dynamics(self, new_value: float|NDArray, t: float):
        if self._last_measurement is None:
            self._last_measurement = new_value
        dt = t - self._last_t
        if dt < 1e-12:
            return self._last_measurement
        lamb = dt / (self.time_constant + dt)
        self._last_t = t
        self._last_measurement = lamb * new_value + (1-lamb) * self._last_measurement
        return self._last_measurement
    @property
    def _last_value(self) -> TagData:
        return self._tag_info.data
    
    @abstractmethod
    def _should_update(self, t: float) -> bool:
        """
        Subclass hook that decides whether a fresh measurement is required.

        Typical implementations enforce sampling periods, latency, or
        dead-band logic.  Returning ``False`` short-circuits ``measure`` and
        reuses the previous :class:`TagData` sample.
        """
        pass

    @abstractmethod
    def _get_processed_value(self, t: float, raw_value: float | NDArray) -> float | NDArray:
        """
        Transform the true value into the sensor's processed output.

        Subclasses should perform any latency simulation, biasing, filtering,
        or other transformations here and return a value with the same shape as
        ``raw_value``.
        """
        pass

    def _apply_noise_and_faults(self, value: float | NDArray) -> tuple[float | NDArray, bool]:
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
            if self._last_value is not None:
                # A fault occurred, simulate what kind
                # 1. frozen value
                if self._rng.random() < 0.5:
                    return self._last_value.value, faulty
                # 2. spike
                value = self._last_value.value * (0.5 + self._rng.random())
                return value, faulty
            # No previous measurement to fall back on, so we let it pass but flag it
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

    @field_serializer("unit", mode="plain")
    def _serialize_unit(self, unit: UnitBase) -> str:
        return str(unit)