from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from astropy.units import UnitBase, Unit
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer, field_validator

from modular_simulation.interfaces.tag_info import TagData, TagInfo

if TYPE_CHECKING:
    from modular_simulation.core.dynamic_model import DynamicModel


def make_measurement_getter(source: "DynamicModel", tag: str, converter: Callable[[Any], Any]) -> Callable[[], float | NDArray]:
    def measurement_getter() -> float | NDArray:
        return converter(getattr(source, tag))

    return measurement_getter


class SensorBase(BaseModel, ABC):
    """Abstract base class for all sensors in the modular simulation framework."""

    measurement_tag: str = Field(
        ..., description="tag of the state or control element to measure."
    )
    alias_tag: str | None = Field(
        default=None,
        description=(
            "alias of the measurement to be used when the sensor returns results."
            "e.g., if measurement_tag was 'cumm_MI' and alias tag was 'lab_MI', "
            "then, a usable with tag 'lab_MI' will be available, while 'cumm_MI' would not be available."
        ),
    )
    unit: str | UnitBase = Field(
        description="Unit of the measured quantity. Will be parsed with Astropy if provided as a string."
    )
    description: str | None = Field(
        default=None, description="Description of the sensor's measurement."
    )
    coefficient_of_variance: float = Field(
        default=0.0,
        description=(
            "the standard deviation of the measurement noise, defined as "
            "a fraction of the true value of measurement."
        ),
    )

    faulty_probability: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="probability for the sensor to err (stall or spike)",
    )

    faulty_aware: bool = Field(
        default=False,
        description=(
            "whether or not the sensor knows it has erred. "
            "if True, the .ok field of the measurement result will be set automatically. "
            "if False, the .ok field of the measurement will not be set by the sensor routine. "
        ),
    )
    instrument_range: tuple[float, float] = Field(
        default_factory=lambda: (-float("inf"), float("inf")),
        description=(
            "the range of value this sensor can return. "
            "If true value is beyond the range, it is clipped and returned."
        ),
    )
    time_constant: float = Field(
        default=0.0,
        description=(
            "The timeconstant associated with the dynamics of "
            "this sensor. The final measurement is then filtered "
            "based on this time constant to mimic sensor dynamics. "
        ),
    )
    random_seed: int = Field(0)

    _rng: np.random.Generator = PrivateAttr()
    _tag_info: TagInfo = PrivateAttr()
    _last_measurement: float | NDArray | None = PrivateAttr(default=None)
    _last_t: float = PrivateAttr(default=0)
    _initialized: bool = PrivateAttr(default=False)
    _measurement_getter: Callable[[], float | NDArray] = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("unit", mode="before")
    @classmethod
    def convert_unit(cls, unit: str | UnitBase) -> UnitBase:
        if isinstance(unit, str):
            return Unit(unit)
        return unit

    def model_post_init(self, context: Any) -> None:
        self._rng = np.random.default_rng(self.random_seed)
        if self.alias_tag is None:
            self.alias_tag = self.measurement_tag
        self._tag_info = TagInfo(
            tag=self.alias_tag,
            unit=self.unit,
            description="" if self.description is None else self.description,
        )

    def _initialize(self, dynamic_model: "DynamicModel") -> None:
        measurement_unit = dynamic_model.tag_unit_info[self.measurement_tag]
        converter = measurement_unit.get_converter(self._tag_info.unit)
        self._measurement_getter = make_measurement_getter(
            dynamic_model, self.measurement_tag, converter
        )
        self._initialized = True
        self.measure(t=0)

    def measure(self, t: float) -> TagData:
        if not self._initialized:
            raise RuntimeError(
                "Tried to call 'measure' before the system orchestrated the various quantities. "
                "Make sure this sensor is part of a system and the system has been constructed."
            )
        if not self._should_update(t) and self._last_value is not None:
            return self._last_value

        raw_value = self._measurement_getter()

        processed_value = np.clip(
            self._get_processed_value(t=t, raw_value=raw_value), *self.instrument_range
        )
        processed_value = self._sensor_dynamics(processed_value, t)

        final_value, is_faulty = self._apply_noise_and_faults(processed_value)

        ok = True if not self.faulty_aware else (not is_faulty)
        self._tag_info.data = TagData(time=t, value=final_value, ok=ok)
        return self._tag_info.data

    def _sensor_dynamics(self, new_value: float | NDArray, t: float):
        if self._last_measurement is None:
            self._last_measurement = new_value
        dt = t - self._last_t
        if dt < 1e-12:
            return self._last_measurement
        lamb = dt / (self.time_constant + dt)
        self._last_t = t
        self._last_measurement = lamb * new_value + (1 - lamb) * self._last_measurement
        return self._last_measurement

    @property
    def _last_value(self) -> TagData:
        return self._tag_info.data

    @abstractmethod
    def _should_update(self, t: float) -> bool:
        pass

    @abstractmethod
    def _get_processed_value(self, t: float, raw_value: float | NDArray) -> float | NDArray:
        pass

    def _apply_noise_and_faults(self, value: float | NDArray) -> tuple[float | NDArray, bool]:
        noisy_value = value
        is_faulty = False
        if self.coefficient_of_variance > 0.0:
            std_dev = self.coefficient_of_variance * np.abs(value)
            noisy_value = self._rng.normal(value, std_dev)
        if self.faulty_probability > 0.0 and self._rng.random() < self.faulty_probability:
            is_faulty = True
            noisy_value = noisy_value + self._rng.normal(0.0, np.abs(value))
        return noisy_value, is_faulty

    @field_serializer("unit")
    def serialize_unit(self, unit: UnitBase) -> str:
        return str(unit)


__all__ = ["SensorBase", "make_measurement_getter"]
