from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
)
from typing import Any, TYPE_CHECKING, override, cast
from collections.abc import Callable
from collections import deque
from modular_simulation.usables.tag_info import TagData, TagInfo
from modular_simulation.utils.typing import StateValue, SerializableUnit, Seconds
from astropy.units import UnitBase
from modular_simulation.validation.exceptions import SensorConfigurationError

if TYPE_CHECKING:
    from modular_simulation.framework.system import System


class RNGCache:
    """Efficient cache for random numbers to reduce per-call RNG overhead."""

    _rng: np.random.Generator
    _cache_size: int
    _uniform_cache: deque[float]
    _normal_cache: deque[float]

    def __init__(self, seed: int | None = None, cache_size: int = 10_000):
        self._rng = np.random.default_rng(seed)
        self._cache_size = cache_size
        self._uniform_cache = deque()
        self._normal_cache = deque()

    def random(self) -> float:
        """Return one uniform random number [0,1)."""
        if not self._uniform_cache:
            self._uniform_cache.extend(self._rng.random(self._cache_size))
        return self._uniform_cache.popleft()

    def normal(self) -> float:
        """Return one standard normal random number (mean 0, std 1)."""
        if not self._normal_cache:
            self._normal_cache.extend(self._rng.standard_normal(self._cache_size))
        return self._normal_cache.popleft()


class SensorBase(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """Abstract base class for all sensors in the modular simulation framework.

    Subclasses implement the scheduling and signal conditioning steps while
    the base class wires the sensor into the orchestrated
    :class:`~modular_simulation.measurables.base_classes.MeasurableQuantities`
    instance.  Common responsibilities such as unit conversion, alias tag
    management, clipping to instrument ranges, and the injection of noise/fault
    models are handled here.  The latest
    :class:`~modular_simulation.usables.tag_info.TagData` sample is stored on
    the sensor's :class:`~modular_simulation.usables.tag_info.TagInfo`, which
    also historizes every emitted measurement for later analysis.
    """

    measurement_tag: str = Field(..., description="tag of the state or control element to measure.")
    alias_tag: str | None = Field(
        default=None,
        description=(
            "alias of the measurement to be used when the sensor returns results."
            "e.g., if measurement_tag was 'cumm_MI' and alias tag was 'lab_MI', "
            "then, a usable with tag 'lab_MI' will be available, while 'cumm_MI' would not be available."
        ),
    )
    unit: SerializableUnit = Field(
        description="Unit of the measured quantity. Will be parsed with Astropy if is a string. "
    )
    description: str = Field(
        default="No description provided.",
        description="Description of the sensor's measurement.",
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
    instrument_range: tuple[StateValue, StateValue] = Field(
        default_factory=lambda: (-float("inf"), float("inf")),
        description=(
            "the range of value this sensor can return. "
            "If true value is beyong the range, it is clipped and returned. "
        ),
    )
    random_seed: int = Field(default=0, description="Random seed for signal noise injection")

    # ----construction time initialized----
    _rng_cache: RNGCache = PrivateAttr()
    _tag_info: TagInfo = PrivateAttr()
    _alias_tag: str = PrivateAttr()

    # ----commission time initialized----
    _measurement_getter: Callable[[], StateValue] = PrivateAttr()
    # all internal calculation that requires the previous measurement and timestamp should use these
    # these may or may not be equivalent to _tag_info.data.value and .t
    _measurement: StateValue = PrivateAttr()
    _initialized: bool = PrivateAttr(default=False)
    _is_scalar: bool = PrivateAttr(default=False)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def model_post_init(self, context: Any) -> None:  # pyright: ignore[reportAny, reportExplicitAny]
        self._rng_cache = RNGCache(seed=self.random_seed, cache_size=100_00)
        if self.alias_tag is None:
            self.alias_tag = self.measurement_tag
        self._alias_tag = self.alias_tag
        self._tag_info = TagInfo(
            tag=self.alias_tag,
            unit=cast(UnitBase, self.unit),
            type="measured",
            description=self.description,
            _raw_tag=self.measurement_tag,
        )

    def commission(self, system: System) -> bool:
        """
        Commission the sensor for the process (lol)
        Resolve the measurement source and prime the sensor state.
        Note that all necessary sensor states (last measurements, times, etc.)
        are initialized here as well, so no subsequent operations
        will check for "if self.something is None"
        """
        desired_state = self.measurement_tag
        process = system.process_model
        state_info = process.state_metadata_dict.get(desired_state)
        if state_info is None:
            raise SensorConfigurationError(
                f"'{desired_state}' sensor's desired measurement doesn't exist in the process model."
            )
        unit = cast(UnitBase, self.unit)
        if not unit.is_equivalent(state_info.unit):
            raise SensorConfigurationError(
                f"'{desired_state}' sensor's desired unit '{unit}' is not compatible "
                + f"with the measured state's unit '{state_info.unit}'."
            )

        self._measurement_getter = process.make_converted_getter(desired_state, unit)
        self._measurement = self._measurement_getter()
        self._initialized = True
        _ = self.measure(t=system.time, force=True)  # flush the update logic
        if np.isscalar(self._measurement):
            self._is_scalar = True
        # if somehow the measurement returned NAN, something went wrong and the
        # commissioning failed.
        successful = True
        if np.any(np.isnan(self._tag_info.data.value)):
            return not successful

        successful = self._post_commission(system)
        return successful

    def _post_commission(self, system: System) -> bool:
        """Subclass hook that is called after commissioning."""
        return True

    def measure(self, t: Seconds, *, force: bool = False) -> TagData:
        """Execute the measurement pipeline and return the resulting :class:`TagData`.

        The method enforces initialization, optionally reuses the most recent
        measurement when no refresh is required, gathers the raw value from the
        linked model, applies subclass-defined processing, and then runs the
        shared noise/fault machinery before historizing the result.
        """
        if not self._initialized:
            raise RuntimeError(
                "Tried to call 'measure' before the system orchestrated the various quantities. "
                + "Make sure this sensor is part of a system and the system has been constructed."
            )
        if not self._should_update(t) and not force:
            return self._tag_info.data

        raw_value = self._measurement_getter()
        processed_value, successful = self._get_processed_value(t=t, raw_value=raw_value)
        if not successful:
            return self._tag_info.data

        if self._is_scalar:
            processed_value = max(
                min(processed_value, self.instrument_range[1]), self.instrument_range[0]
            )
        else:
            processed_value = np.clip(processed_value, *self.instrument_range)

        final_value, is_faulty = self._apply_noise_and_faults(processed_value)

        ok = True if not self.faulty_aware else (not is_faulty)
        self._tag_info.data = TagData(time=t, value=final_value, ok=ok)
        return self._tag_info.data

    @abstractmethod
    def _should_update(self, t: Seconds) -> bool:
        """
        Subclass hook that decides whether a fresh measurement is required.

        Typical implementations enforce sampling periods, latency, or
        dead-band logic.  Returning ``False`` short-circuits ``measure`` and
        reuses the previous :class:`TagData` sample.
        """
        pass

    @abstractmethod
    def _get_processed_value(self, t: Seconds, raw_value: StateValue) -> tuple[StateValue, bool]:
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
        if self._rng_cache.random() < self.faulty_probability:
            faulty = True
            # A fault occurred, simulate what kind
            # 1. frozen value
            if self._rng_cache.random() < 0.5:
                return self._measurement, faulty
            # 2. spike
            value *= 0.5 + self._rng_cache.random()
            return value, faulty

        # If not faulty, apply noise
        if self.coefficient_of_variance > 0:
            noise_std_dev = np.abs(value * self.coefficient_of_variance, dtype=float)
            if isinstance(value, np.ndarray):
                for i in range(len(value)):
                    value[i] += self._rng_cache.normal() * noise_std_dev[i]
            else:
                value += self._rng_cache.normal() * noise_std_dev

        return value, False

    @property
    def tag_info(self) -> TagInfo:
        """Public getter of the private _tag_info attribute."""
        return self._tag_info
