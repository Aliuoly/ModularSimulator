from __future__ import annotations
import collections
from typing import TYPE_CHECKING, override
from pydantic import Field, PrivateAttr
from modular_simulation.usables.sensors.sensor_base import SensorBase
from modular_simulation.utils.typing import Seconds, StateValue
from modular_simulation.usables.tag_info import TagData

if TYPE_CHECKING:
    from modular_simulation.framework.system import System


class SampledDelayedSensor(SensorBase):
    """
    A sensor with a set sampling frequency, measurement deadtime, and gaussian noise.
    """

    deadtime: Seconds = Field(
        default=0.0,
        ge=0.0,
        description="The measurement deadtime (a.k.a delay).",
    )

    sampling_period: Seconds = Field(
        default=0.0,
        ge=0.0,
        description=(
            "The sampling period of the sensor. "
            "This is how long it takes for new measurements to become available."
        ),
    )
    _first_sample: bool = PrivateAttr(default=True)
    _sample_queue: collections.deque[TagData] = PrivateAttr(default_factory=collections.deque)
    _t: Seconds = PrivateAttr()

    # ============================================================
    # ===                    LOGIC METHODS                    ====
    # ============================================================

    @override
    def _should_update(self, t: Seconds) -> bool:
        """Return True if the next sample should be taken."""
        if self._first_sample:
            return True
        return t > (self._t + self.sampling_period)

    @override
    def _post_commission(self, system: System) -> bool:
        """Subclass hook that is called after commissioning."""
        self._t = system.time
        return True

    def _get_sample(self, target_t: Seconds) -> tuple[TagData, bool]:
        """
        Consume and return the latest sample with time ≤ target_t.
        Optimized deque-safe version (bulk removal via loop unrolling).
        """
        q = self._sample_queue
        if not q:
            return self._tag_info.data, False

        # Fast-path: all samples valid → take last, clear queue
        if q[-1].time <= target_t:
            last_valid = q[-1]
            q.clear()
            return last_valid, True

        # Use local variables to reduce attribute lookups
        popleft = q.popleft

        # Small local loop but with reduced overhead
        last_valid = None
        while q and q[0].time <= target_t:
            last_valid = popleft()

        if last_valid is not None:
            return last_valid, True

        return self._tag_info.data, False

    @override
    def _get_processed_value(
        self,
        t: Seconds,
        raw_value: StateValue,
    ) -> tuple[StateValue, bool]:
        """
        Append new sample and return time-delayed measurement value.
        """
        q = self._sample_queue
        q_append = q.append
        q_append(TagData(t, raw_value))
        if self._first_sample:
            self._first_sample = False
            return q[0].value, True  # always return first sample so they have a value
        target_t = t - self.deadtime
        tag_data, successful = self._get_sample(target_t)
        self._t = t
        return tag_data.value, successful
