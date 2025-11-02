import numpy as np
import collections
from pydantic import Field, PrivateAttr
from typing import Annotated
from modular_simulation.usables.sensors.sensor_base import SensorBase, TagData
from modular_simulation.utils.typing import Seconds, StateValue

class SampledDelayedSensor(SensorBase):
    """
    A sensor with a set sampling frequency, measurement deadtime, and gaussian noise.
    """
    
    deadtime: Seconds = Field(
        0.0,
        ge=0.0,
        description="The measurement deadtime (a.k.a delay).",
    )

    sampling_period: Seconds = Field(
        0.0,
        ge=0.0,
        description=(
            "The sampling period of the sensor. "
            "This is how long it takes for new measurements to become available."
        ),
    )

    _sample_queue: collections.deque[TagData] = PrivateAttr(default_factory=collections.deque)

    # ============================================================
    # ===                    LOGIC METHODS                    ====
    # ============================================================

    def _should_update(self, t: Seconds) -> bool:
        """Return True if the next sample should be taken."""
        return t > (self._t + self.sampling_period)
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
        q0 = q[0]

        # Small local loop but with reduced overhead
        last_valid = None
        while q and q[0].time <= target_t:
            last_valid = popleft()

        if last_valid is not None:
            return last_valid, True

        return self._tag_info.data, False
    
    def _get_processed_value(self, raw_value: StateValue, t: Seconds) -> tuple[StateValue, bool]:
        """
        Append new sample and return time-delayed measurement value.
        """
        q = self._sample_queue
        q_append = q.append
        q_append(TagData(t, raw_value))

        target_t = t - self.deadtime
        tag_data, successful = self._get_sample(target_t)
        return tag_data.value, successful
