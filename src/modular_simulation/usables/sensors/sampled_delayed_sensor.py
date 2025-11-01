import numpy as np
from modular_simulation.usables.sensors.sensor_base import SensorBase, TagData
from modular_simulation.utils.typing import TimeValue, StateValue
from modular_simulation.utils.wrappers import second, second_value
import collections
from pydantic import Field, PrivateAttr, PlainSerializer, BeforeValidator
from typing import Annotated


class SampledDelayedSensor(SensorBase):
    """
    A sensor with a set sampling frequency, measurement deadtime, and gaussian noise.
    """
    
    deadtime: Annotated[TimeValue, BeforeValidator(second), PlainSerializer(second_value),] = Field(
        0.0,
        ge = 0.0,
        description = "The measurement deadtime (a.k.a delay) in system units"
    )

    sampling_period: Annotated[TimeValue, BeforeValidator(second), PlainSerializer(second_value),] = Field(
        0.0,
        ge = 0.0,
        description = "The sampling period of the sensor in system units. " \
                        "This is how long it takes for new measurements to become available."
    )

    _sample_queue: collections.deque[TagData] = PrivateAttr(default_factory=collections.deque)

    def _should_update(self, t: TimeValue) -> bool:
        """
        SampledDelayedSensor should update when a new sample, as determined by 
        the configured sampling time and measurement deadtime, is available. 
        """
        if t > self._t + self.sampling_period:
            return True
        return False

    def _get_sample(self, target_t: TimeValue) -> tuple[TagData, bool]:
        """Consume and return the latest sample with time ≤ target_t.
        If none are available, set the successful flag to False
        """
        successful = True
        # Fast-path: empty queue
        if not self._sample_queue:
            return self._tag_info.data, not successful

        # Pop samples while they are valid
        last_valid_sample = None
        while self._sample_queue and self._sample_queue[0].time <= target_t:
            last_valid_sample = self._sample_queue.popleft()

        # If we found at least one valid sample, update and return it
        if last_valid_sample is not None:
            return last_valid_sample, successful

        # Otherwise, no new sample is ready yet
        return self._tag_info.data, not successful
    
    def _get_processed_value(self, raw_value: StateValue, t: TimeValue) -> tuple[StateValue, bool]:
        """
        Retrieves the time-delayed measurement value by looking it up in
        the sample queue. 
        1. Appends new sample to queue for future iterations.
        2. Looks up the time-delayed sample and returns it. 
        """
        self._sample_queue.append(TagData(t, raw_value))
        
        target_t = t - self.deadtime
        return_tagdata, successful = self._get_sample(target_t)

        return return_tagdata.value, successful

        
