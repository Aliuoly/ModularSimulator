import numpy as np
from modular_simulation.usables.sensor import Sensor, TimeValueQualityTriplet
import collections
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr



class SampledDelayedSensor(Sensor):
    """
    A sensor with a set sampling frequency, measurement deadtime, and gaussian noise.
    """
    
    deadtime: float = Field(
        0.0,
        ge = 0.0,
        description = "The measurement deadtime (a.k.a delay) in system units"
    )

    sampling_period: float = Field(
        0.0,
        ge = 0.0,
        description = "The sampling period of the sensor in system units. " \
                        "This is how long it takes for new measurements to become available."
    )

    _sample_queue: collections.deque[TimeValueQualityTriplet] = PrivateAttr(default_factory=collections.deque)

        
    def _should_update(self, t: float) -> bool:
        """
        Sensor specific logic to determine if a new measurement should be processed and returned
        """
        new_sample_available = True # default to do update if no last measurement
        if self._last_value is not None:
            expected_available_t = self._last_t_updated + self.sampling_period
            new_sample_available = t >= expected_available_t
        
        return new_sample_available

    def _get_processed_value(self, raw_value: float | NDArray[np.float64], t: float) -> float | NDArray[np.float64]:
        """
        Takes the possibly faulty and noisy value and applies subclass-specific logic,
        such as time delays, filtering, etc.
        """
        # process value and append to queue
        measurement = TimeValueQualityTriplet(t, raw_value)
        self._sample_queue.append(measurement)

        # determine the timestamp of the measurement to be returned
        target_t = t - self.deadtime

        # Remove samples that are too old.
        # when the oldest sample in queue has timestamp > target_t,
        # then the last removed sample 
        #   (which was the last available sample with timestamp <= target_t) 
        #       is to be returned. 
        return_sample = self._sample_queue.popleft()
        while len(self._sample_queue) > 1 and self._sample_queue[0].t <= target_t:
            return_sample = self._sample_queue.popleft()
        # reappend the sample just in case
        self._sample_queue.appendleft(return_sample)

        self._last_value = return_sample

        self._last_t_updated = t
        return return_sample.value

        
