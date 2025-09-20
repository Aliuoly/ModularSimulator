import numpy as np
from modular_simulation.usables.sensors.sensor import Sensor, TimeValueQualityTriplet
import collections
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr
from typing import Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from modular_simulation.quantities import MeasurableQuantities



class SampledDelayedSensor(Sensor):
    """
    A sensor with a set sampling frequency, measurement deadtime, and gaussian noise.
    """
    
    measurement_tag: str = Field(
        ..., 
        description="Tag used to extract values from an instance of MeasurableQuantities.\
                     Must correspond to an attribute in one of the fields of MeasurableQuantities."
        )
    
    coefficient_of_variance: float = Field(
        0, 
        description = "The standard deviation of the measurement noise as a fraction of the measured value."
        )
    
    faulty_probability: float = Field(
        0.0, 
        ge = 0.0, 
        lt = 1.0, 
        description = "The probability that a given measurement is faulty."
        )
    
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
    
    random_seed: int = Field(
        0, 
        description = "Random seed used in measurement noise sampling"
        )
    
    _rng: np.random.Generator = PrivateAttr()

    _last_value: TimeValueQualityTriplet | None = PrivateAttr()

    _sample_queue: collections.deque[TimeValueQualityTriplet] = PrivateAttr()
    _measurement_function: Callable[["MeasurableQuantities"], float | NDArray] | None = PrivateAttr(default=None)

    def __init__(
            self,
            measurement_tag: str, 
            coefficient_of_variance: float = 0., 
            faulty_probability: float = 0., 
            deadtime: float = 0., 
            sampling_period: float = 0., 
            random_seed: int = 0
            ):
        super().__init__(
            measurement_tag = measurement_tag,
            coefficient_of_variance = coefficient_of_variance,
            faulty_probability = faulty_probability,
            random_seed = random_seed
        )
        self.measurement_tag  = measurement_tag
        self.coefficient_of_variance = coefficient_of_variance
        self.faulty_probability = faulty_probability
        self.deadtime = deadtime
        self.sampling_period = sampling_period
        self.random_seed = random_seed

        self._sample_queue = collections.deque()
        self._last_value = None
        self._last_t_updated = -np.inf # force update first iteration
        
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

        
