from numpy.typing import NDArray
import numpy as np
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from typing import TYPE_CHECKING, Any, Callable
from dataclasses import dataclass
import functools

if TYPE_CHECKING:
    from modular_simulation.quantities import MeasurableQuantities

@dataclass(slots = True)
class Measurement:
    """Simple container class for a single measurement."""
    t: float
    value: float | NDArray
    faulty: bool = False

class Sensor(BaseModel, ABC):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    measurement_tag: str = Field(...)
    coefficient_of_variance: float = Field(0.0)
    faulty_probability: float = Field(0.0, ge=0.0, lt=1.0)
    random_seed: int = Field(0)

    _rng: np.random.Generator = PrivateAttr()
    _last_measurement: Measurement | None = PrivateAttr(default=None)
    _measurement_function: Callable[["MeasurableQuantities"], float | NDArray] | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._rng = np.random.default_rng(self.random_seed)

    def _initialize_measurement_function(self, measurable_quantities: "MeasurableQuantities") -> None:
        """
        Finds the correct attribute to measure and creates a simple callable for it.
        This runs only once when .measure is called the first time.
        """
        search_order = list(measurable_quantities.__class__.model_fields.keys())
        found_path = None

        for category in search_order:
            category_obj = getattr(measurable_quantities, category)
            if category_obj and hasattr(category_obj, self.measurement_tag):
                if found_path is not None:
                    raise AttributeError(f"Tag '{self.measurement_tag}' \
                                         is ambiguous and found in multiple fields of measurable_quantities.")
                found_path = f"{category}.{self.measurement_tag}"
        
        if found_path is None:
            raise AttributeError(f"Tag '{self.measurement_tag}' not found in any field of measurable_quantities.")
        
        # Create a simple function to get the value from the path
        path_parts = found_path.split('.')
        # Create a function that will start with measurable_quantities and
        # successively call getattr on it with each part of the path.
        self._measurement_function = lambda mq: functools.reduce(getattr, path_parts, mq)

    # --- Template Method ---
    def measure(self, measurable_quantities: "MeasurableQuantities", t: float) -> Measurement:
        """
        Public method that defines the complete measurement algorithm.
        """
        # One-time setup for the measurement function
        if self._measurement_function is None:
            self._initialize_measurement_function(measurable_quantities)

        # 1. Decide if a new measurement should be taken
        if not self._should_update(t) and self._last_measurement is not None:
            return self._last_measurement

        # 2. Get the true, raw value from the system. At this point, 
        #       measurement function is NOT None, so Mypy shut up lol (hence type: ignore)
        raw_value = self._measurement_function(measurable_quantities) #type: ignore

        # 3. Apply subclass-specific processing (e.g., time delay) to the true value
        processed_value = self._get_processed_value(raw_value, t)

        # 4. Apply noise and faults to the processed value
        final_value, is_faulty = self._apply_noise_and_faults(processed_value)

        # 5. Create and store the new measurement. Notice is_faulty is not used. 
        #     This is to simulate the fact that the sensor itself doesnt know its faulty yet. 
        #     If you want to use it though, change it lol. 
        new_measurement = Measurement(t=t, value=final_value, faulty=False)
        self._last_measurement = new_measurement
        
        return new_measurement
    
    @abstractmethod
    def _should_update(self, t: float) -> bool:
        """
        Subclass logic to determine if a new measurement should be processed.
        (e.g., for handling sampling periods).
        """
        pass

    @abstractmethod
    def _get_processed_value(self, raw_value: float | NDArray, t: float) -> float | NDArray:
        """
        Takes the true, raw value and applies subclass-specific logic,
        such as time delays, filtering, etc.
        """
        pass
    
    def _apply_noise_and_faults(self, value: float | NDArray) -> tuple[float | NDArray, bool]:
        """
        Applies common fault and noise models. Returns the final value and a fault flag.
        """
        # Check for a fault
        if self._rng.random() < self.faulty_probability:
            if self._last_measurement is not None:
                # A fault occurred, return the last known value
                return self._last_measurement.value, True
            # No previous measurement to fall back on, so we let it pass but flag it
            return value, True

        # If not faulty, apply noise
        noisy_value = value
        if self.coefficient_of_variance > 0:
            noise_std_dev = np.abs(value * self.coefficient_of_variance)
            noise = self._rng.normal(loc=0.0, scale=noise_std_dev)
            noisy_value = value + noise
        
        return noisy_value, False