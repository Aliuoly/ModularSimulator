from numpy.typing import NDArray
import numpy as np
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from typing import TYPE_CHECKING, Any, List
from modular_simulation.usables.time_value_quality_triplet import TimeValueQualityTriplet

if TYPE_CHECKING:
    from modular_simulation.quantities import MeasurableQuantities
    from modular_simulation.measurables.base_classes import BaseIndexedModel

class Sensor(BaseModel, ABC):

    measurement_tag: str = Field(
        ...,
        description = "tag of the state or control element to measure."
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
    
    random_seed: int = Field(0)

    _rng: np.random.Generator = PrivateAttr()
    _last_value: TimeValueQualityTriplet | None = PrivateAttr(default = None)
    _measurement_owner: "BaseIndexedModel" = PrivateAttr()
    _initialized: bool = PrivateAttr(default = False)
    _history: List["TimeValueQualityTriplet"] = PrivateAttr(default_factory = list)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._rng = np.random.default_rng(self.random_seed)

    def _initialize(self, measurable_quantities: "MeasurableQuantities") -> None:
        """
        Links the measurable quantities instance and
        finds the correct attribute to measure and creates a simple callable for it.
        This runs during system initialization.
        """
        search_order = list(measurable_quantities.__class__.model_fields.keys())
        found_owner = None
        
        for category in search_order:
            owner = getattr(measurable_quantities, category)
            if owner is not None and hasattr(owner, self.measurement_tag):
                if found_owner is not None:
                    raise AttributeError(
                        f"Tag '{self.measurement_tag}' is ambiguous and found in multiple fields of measurable_quantities."
                        )
                found_owner = owner
        if found_owner is None:
            raise AttributeError(
                f"Tag '{self.measurement_tag}' not found in any field of measurable_quantities. "
                f"Available measurable quantities are: {', '.join(measurable_quantities.available_tags)}"
                )
        
        # Store raw pieces
        self._measurement_owner = found_owner          # new PrivateAttr
        self._initialized = True

    # --- Template Method ---
    def measure(self, t: float) -> TimeValueQualityTriplet:
        """
        Public method that defines the complete measurement algorithm.
        """

        # 1. Decide if a new measurement should be taken
        if not self._should_update(t) and self._last_value is not None:
            return self._last_value

        # 2. Get the true, raw value from the system. At this point, 
        #       measurement function is NOT None, so Mypy shut up lol (hence type: ignore)
        raw_value = getattr(self._measurement_owner, self.measurement_tag)

        # 3. Apply subclass-specific processing (e.g., time delay) to the true value
        processed_value = self._get_processed_value(raw_value, t)

        # 4. Apply noise and faults to the processed value
        final_value, is_faulty = self._apply_noise_and_faults(processed_value)

        # 5. Create and store the new measurement. Notice is_faulty is not used. 
        #     This is to simulate the fact that the sensor itself doesnt know its faulty yet. 
        #     If you want to use it though, change it lol. 
        ok = True if not self.faulty_aware else (not is_faulty)
        new_measurement = TimeValueQualityTriplet(t = t, value = final_value, ok = ok)
        self._last_value = new_measurement
        self._history.append(new_measurement)
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

    def measurement_history(self) -> List[TimeValueQualityTriplet]:
        return self._history.copy()
