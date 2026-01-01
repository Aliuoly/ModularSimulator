from .abstract_sensor import AbstractSensor
from .sampled_delayed_sensor import SampledDelayedSensor

# Backward compatibility alias
SensorBase = AbstractSensor

__all__ = [
    "AbstractSensor",
    "SensorBase",  # Deprecated alias for backward compatibility
    "SampledDelayedSensor",
]
