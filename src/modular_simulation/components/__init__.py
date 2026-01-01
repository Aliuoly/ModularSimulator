from .calculations import (
    AbstractCalculation,
    CalculationBase,
    PointMetadata,
    TagType,
    FirstOrderFilter,
)
from .sensors import AbstractSensor, SensorBase
from .point import DataValue, Point, PointRegistry
from .sensors.sampled_delayed_sensor import SampledDelayedSensor
from .control_system import (
    AbstractController,
    CalculationModelPath,
    ControllerBase,
    InternalModelController,
    PIDController,
    BangBangController,
    Trajectory,
    FirstOrderTrajectoryController,
    ControllerMode,
    MVController,
    ControlElement,
)

# Backward compatibility aliases
TagData = DataValue

__all__ = [
    # New abstract base class names
    "AbstractSensor",
    "AbstractController",
    "AbstractCalculation",
    # Legacy aliases (deprecated)
    "SensorBase",
    "ControllerBase",
    "CalculationBase",
    "TagData",  # Deprecated, use DataValue
    # Controllers
    "ControllerMode",
    "MVController",
    "ControlElement",
    "Trajectory",
    "PIDController",
    "InternalModelController",
    "CalculationModelPath",
    "BangBangController",
    "FirstOrderTrajectoryController",
    # Sensors
    "SampledDelayedSensor",
    # Point module
    "DataValue",
    "Point",
    "PointRegistry",
    # Calculations
    "PointMetadata",
    "TagType",
    "FirstOrderFilter",
]
