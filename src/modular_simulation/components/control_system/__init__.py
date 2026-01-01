from .control_element import ControlElement
from .trajectory import Trajectory
from .controller_mode import ControllerMode
from .abstract_controller import AbstractController
from .controllers import (
    PIDController,
    InternalModelController,
    CalculationModelPath,
    MVController,
    BangBangController,
    FirstOrderTrajectoryController,
)

# Backward compatibility alias
ControllerBase = AbstractController

__all__ = [
    "ControlElement",
    "Trajectory",
    "ControllerMode",
    "AbstractController",
    "ControllerBase",  # Deprecated alias for backward compatibility
    "PIDController",
    "InternalModelController",
    "CalculationModelPath",
    "MVController",
    "BangBangController",
    "FirstOrderTrajectoryController",
]
