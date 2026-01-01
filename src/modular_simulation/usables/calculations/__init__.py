from .abstract_calculation import AbstractCalculation
from .point_metadata import PointMetadata, TagType
from .first_order_filter import FirstOrderFilter

# Backward compatibility alias
CalculationBase = AbstractCalculation

__all__ = [
    "AbstractCalculation",
    "CalculationBase",  # Deprecated alias for backward compatibility
    "PointMetadata",
    "TagType",
    "FirstOrderFilter",
]
