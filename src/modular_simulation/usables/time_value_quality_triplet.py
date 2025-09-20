from dataclasses import dataclass
from numpy.typing import NDArray

@dataclass(slots = True)
class TimeValueQualityTriplet:
    """Simple container class for a single measurement or calculation"""
    t: float
    value: float | NDArray
    ok: bool = False # whether or not the value is ok (not faulty)