from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np


@dataclass(slots=True)
class TagData:
    """Simple container class for a single measurement or calculation."""

    time: float = 0.0
    value: float | NDArray = np.nan
    ok: bool = False  # whether or not the value is ok (not faulty)
