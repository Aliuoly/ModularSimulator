from numpy.typing import NDArray
import numpy as np
from typing import TypeAlias
from astropy.units import Unit, Quantity

StateValue: TypeAlias = float | NDArray[np.float64]
ArrayIndex: TypeAlias = int | slice

# intended for type hint use in method signitures where
# validation and serialization is not necessry
TimeValue: TypeAlias = \
    Quantity[Unit("day")]    | Quantity[Unit("hour")] |    \
    Quantity[Unit("minute")] | Quantity[Unit("second")]

PerTimeValue: TypeAlias = \
    Quantity[Unit("1/day")]    | Quantity[Unit("1/hour")] |    \
    Quantity[Unit("1/minute")] | Quantity[Unit("1/second")]

