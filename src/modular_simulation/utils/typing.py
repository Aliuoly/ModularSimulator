from numpy.typing import NDArray
import numpy as np
from typing import TypeAlias, Annotated
from pydantic import BeforeValidator, PlainSerializer
from astropy.units import Unit, Quantity, UnitBase, UnitsError

StateValue: TypeAlias = float | NDArray[np.float64]
ArrayIndex: TypeAlias = int | slice
# Broad, user-facing type alias
TimeQuantity: TypeAlias = (
    Quantity[Unit("day")] |
    Quantity[Unit("hour")] |
    Quantity[Unit("minute")] |
    Quantity[Unit("second")]
)
PerTimeQuantity: TypeAlias = (
    Quantity[Unit("1/day")] |
    Quantity[Unit("1/hour")] |
    Quantity[Unit("1/minute")] |
    Quantity[Unit("1/second")]
)
"""Represents any Astropy Quantity with per-time units (1/s, 1/min, 1/hr, 1/day)."""

# below are for pydantic model fields to handle additional
# validation and serialization
SerializableUnit = Annotated[
    UnitBase,
    BeforeValidator(lambda u: u if isinstance(u, UnitBase) else Unit(u)),
    PlainSerializer(lambda u: str(u)),
]

def second_validator(v: dict | Quantity | float) -> float:
    # 1. convert dictionary input to quantity
    #   e.g. {"value": 10, "unit": "s"}
    if isinstance(v, dict) and "value" in v:
        val = v["value"]
        unit = v.get("unit", "s")
        v = Quantity(val, unit)
    # try to convert quantity, which was either parsed
    # from dict or passed directly, to seconds and return its magnitude
    if isinstance(v, Quantity):
        if v.unit.is_equivalent("second"):
            return v.to_value("second")
        else:
            raise UnitsError(f"Expected a time unit, got {v.unit}")
    # if not a quantity, assume it is already in seconds as float
    elif not isinstance(v, Quantity):
        return float(v)
    raise TypeError(f"Expected float or Quantity, got {type(v)}")

def per_second_validator(v):
    # 1. convert dictionary input to quantity
    #   e.g. {"value": 10, "unit": "1/s"}
    if isinstance(v, dict) and "value" in v:
        val = v["value"]
        unit = v.get("unit", "1/s")
        v = Quantity(val, unit)
    # try to convert quantity, which was either parsed
    # from dict or passed directly, to 1/seconds and return its magnitude
    if isinstance(v, Quantity):
        if v.unit.is_equivalent("1/second"):
            return v.to_value("1/second")
        else:
            raise UnitsError(f"Expected a per-time unit, got {v.unit}")
    # if not a quantity, assume it is already in 1/seconds as float
    elif not isinstance(v, Quantity):
        return float(v)
    raise TypeError(f"Expected float or Quantity, got {type(v)}")
    
Seconds = Annotated[
    float,
    BeforeValidator(second_validator),
    PlainSerializer(lambda v: {"value": v, "unit": "s"}),
]
"""A time value in **seconds**.

If used to type hint a pydantic model field, 
will convert to seconds internally if given a `Quantity` with compatible time units.
If used to type hint a method argument, it is assumed to be a float in seconds - no 
conversion is performed inherently - be sure to use the 'second_value' wrapper if needed.
It can be found at modular_simulation.utils.wrappers

Serializes as `{"value": float, "unit": "s"}`.

float docstring:
"""

PerSeconds = Annotated[
    float,
    BeforeValidator(per_second_validator),
    PlainSerializer(lambda v: {"value": v, "unit": "1/s"}),
]
"""A time value in **1/seconds**.

If used to type hint a pydantic model field, 
will convert to 1/seconds internally if given a `Quantity` with compatible time units.
If used to type hint a method argument, it is assumed to be a float in 1/seconds - no 
conversion is performed inherently - be sure to use the 'per_second_value' wrapper if needed.
It can be found at modular_simulation.utils.wrappers

Serializes as `{"value": float, "unit": "1/s"}`.

float docstring:
"""