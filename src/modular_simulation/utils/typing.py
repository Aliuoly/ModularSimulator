from numpy.typing import NDArray
import numpy as np
from typing import TypeAlias, Annotated
from pydantic import BeforeValidator, PlainSerializer
from astropy.units import Unit, Quantity, UnitBase, UnitsError

StateValue: TypeAlias = float | NDArray[np.float64]
ArrayIndex: TypeAlias = int | slice
# Broad, user-facing type alias
TimeQuantity: TypeAlias = (
    Annotated[Quantity, Unit("day")]
    | Annotated[Quantity, Unit("hour")]
    | Annotated[Quantity, Unit("minute")]
    | Annotated[Quantity, Unit("second")]
)
PerTimeQuantity: TypeAlias = (
    Annotated[Quantity, Unit("1/day")]
    | Annotated[Quantity, Unit("1/hour")]
    | Annotated[Quantity, Unit("1/minute")]
    | Annotated[Quantity, Unit("1/second")]
)
"""Represents any Astropy Quantity with per-time units (1/s, 1/min, 1/hr, 1/day)."""


# below are for pydantic model fields to handle additional
# validation and serialization
def unit_validator(v: UnitBase | str) -> UnitBase:
    return v if isinstance(v, UnitBase) else Unit(v)


def unit_serializer(v: UnitBase) -> str:
    return str(v)


SerializableUnit = Annotated[
    UnitBase | str,
    BeforeValidator(unit_validator),
    PlainSerializer(unit_serializer),
]
"""
Represents any Astropy Unit that also parses string inputs. 
For pydantic BaseModels with fields annotated with SerializableUnit,
when serliazing, the unit is converted to its string representation, since
astropy.unit.Unit is NOT serializable. 
"""


def second_validator(v: dict[str, float | str] | Quantity | float) -> float:
    # 1. convert dictionary input to quantity
    #   e.g. {"value": 10, "unit": "s"}
    if isinstance(v, dict):
        try:
            val = v["value"]
            unit = v.get("unit", "s")
            v = Quantity(float(val), str(unit))
        except Exception:
            raise ValueError(f"Invalid input dictionary v: {v}")
    # try to convert quantity, which was either parsed
    # from dict or passed directly, to seconds and return its magnitude
    if isinstance(v, Quantity):
        if np.isscalar(v.value):
            if v.unit.is_equivalent("second"):
                return float(v.to_value("second"))
            else:
                raise UnitsError(f"Expected a time unit, got {v.unit}")
        else:
            raise ValueError(f"Expected a scalar Quantity, got {v}")
    # if not a quantity, assume it is already in seconds as float
    elif not isinstance(v, Quantity):
        return float(v)
    raise TypeError(f"Expected float or Quantity, got {type(v)}")


def second_serializer(value: float) -> dict[str, float | str]:
    return {"value": value, "unit": "s"}


Seconds = Annotated[
    float,
    BeforeValidator(second_validator),
    PlainSerializer(second_serializer),
]
"""A time value in **seconds**.

If used to type hint a pydantic model field, 
will convert to seconds internally if given a `Quantity` with compatible time units.
If used to type hint a method argument, it is assumed to be a float in seconds - no 
conversion is performed inherently - be sure to use the 'second' wrapper if needed.
It can be found at modular_simulation.utils.wrappers

Serializes as `{"value": float, "unit": "s"}`.

float docstring:
"""


def per_second_validator(v: dict[str, float | str] | Quantity | float) -> float:
    # 1. convert dictionary input to quantity
    #   e.g. {"value": 10, "unit": "1/s"}
    if isinstance(v, dict):
        try:
            val = v["value"]
            unit = v.get("unit", "1/s")
            v = Quantity(float(val), str(unit))
        except Exception:
            raise ValueError(f"Invalid input dictionary v: {v}")
    # try to convert quantity, which was either parsed
    # from dict or passed directly, to 1/seconds and return its magnitude
    if isinstance(v, Quantity):
        if np.isscalar(v.value):
            if v.unit.is_equivalent("1/second"):
                return float(v.to_value("1/second"))
            else:
                raise UnitsError(f"Expected a per-time unit, got {v.unit}")
        else:
            raise ValueError(f"Expected a scalar Quantity, got {v}")
    # if not a quantity, assume it is already in 1/seconds as float
    elif not isinstance(v, Quantity):
        return float(v)
    raise TypeError(f"Expected float or Quantity, got {type(v)}")


def per_second_serializer(value: float) -> dict[str, float | str]:
    return {"value": value, "unit": "1/s"}


PerSeconds = Annotated[
    float,
    BeforeValidator(per_second_validator),
    PlainSerializer(per_second_serializer),
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
