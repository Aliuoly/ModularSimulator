# pyright: reportExplicitAny=false
# pyright: reportAny=false
from typing import Any, Callable, TypeAlias, overload
import numpy as np
from numpy.typing import NDArray
from modular_simulation.utils.typing import StateValue

# 1. Define what a "value" looks like (numbers or arrays)
# If you use numpy, you can use npt.ArrayLike here for stricter checking
UnitLike: TypeAlias = UnitBase | str | Quantity
ValueType: TypeAlias = int | StateValue
KNOWN_GOOD = NDArray[np.float64] | float | int | complex

def _condition_arg(value: KNOWN_GOOD) -> KNOWN_GOOD: ...
def unit_scale_converter(val: KNOWN_GOOD) -> KNOWN_GOOD: ...

class UnitsError(Exception): ...

class UnitBase:
    """
    The base class for all units to hold the method definition.
    """
    def get_converter(
        self, other: UnitLike, equivalencies: list[Any] | None = None
    ) -> Callable[[ValueType], ValueType]: ...

    # Including .to() is usually helpful as it's the immediate sibling of get_converter
    def to(
        self,
        other: UnitLike,
        value: ValueType = ...,
        equivalencies: list[Any] | None = None,
    ) -> ValueType: ...
    def is_equivalent(self, other: UnitLike, equivalencies: list[Any] = []) -> bool: ...

class Unit(UnitBase):
    def __init__(
        self,
        st: str,
        represents: UnitLike | None = None,
        doc: str | None = None,
        format: dict[str, Any] | None = None,
        namespace: dict[str, Any] | None = None,
    ) -> None: ...
    def __mul__(self, other: ValueType) -> Quantity: ...
    def __rmul__(self, other: ValueType) -> Quantity: ...
    @overload
    def __truediv__(self, other: UnitBase) -> Unit: ...
    @overload
    def __truediv__(self, other: Quantity) -> Quantity: ...
    def __truediv__(self, other: ValueType) -> Quantity: ...
    def __pow__(self, other: Any) -> Unit: ...

class Quantity:
    """
    A Quantity represents a number with an associated unit.
    """

    value: ValueType
    unit: UnitBase

    def __init__(
        self,
        value: ValueType,
        unit: UnitLike | None = None,
        dtype: Any = ...,
        copy: bool = ...,
    ) -> None: ...

    # -- Core Conversion Methods --
    def to(self, unit: UnitLike, equivalencies: Any | None = None) -> Quantity:
        """Returns a new Quantity in the specified unit."""
        ...

    def to_value(self, unit: UnitLike, equivalencies: Any | None = None) -> ValueType:
        """Returns the raw value (float/array) in the specified unit."""
        ...

    # -- Math Operations --
    def __add__(self, other: Any) -> Quantity: ...
    def __sub__(self, other: Any) -> Quantity: ...
    def __mul__(self, other: Any) -> Quantity: ...
    def __truediv__(self, other: Any) -> Quantity: ...
    def __pow__(self, other: Any) -> Quantity: ...
    def __neg__(self) -> Quantity: ...
