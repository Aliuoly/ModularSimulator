"""Compatibility helpers for unit-aware components.

This module provides ``Unit`` and ``Quantity`` shims when ``astropy`` is not
available so that the simulation framework can still operate with basic unit
awareness in restricted environments (such as the execution sandbox used for
these exercises).  When ``astropy`` is installed the real implementations are
re-exported transparently.
"""
from __future__ import annotations

from typing import Callable

try:  # pragma: no cover - exercised only when astropy is installed
    from astropy.units import Unit as AstropyUnit  # type: ignore
    from astropy.units import Quantity as AstropyQuantity  # type: ignore
    import astropy.units as astropy_units  # type: ignore

    Unit = AstropyUnit
    Quantity = AstropyQuantity
    units = astropy_units
except ModuleNotFoundError:  # pragma: no cover - exercised in the sandbox
    class Unit:
        """Minimal stand-in for :class:`astropy.units.Unit`.

        The shim tracks only the textual definition of the unit.  Two units are
        considered equivalent when they share the same definition string.  This
        is sufficient for the existing examples where a single, consistent
        string is used for each physical quantity.
        """

        __slots__ = ("_definition",)

        def __init__(self, definition: str | None = None) -> None:
            self._definition = (definition or "dimensionless").strip()

        def __repr__(self) -> str:  # pragma: no cover - debugging helper
            return f"Unit({self._definition!r})"

        def __str__(self) -> str:  # pragma: no cover - debugging helper
            return self._definition or "dimensionless"

        # ``astropy`` exposes ``PhysicalType`` based comparisons; here we simply
        # compare the stored definition.
        def is_equivalent(self, other: "Unit") -> bool:
            return isinstance(other, Unit) and self._definition == other._definition

        def get_converter(self, other: "Unit") -> Callable[[float], float]:
            if not self.is_equivalent(other):
                raise ValueError(
                    f"Units '{self}' and '{other}' are not compatible in the shim implementation."
                )
            return lambda value: value

    class Quantity:
        """Simple quantity that mimics the subset of ``astropy`` used here."""

        __slots__ = ("value", "unit")

        def __init__(self, value: float, unit: Unit) -> None:
            self.value = value
            self.unit = unit

        def to(self, unit: Unit) -> "Quantity":
            converter = self.unit.get_converter(unit)
            return Quantity(converter(self.value), unit)

        # ``astropy``'s quantities expose ``is_equivalent`` via their unit; the
        # code base only relies on the attributes defined above.

    class _UnitNamespace:
        dimensionless_unscaled = Unit("dimensionless")

        def __getattr__(self, name: str) -> Unit:  # pragma: no cover - debug helper
            raise AttributeError(name)

    units = _UnitNamespace()

__all__ = ["Unit", "Quantity", "units"]
