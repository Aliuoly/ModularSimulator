"""Helper utilities for assembling runtime systems."""
from __future__ import annotations

from typing import Any, Type

from astropy.units import Quantity  # type: ignore

from modular_simulation.core.dynamic_model import DynamicModel
from modular_simulation.core.system import System
from modular_simulation.interfaces import ModelInterface


def create_system(
    *,
    dt: Quantity,
    dynamic_model: DynamicModel,
    model_interface: ModelInterface,
    system_class: Type[System] = System,
    use_numba: bool = False,
    numba_options: dict[str, Any] | None = None,
    solver_options: dict[str, Any] | None = None,
    record_history: bool = True,
) -> System:
    """Factory to build a :class:`System` with fresh component copies."""

    numba_opts = {"nopython": True, "cache": True} if numba_options is None else dict(numba_options)
    solver_opts = {"method": "LSODA"} if solver_options is None else dict(solver_options)

    copied_model = dynamic_model.model_copy()
    copied_interface = model_interface.model_copy()

    return system_class(
        dt=dt,
        dynamic_model=copied_model,
        model_interface=copied_interface,
        solver_options=solver_opts,
        record_history=record_history,
        use_numba=use_numba,
        numba_options=numba_opts,
    )


__all__ = ["create_system"]
