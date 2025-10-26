"""Helper utilities for assembling runtime systems."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Type

from astropy.units import Quantity  # type: ignore

from modular_simulation.core.dynamic_model import DynamicModel
from modular_simulation.core.system import System
from modular_simulation.interfaces import (
    CalculationBase,
    ControllerBase,
    SensorBase,
)


def create_system(
    *,
    dt: Quantity,
    dynamic_model: DynamicModel,
    sensors: Sequence[SensorBase] | None = None,
    calculations: Sequence[CalculationBase] | None = None,
    controllers: Sequence[ControllerBase] | None = None,
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
    copied_sensors = [] if sensors is None else [sensor.model_copy() for sensor in sensors]
    copied_calculations = (
        [] if calculations is None else [calculation.model_copy() for calculation in calculations]
    )
    copied_controllers = (
        [] if controllers is None else [controller.model_copy() for controller in controllers]
    )

    return system_class(
        dt=dt,
        dynamic_model=copied_model,
        sensors=copied_sensors,
        calculations=copied_calculations,
        controllers=copied_controllers,
        solver_options=solver_opts,
        record_history=record_history,
        use_numba=use_numba,
        numba_options=numba_opts,
    )


__all__ = ["create_system"]
