from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ElementOutputSpec",
    "ElementParameterSpec",
    "ElementUnknownSpec",
    "HydraulicElement",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module("modular_simulation.connection.hydraulic_element")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
