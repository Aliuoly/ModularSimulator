"""Process-facing binding exports after the connection hard switch.

Legacy adapter/manual-wiring interfaces were intentionally removed.
Use `ConnectionNetwork.compile().process_bindings[...]` and the `ProcessBinding`
API for all active connection authoring flows.
"""

from __future__ import annotations

from modular_simulation.connection.process_binding import (
    BindingError,
    OutletBinding,
    OutletRole,
    ProcessBinding,
)

PortRole = OutletRole

__all__ = [
    "BindingError",
    "OutletBinding",
    "OutletRole",
    "PortRole",
    "ProcessBinding",
]
