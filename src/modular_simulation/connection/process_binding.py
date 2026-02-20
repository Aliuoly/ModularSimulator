from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


class BindingError(Exception):
    """Deterministic, actionable error for invalid process bindings."""


OutletRole = Literal["inlet", "outlet"]


@dataclass(frozen=True)
class OutletBinding:
    role: OutletRole
    flow: float
    composition: tuple[float, ...]

    @property
    def normalized_inlet_flow(self) -> float:
        return self.flow if self.role == "inlet" else -self.flow

    @property
    def inlet_flow(self) -> float:
        return max(self.normalized_inlet_flow, 0.0)

    @property
    def outlet_flow(self) -> float:
        return max(-self.normalized_inlet_flow, 0.0)

    def as_dict(self) -> dict[str, float | tuple[float, ...] | OutletRole]:
        return {
            "role": self.role,
            "flow": self.flow,
            "composition": self.composition,
            "normalized_inlet_flow": self.normalized_inlet_flow,
            "inlet_flow": self.inlet_flow,
            "outlet_flow": self.outlet_flow,
        }


class ProcessBinding:
    """Explicit inlet/outlet binding abstraction.

    - Ports are defined as two ordered lists: inlets and outlets.
    - Inlet bindings are set via set_inlet or bind_inlets.
    - Outlet bindings are set via bind_outlets and retrieved via get_outlet.
    - validate() ensures all ports have concrete bindings.
    """

    _required_outlet_fields: tuple[str, ...] = ("role", "flow", "composition")

    def __init__(self, inlet_ports: list[str], outlet_ports: list[str]):
        if len(inlet_ports) != len(outlet_ports):
            raise BindingError(
                f"Inlet/outlet port count mismatch: {len(inlet_ports)} inlets, {len(outlet_ports)} outlets."
            )
        self._inlet_ports: list[str] = list(inlet_ports)
        self._outlet_ports: list[str] = list(outlet_ports)
        self._inlet_bindings: dict[str, float | None] = {p: None for p in inlet_ports}
        self._outlet_bindings: dict[str, OutletBinding | None] = {p: None for p in outlet_ports}

    # Inlet API
    def set_inlet(self, port: str, value: float) -> None:
        if port not in self._inlet_bindings:
            raise BindingError(f"Unknown inlet port '{port}'. Expected one of: {self._inlet_ports}")
        self._inlet_bindings[port] = float(value)

    def bind_inlets(self, bindings: Mapping[str, float]) -> None:
        missing = [p for p in self._inlet_bindings if p not in bindings]
        extra = [p for p in bindings if p not in self._inlet_bindings]
        if missing:
            raise BindingError(f"Missing inlet bindings for ports: {sorted(missing)}")
        if extra:
            raise BindingError(f"Unknown inlet bindings for ports: {sorted(extra)}")
        for port, val in bindings.items():
            self.set_inlet(port, val)
        self._ensure_inlets_bound()

    # Outlet API
    def get_outlet(self, port: str) -> dict[str, float | tuple[float, ...] | OutletRole]:
        if port not in self._outlet_bindings:
            raise BindingError(
                f"Unknown outlet port '{port}'. Expected one of: {self._outlet_ports}"
            )
        val = self._outlet_bindings[port]
        if val is None:
            raise BindingError(f"Outlet '{port}' has no binding yet.")
        return val.as_dict()

    def bind_outlets(self, bindings: Mapping[str, Mapping[str, object]]) -> None:
        missing = [p for p in self._outlet_bindings if p not in bindings]
        extra = [p for p in bindings if p not in self._outlet_bindings]
        if missing:
            raise BindingError(f"Missing outlet bindings for ports: {sorted(missing)}")
        if extra:
            raise BindingError(f"Unknown outlet bindings for ports: {sorted(extra)}")

        parsed: dict[str, OutletBinding] = {}
        composition_lengths: dict[str, int] = {}
        for port in self._outlet_ports:
            parsed_binding = self._parse_outlet_binding(port, bindings[port])
            parsed[port] = parsed_binding
            composition_lengths[port] = len(parsed_binding.composition)

        unique_lengths = set(composition_lengths.values())
        if len(unique_lengths) > 1:
            raise BindingError(
                f"Incompatible composition lengths across outlet bindings: {composition_lengths}"
            )

        for port in self._outlet_ports:
            self._outlet_bindings[port] = parsed[port]
        self._ensure_outlets_bound()

    # Validation helpers
    def validate(self) -> None:
        self._ensure_inlets_bound()
        self._ensure_outlets_bound()

    def _ensure_inlets_bound(self) -> None:
        unbound = [p for p, v in self._inlet_bindings.items() if v is None]
        if unbound:
            raise BindingError(f"Unbound inlet ports: {unbound}")

    def _ensure_outlets_bound(self) -> None:
        unbound = [p for p, v in self._outlet_bindings.items() if v is None]
        if unbound:
            raise BindingError(f"Unbound outlet ports: {unbound}")

    def _parse_outlet_binding(self, port: str, binding: Mapping[str, object]) -> OutletBinding:
        keys = set(binding.keys())
        required_keys = {"role", "flow", "composition"}
        missing_fields = sorted(required_keys - keys)
        if missing_fields:
            raise BindingError(f"Missing outlet binding field(s) for '{port}': {missing_fields}")

        unknown_fields = sorted(keys - required_keys)
        if unknown_fields:
            raise BindingError(f"Unknown outlet binding field(s) for '{port}': {unknown_fields}")

        raw_role = binding["role"]
        if raw_role not in ("inlet", "outlet"):
            raise BindingError(
                f"Invalid outlet binding role for '{port}': {raw_role!r}. Expected one of: ['inlet', 'outlet']"
            )

        raw_composition = binding["composition"]
        if not isinstance(raw_composition, Sequence) or isinstance(raw_composition, (str, bytes)):
            raise BindingError(
                f"Invalid outlet binding composition for '{port}': expected a sequence of numeric values."
            )

        composition_values: list[float] = []
        for value in raw_composition:
            if not isinstance(value, (int, float)):
                raise BindingError(
                    f"Invalid outlet binding composition for '{port}': expected numeric values."
                )
            composition_values.append(float(value))
        composition = tuple(composition_values)
        if len(composition) == 0:
            raise BindingError(
                f"Invalid outlet binding composition for '{port}': must not be empty."
            )

        raw_flow = binding["flow"]
        if not isinstance(raw_flow, (int, float)):
            raise BindingError(f"Invalid outlet binding flow for '{port}': expected numeric value.")

        return OutletBinding(
            role=raw_role,
            flow=float(raw_flow),
            composition=composition,
        )

    # Convenience for chaining in tests
    def set_inlet_and_validate(self, port: str, value: float) -> None:
        self.set_inlet(port, value)
        self.validate()
