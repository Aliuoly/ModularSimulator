from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from astropy.units import UnitBase

from modular_simulation.connection.state import MaterialState, PortCondition
from modular_simulation.measurables.process_model import ProcessModel
from modular_simulation.utils.typing import StateValue


@runtime_checkable
class ProcessInterface(Protocol):
    @property
    def state_names(self) -> tuple[str, ...]: ...

    @property
    def input_port_names(self) -> tuple[str, ...]: ...

    @property
    def output_port_names(self) -> tuple[str, ...]: ...

    def get_state(self, state_name: str) -> StateValue: ...

    def set_state(self, state_name: str, value: StateValue) -> None: ...

    def make_state_getter(
        self, state_name: str, target_unit: UnitBase | str | None = None
    ) -> Callable[[], StateValue]: ...

    def make_state_setter(
        self, state_name: str, source_unit: UnitBase | str | None = None
    ) -> Callable[[StateValue], None]: ...

    def read_port_conditions(self) -> Mapping[str, PortCondition]: ...

    def write_port_conditions(self, port_conditions: Mapping[str, PortCondition]) -> None: ...

    def get_port_condition(self, port_name: str) -> PortCondition: ...

    def map_port_conditions_to_balance_terms(
        self,
        *,
        incoming_material_states: Mapping[str, MaterialState] | None = None,
    ) -> Mapping[str, "ProcessBalanceTerm"]: ...


PortRole = Literal["inlet", "outlet"]


@dataclass(frozen=True)
class ProcessBalanceTerm:
    port_name: str
    port_role: PortRole
    through_molar_flow_rate: float
    inlet_molar_flow_rate: float
    outlet_molar_flow_rate: float
    solved_pressure: float
    solved_temperature: float
    solved_mole_fractions: tuple[float, ...]
    incoming_pressure: float
    incoming_temperature: float
    incoming_mole_fractions: tuple[float, ...]


@dataclass
class ProcessModelAdapter(ProcessInterface):
    process_model: ProcessModel
    _port_conditions: dict[str, PortCondition] = field(default_factory=dict, init=False)

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(self.process_model.state_list)

    @property
    def input_port_names(self) -> tuple[str, ...]:
        return tuple(self.process_model.list_input_streams())

    @property
    def output_port_names(self) -> tuple[str, ...]:
        return tuple(self.process_model.list_output_streams())

    def get_state(self, state_name: str) -> StateValue:
        self._validate_state_name(state_name)
        return self.process_model.state_getter(state_name)

    def set_state(self, state_name: str, value: StateValue) -> None:
        self._validate_state_name(state_name)
        setattr(self.process_model, state_name, value)

    def make_state_getter(
        self, state_name: str, target_unit: UnitBase | str | None = None
    ) -> Callable[[], StateValue]:
        self._validate_state_name(state_name)
        return self.process_model.make_converted_getter(state_name, target_unit)

    def make_state_setter(
        self, state_name: str, source_unit: UnitBase | str | None = None
    ) -> Callable[[StateValue], None]:
        self._validate_state_name(state_name)
        return self.process_model.make_converted_setter(state_name, source_unit)

    def read_port_conditions(self) -> Mapping[str, PortCondition]:
        return dict(self._port_conditions)

    def write_port_conditions(self, port_conditions: Mapping[str, PortCondition]) -> None:
        self._port_conditions = dict(port_conditions)

    def get_port_condition(self, port_name: str) -> PortCondition:
        port_condition = self._port_conditions.get(port_name)
        if port_condition is None:
            available = sorted(self._port_conditions.keys())
            raise KeyError(
                f"Unknown port condition '{port_name}'. Available port keys: {available}"
            )
        return port_condition

    def map_port_conditions_to_balance_terms(
        self,
        *,
        incoming_material_states: Mapping[str, MaterialState] | None = None,
    ) -> Mapping[str, ProcessBalanceTerm]:
        incoming_by_port = dict(incoming_material_states or {})
        mapped_terms: dict[str, ProcessBalanceTerm] = {}

        for port_name in sorted(self._port_conditions):
            port_role = self._port_role(port_name)
            port_condition = self._port_conditions[port_name]

            flow = float(port_condition.through_molar_flow_rate)
            normalized_inlet_flow = flow if port_role == "inlet" else -flow
            inlet_flow = max(normalized_inlet_flow, 0.0)
            outlet_flow = max(-normalized_inlet_flow, 0.0)

            solved_state = port_condition.state
            incoming_state = incoming_by_port.get(port_name, solved_state)
            mapped_terms[port_name] = ProcessBalanceTerm(
                port_name=port_name,
                port_role=port_role,
                through_molar_flow_rate=flow,
                inlet_molar_flow_rate=inlet_flow,
                outlet_molar_flow_rate=outlet_flow,
                solved_pressure=solved_state.pressure,
                solved_temperature=solved_state.temperature,
                solved_mole_fractions=solved_state.mole_fractions,
                incoming_pressure=incoming_state.pressure,
                incoming_temperature=incoming_state.temperature,
                incoming_mole_fractions=incoming_state.mole_fractions,
            )

        return mapped_terms

    def _validate_state_name(self, state_name: str) -> None:
        if state_name in self.process_model.state_metadata_dict:
            return
        available = sorted(self.process_model.state_metadata_dict.keys())
        raise KeyError(f"Unknown state '{state_name}'. Available states: {available}")

    def _port_role(self, port_name: str) -> PortRole:
        if port_name in self.input_port_names:
            return "inlet"
        if port_name in self.output_port_names:
            return "outlet"
        available = sorted(set(self.input_port_names) | set(self.output_port_names))
        raise KeyError(f"Unknown process port '{port_name}'. Available process ports: {available}")


__all__ = ["ProcessBalanceTerm", "ProcessInterface", "ProcessModelAdapter"]
