from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from modular_simulation.connection.connection_layer import (
    MacroStepBookkeeping,
    PicardIterationGateConfig,
    TransportAndMixingUpdate,
    run_macro_coupling_step,
)
from modular_simulation.connection.hydraulic_compile import solve_compiled_hydraulic_graph
from modular_simulation.connection.hydraulic_solver import HydraulicSolveResult
from modular_simulation.connection.network import ConnectionNetwork
from modular_simulation.connection.state import PortCondition


@dataclass(frozen=True)
class BarrierReconfigurationReport:
    queue_size_before: int
    queue_size_after: int
    applied_request_ids: tuple[str, ...]
    skipped_request_ids: tuple[str, ...]
    graph_revision: str | None


@dataclass(frozen=True)
class ConnectionRuntimeStepResult:
    macro_step_index: int
    macro_step_time_s: float
    simulation_time_s: float
    graph_revision: str | None
    bookkeeping: MacroStepBookkeeping
    barrier_reconfiguration: BarrierReconfigurationReport


class RuntimeHydraulicSolveStep(Protocol):
    def __call__(
        self,
        *,
        network: ConnectionNetwork,
        macro_step_time_s: float,
    ) -> HydraulicSolveResult: ...


class RuntimeTransportAndMixingStep(Protocol):
    def __call__(
        self,
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate: ...


class RuntimeBoundaryPropagationStep(Protocol):
    def __call__(
        self,
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> Mapping[str, PortCondition]: ...


class ConnectionRuntimeOrchestrator:
    def __init__(
        self,
        *,
        macro_step_hook: (Callable[[ConnectionNetwork, int, float], None] | None) = None,
        hydraulic_solve_step: RuntimeHydraulicSolveStep | None = None,
        transport_and_mixing_step: RuntimeTransportAndMixingStep | None = None,
        boundary_propagation_step: RuntimeBoundaryPropagationStep | None = None,
        picard_gate_config: PicardIterationGateConfig | None = None,
    ) -> None:
        self._macro_step_hook: Callable[[ConnectionNetwork, int, float], None] | None = (
            macro_step_hook
        )
        self._hydraulic_solve_step: RuntimeHydraulicSolveStep = (
            hydraulic_solve_step
            if hydraulic_solve_step is not None
            else self._default_hydraulic_solve_step
        )
        self._transport_and_mixing_step: RuntimeTransportAndMixingStep = (
            transport_and_mixing_step
            if transport_and_mixing_step is not None
            else self._default_transport_and_mixing_step
        )
        self._boundary_propagation_step: RuntimeBoundaryPropagationStep = (
            boundary_propagation_step
            if boundary_propagation_step is not None
            else self._default_boundary_propagation_step
        )
        self._picard_gate_config: PicardIterationGateConfig | None = picard_gate_config
        self._simulation_time_s: float = 0.0
        self._macro_step_index: int = 0
        self._macro_step_active: bool = False
        self._committed_idempotency_keys: set[str] = set()

    @property
    def simulation_time_s(self) -> float:
        return self._simulation_time_s

    @property
    def macro_step_index(self) -> int:
        return self._macro_step_index

    @property
    def macro_step_active(self) -> bool:
        return self._macro_step_active

    @property
    def committed_idempotency_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._committed_idempotency_keys))

    def set_runtime_state(
        self,
        *,
        simulation_time_s: float,
        macro_step_index: int,
        macro_step_active: bool,
        committed_idempotency_keys: Sequence[str],
    ) -> None:
        self._simulation_time_s = simulation_time_s
        self._macro_step_index = macro_step_index
        self._macro_step_active = macro_step_active
        self._committed_idempotency_keys = set(committed_idempotency_keys)

    def step(
        self, *, network: ConnectionNetwork, macro_step_time_s: float
    ) -> ConnectionRuntimeStepResult:
        if macro_step_time_s < 0.0 or not isfinite(macro_step_time_s):
            raise ValueError("macro_step_time_s must be finite and non-negative")
        if self._macro_step_active:
            raise RuntimeError(
                "runtime mutation barrier is locked during active macro-step execution"
            )

        barrier_report = self._apply_reconfiguration_batch(network=network)

        self._macro_step_active = True
        try:
            if self._macro_step_hook is not None:
                self._macro_step_hook(
                    network,
                    self._macro_step_index + 1,
                    self._simulation_time_s + macro_step_time_s,
                )
            macro_step_result = run_macro_coupling_step(
                macro_step_time_s=macro_step_time_s,
                macro_step_index=self._macro_step_index + 1,
                hydraulic_solve_step=lambda *, macro_step_time_s: self._hydraulic_solve_step(
                    network=network,
                    macro_step_time_s=macro_step_time_s,
                ),
                transport_update_step=(
                    lambda *, hydraulic_result, macro_step_time_s: self._transport_and_mixing_step(
                        network=network,
                        hydraulic_result=hydraulic_result,
                        macro_step_time_s=macro_step_time_s,
                    )
                ),
                boundary_propagation_step=(
                    lambda *,
                    hydraulic_result,
                    transport_and_mixing_update,
                    macro_step_time_s: self._boundary_propagation_step(
                        network=network,
                        hydraulic_result=hydraulic_result,
                        transport_and_mixing_update=transport_and_mixing_update,
                        macro_step_time_s=macro_step_time_s,
                    )
                ),
                picard_gate_config=self._picard_gate_config,
            )
            self._macro_step_index += 1
            self._simulation_time_s += macro_step_time_s
        finally:
            self._macro_step_active = False

        current_graph_revision = None
        if network.compiled is not None:
            current_graph_revision = str(network.compiled.graph_revision)

        return ConnectionRuntimeStepResult(
            macro_step_index=self._macro_step_index,
            macro_step_time_s=macro_step_time_s,
            simulation_time_s=self._simulation_time_s,
            graph_revision=current_graph_revision,
            bookkeeping=macro_step_result.bookkeeping,
            barrier_reconfiguration=BarrierReconfigurationReport(
                queue_size_before=barrier_report.queue_size_before,
                queue_size_after=barrier_report.queue_size_after,
                applied_request_ids=barrier_report.applied_request_ids,
                skipped_request_ids=barrier_report.skipped_request_ids,
                graph_revision=current_graph_revision,
            ),
        )

    def save_runtime_snapshot(self, *, network: ConnectionNetwork) -> Mapping[str, object]:
        if self._macro_step_active:
            raise RuntimeError(
                "runtime snapshot capture is unavailable during active macro-step execution"
            )
        compiled = network.compiled
        graph_revision = None if compiled is None else str(compiled.graph_revision)
        queue = network.queued_reconfigurations
        return {
            "schema": "connection_runtime_v1",
            "simulation_time_s": self._simulation_time_s,
            "macro_step_index": self._macro_step_index,
            "macro_step_active": self._macro_step_active,
            "graph_revision": graph_revision,
            "queued_request_ids": tuple(item.request_id for item in queue),
            "queued_request_count": len(queue),
            "committed_idempotency_keys": tuple(sorted(self._committed_idempotency_keys)),
        }

    def resume_from_snapshot(
        self, *, network: ConnectionNetwork, snapshot: Mapping[str, object]
    ) -> None:
        schema = snapshot.get("schema")
        if schema != "connection_runtime_v1":
            raise ValueError(
                "invalid runtime snapshot payload: expected field 'schema' == 'connection_runtime_v1'"
            )

        simulation_time = snapshot.get("simulation_time_s")
        if not isinstance(simulation_time, (float, int)):
            raise ValueError(
                "invalid runtime snapshot payload: field 'simulation_time_s' must be finite number"
            )
        simulation_time_s = float(simulation_time)
        if simulation_time_s < 0.0 or not isfinite(simulation_time_s):
            raise ValueError(
                "invalid runtime snapshot payload: field 'simulation_time_s' must be finite number"
            )

        macro_step_index = snapshot.get("macro_step_index")
        if not isinstance(macro_step_index, int) or macro_step_index < 0:
            raise ValueError(
                "invalid runtime snapshot payload: field 'macro_step_index' must be non-negative integer"
            )

        macro_step_active = snapshot.get("macro_step_active")
        if macro_step_active is not False:
            raise ValueError(
                "invalid runtime snapshot payload: field 'macro_step_active' must be false at barrier"
            )

        idempotency_keys = snapshot.get("committed_idempotency_keys", ())
        if isinstance(idempotency_keys, str) or not isinstance(idempotency_keys, Sequence):
            raise ValueError(
                "invalid runtime snapshot payload: field 'committed_idempotency_keys' must be sequence of strings"
            )
        normalized_keys: set[str] = set()
        for key in idempotency_keys:
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    "invalid runtime snapshot payload: field 'committed_idempotency_keys' must be sequence of strings"
                )
            normalized_keys.add(key.strip())

        del network
        self._simulation_time_s = simulation_time_s
        self._macro_step_index = macro_step_index
        self._macro_step_active = False
        self._committed_idempotency_keys = normalized_keys

    def _apply_reconfiguration_batch(
        self, *, network: ConnectionNetwork
    ) -> BarrierReconfigurationReport:
        network_snapshot = network.capture_reconfiguration_state()
        queued_requests = network.drain_reconfiguration_queue()
        if not queued_requests:
            compiled = network.compiled
            return BarrierReconfigurationReport(
                queue_size_before=0,
                queue_size_after=0,
                applied_request_ids=(),
                skipped_request_ids=(),
                graph_revision=None if compiled is None else compiled.graph_revision,
            )

        runtime_snapshot = (
            self._simulation_time_s,
            self._macro_step_index,
            set(self._committed_idempotency_keys),
        )

        applied_request_ids: list[str] = []
        skipped_request_ids: list[str] = []
        try:
            for request in queued_requests:
                idempotency_key = request.idempotency_key
                if (
                    idempotency_key is not None
                    and idempotency_key in self._committed_idempotency_keys
                ):
                    skipped_request_ids.append(request.request_id)
                    continue

                for mutation_index, mutation in enumerate(request.mutations, start=1):
                    try:
                        self._apply_mutation(network=network, mutation=mutation)
                    except Exception as error:
                        raise ValueError(
                            f"request '{request.request_id}' mutation {mutation_index} failed: {error}"
                        ) from error

                if idempotency_key is not None:
                    self._committed_idempotency_keys.add(idempotency_key)
                applied_request_ids.append(request.request_id)

            if applied_request_ids:
                try:
                    compiled = network.compile()
                except Exception as error:
                    raise ValueError(
                        f"queued reconfiguration compile validation failed: {error}"
                    ) from error
                graph_revision = compiled.graph_revision
            else:
                compiled = network.compiled
                graph_revision = None if compiled is None else compiled.graph_revision

            return BarrierReconfigurationReport(
                queue_size_before=len(queued_requests),
                queue_size_after=0,
                applied_request_ids=tuple(applied_request_ids),
                skipped_request_ids=tuple(skipped_request_ids),
                graph_revision=graph_revision,
            )
        except Exception as error:
            network.restore_reconfiguration_state(network_snapshot)
            self._simulation_time_s = runtime_snapshot[0]
            self._macro_step_index = runtime_snapshot[1]
            self._committed_idempotency_keys = runtime_snapshot[2]
            raise RuntimeError(f"reconfiguration transaction rollback: {error}") from error

    def _apply_mutation(
        self, *, network: ConnectionNetwork, mutation: Mapping[str, object]
    ) -> None:
        operation_value = mutation.get("operation")
        if not isinstance(operation_value, str) or not operation_value.strip():
            raise ValueError("invalid mutation payload: field 'operation' must be non-empty string")
        operation = operation_value.strip()

        if operation == "add_connection":
            source = self._require_non_empty_string(mutation, "source", operation)
            target = self._require_non_empty_string(mutation, "target", operation)
            edge_id_value = mutation.get("edge_id")
            if edge_id_value is None:
                _ = network.connect(source, target)
                return
            if not isinstance(edge_id_value, str) or not edge_id_value.strip():
                raise ValueError(
                    "operation 'add_connection' optional field 'edge_id' must be non-empty string"
                )
            _ = network.connect(source, target, edge_id=edge_id_value.strip())
            return

        if operation == "remove_connection":
            _ = network.remove_connection(
                edge_id=mutation.get("edge_id"),
                source=mutation.get("source"),
                target=mutation.get("target"),
            )
            return

        if operation == "rewire_connection":
            edge_id = self._require_non_empty_string(mutation, "edge_id", operation)
            source = self._require_non_empty_string(mutation, "source", operation)
            target = self._require_non_empty_string(mutation, "target", operation)
            network.rewire_connection(edge_id=edge_id, source=source, target=target)
            return

        if operation == "add_boundary_source":
            boundary_id = self._require_non_empty_string(mutation, "boundary_id", operation)
            port_name = mutation.get("port_name", "outlet")
            if not isinstance(port_name, str) or not port_name.strip():
                raise ValueError(
                    "operation 'add_boundary_source' optional field 'port_name' must be non-empty string"
                )
            network.add_boundary_source(boundary_id, port_name=port_name.strip())
            return

        if operation == "add_boundary_sink":
            boundary_id = self._require_non_empty_string(mutation, "boundary_id", operation)
            port_name = mutation.get("port_name", "inlet")
            if not isinstance(port_name, str) or not port_name.strip():
                raise ValueError(
                    "operation 'add_boundary_sink' optional field 'port_name' must be non-empty string"
                )
            network.add_boundary_sink(boundary_id, port_name=port_name.strip())
            return

        if operation == "remove_boundary":
            network.remove_boundary(boundary_id=mutation.get("boundary_id"))
            return

        if operation == "add_process_port":
            network.add_process_port(
                process_id=mutation.get("process_id"),
                direction=mutation.get("direction"),
                port_name=mutation.get("port_name"),
            )
            return

        if operation == "remove_process_port":
            network.remove_process_port(
                process_id=mutation.get("process_id"),
                direction=mutation.get("direction"),
                port_name=mutation.get("port_name"),
            )
            return

        if operation == "add_process":
            process_id = self._require_non_empty_string(mutation, "process_id", operation)
            inlet_ports = self._require_string_sequence(mutation, "inlet_ports", operation)
            outlet_ports = self._require_string_sequence(mutation, "outlet_ports", operation)
            network.add_process(process_id, inlet_ports=inlet_ports, outlet_ports=outlet_ports)
            return

        raise ValueError(f"unsupported reconfiguration operation '{operation}'")

    def _require_non_empty_string(
        self,
        mutation: Mapping[str, object],
        field_name: str,
        operation: str,
    ) -> str:
        value = mutation.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"operation '{operation}' requires non-empty string field '{field_name}'"
            )
        return value.strip()

    def _require_string_sequence(
        self,
        mutation: Mapping[str, object],
        field_name: str,
        operation: str,
    ) -> tuple[str, ...]:
        value = mutation.get(field_name)
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError(f"operation '{operation}' requires sequence field '{field_name}'")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"operation '{operation}' requires sequence field '{field_name}' with non-empty strings"
                )
            normalized.append(item.strip())
        return tuple(normalized)

    def _default_hydraulic_solve_step(
        self,
        *,
        network: ConnectionNetwork,
        macro_step_time_s: float,
    ) -> HydraulicSolveResult:
        del macro_step_time_s
        compiled = network.compiled
        if compiled is None or compiled.hydraulic is None:
            raise RuntimeError(
                "runtime macro-step is unavailable: compiled hydraulic graph is required"
            )
        return solve_compiled_hydraulic_graph(compiled.hydraulic)

    def _default_transport_and_mixing_step(
        self,
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        del network, hydraulic_result, macro_step_time_s
        return TransportAndMixingUpdate(transport_results={})

    def _default_boundary_propagation_step(
        self,
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> Mapping[str, PortCondition]:
        del network, hydraulic_result, transport_and_mixing_update, macro_step_time_s
        return {}


__all__ = [
    "BarrierReconfigurationReport",
    "ConnectionRuntimeOrchestrator",
    "ConnectionRuntimeStepResult",
    "RuntimeBoundaryPropagationStep",
    "RuntimeHydraulicSolveStep",
    "RuntimeTransportAndMixingStep",
]
