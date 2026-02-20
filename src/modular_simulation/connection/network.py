from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from modular_simulation.connection.hydraulic_compile import (
    CompiledHydraulicGraph,
    HydraulicCompileLifecycle,
)
from modular_simulation.connection.hydraulic_solver import HydraulicSystemDefinition
from modular_simulation.connection.process_binding import ProcessBinding
from modular_simulation.connection.topology import (
    BoundaryNode,
    ConnectionEdge,
    PortReference,
    ProcessNode,
    TopologyGraph,
)


@dataclass(frozen=True)
class CompiledConnectionNetwork:
    graph_revision: str
    topology: TopologyGraph
    process_bindings: Mapping[str, ProcessBinding]
    hydraulic: CompiledHydraulicGraph | None


@dataclass(frozen=True)
class _BoundarySpec:
    boundary_id: str
    direction: Literal["source", "sink"]
    port_name: str


@dataclass(frozen=True)
class _ProcessSpec:
    process_id: str
    inlet_ports: tuple[str, ...]
    outlet_ports: tuple[str, ...]


@dataclass(frozen=True)
class _QueuedReconfigurationTransaction:
    request_id: str
    idempotency_key: str | None
    mutations: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _NetworkReconfigurationState:
    process_specs: dict[str, _ProcessSpec]
    boundary_specs: dict[str, _BoundarySpec]
    edges: list[ConnectionEdge]
    edge_lookup: dict[tuple[str, str, str, str], str]
    next_edge_index: int
    next_reconfiguration_index: int
    next_revision_index: int
    queued_reconfigurations: list[_QueuedReconfigurationTransaction]
    compiled: CompiledConnectionNetwork | None


class _HydraulicSystemBuilder(Protocol):
    def __call__(self, topology: TopologyGraph) -> HydraulicSystemDefinition: ...


class _RuntimeOrchestrator(Protocol):
    def step(self, *, network: "ConnectionNetwork", macro_step_time_s: float) -> object: ...

    def save_runtime_snapshot(self, *, network: "ConnectionNetwork") -> Mapping[str, object]: ...

    def resume_from_snapshot(
        self, *, network: "ConnectionNetwork", snapshot: Mapping[str, object]
    ) -> None: ...


class ConnectionNetwork:
    def __init__(
        self,
        *,
        compile_lifecycle: HydraulicCompileLifecycle | None = None,
        hydraulic_system_builder: _HydraulicSystemBuilder | None = None,
        runtime_orchestrator: _RuntimeOrchestrator | None = None,
    ):
        self._compile_lifecycle: HydraulicCompileLifecycle | None = compile_lifecycle
        self._hydraulic_system_builder: _HydraulicSystemBuilder | None = hydraulic_system_builder
        self._runtime_orchestrator: _RuntimeOrchestrator | None = runtime_orchestrator

        self._process_specs: dict[str, _ProcessSpec] = {}
        self._boundary_specs: dict[str, _BoundarySpec] = {}
        self._edges: list[ConnectionEdge] = []
        self._edge_lookup: dict[tuple[str, str, str, str], str] = {}
        self._next_edge_index: int = 1
        self._next_reconfiguration_index: int = 1
        self._next_revision_index: int = 1
        self._queued_reconfigurations: list[_QueuedReconfigurationTransaction] = []
        self._compiled: CompiledConnectionNetwork | None = None

    def add_process(
        self,
        process_id: str,
        *,
        inlet_ports: Sequence[str],
        outlet_ports: Sequence[str],
    ) -> None:
        self._validate_node_id(process_id)
        self._ensure_node_id_is_available(process_id)

        normalized_inlets = self._normalize_port_names(
            node_id=process_id,
            port_names=inlet_ports,
            role="inlet",
        )
        normalized_outlets = self._normalize_port_names(
            node_id=process_id,
            port_names=outlet_ports,
            role="outlet",
        )
        self._process_specs[process_id] = _ProcessSpec(
            process_id=process_id,
            inlet_ports=normalized_inlets,
            outlet_ports=normalized_outlets,
        )

    def add_boundary_source(self, boundary_id: str, *, port_name: str = "outlet") -> None:
        self._add_boundary(boundary_id=boundary_id, direction="source", port_name=port_name)

    def add_boundary_sink(self, boundary_id: str, *, port_name: str = "inlet") -> None:
        self._add_boundary(boundary_id=boundary_id, direction="sink", port_name=port_name)

    def connect(self, source: str, target: str, *, edge_id: str | None = None) -> str:
        source_ref = self._parse_endpoint(source)
        target_ref = self._parse_endpoint(target)

        edge_key = (
            source_ref.node_id,
            source_ref.port_name,
            target_ref.node_id,
            target_ref.port_name,
        )
        existing_edge_id = self._edge_lookup.get(edge_key)
        if existing_edge_id is not None:
            edge_name = (
                f"{source_ref.node_id}.{source_ref.port_name}"
                f"->{target_ref.node_id}.{target_ref.port_name}"
            )
            raise ValueError(
                f"duplicate connection '{edge_name}' already exists as edge_id '{existing_edge_id}'"
            )

        selected_edge_id = edge_id or f"edge_{self._next_edge_index:04d}"
        self._validate_edge_id(selected_edge_id)
        if any(existing.edge_id == selected_edge_id for existing in self._edges):
            raise ValueError(f"duplicate edge id '{selected_edge_id}' is not allowed")

        self._edges.append(
            ConnectionEdge(
                edge_id=selected_edge_id,
                source=source_ref,
                target=target_ref,
            )
        )
        self._edge_lookup[edge_key] = selected_edge_id
        self._next_edge_index += 1
        return selected_edge_id

    def compile(self) -> CompiledConnectionNetwork:
        topology = TopologyGraph(
            process_nodes=tuple(
                ProcessNode(
                    process_id=spec.process_id,
                    inlet_ports=spec.inlet_ports,
                    outlet_ports=spec.outlet_ports,
                )
                for spec in sorted(self._process_specs.values(), key=lambda item: item.process_id)
            ),
            boundary_nodes=tuple(
                BoundaryNode(
                    boundary_id=spec.boundary_id,
                    port_name=spec.port_name,
                    direction=spec.direction,
                )
                for spec in sorted(self._boundary_specs.values(), key=lambda item: item.boundary_id)
            ),
            junction_nodes=(),
            edges=tuple(self._edges),
        )

        bindings: dict[str, ProcessBinding] = {
            spec.process_id: ProcessBinding(
                inlet_ports=list(spec.inlet_ports),
                outlet_ports=list(spec.outlet_ports),
            )
            for spec in sorted(self._process_specs.values(), key=lambda item: item.process_id)
        }

        graph_revision = f"graph_rev_{self._next_revision_index:04d}"
        self._next_revision_index += 1

        hydraulic: CompiledHydraulicGraph | None = None
        if self._compile_lifecycle is not None or self._hydraulic_system_builder is not None:
            if self._compile_lifecycle is None or self._hydraulic_system_builder is None:
                raise RuntimeError(
                    "hydraulic compile delegation requires both compile_lifecycle and hydraulic_system_builder"
                )
            system = self._hydraulic_system_builder(topology)
            hydraulic = self._compile_lifecycle.compile(
                system=system, graph_revision=graph_revision
            )

        self._compiled = CompiledConnectionNetwork(
            graph_revision=graph_revision,
            topology=topology,
            process_bindings=bindings,
            hydraulic=hydraulic,
        )
        return self._compiled

    def step(self, *, macro_step_time_s: float) -> object:
        if self._runtime_orchestrator is None:
            raise RuntimeError(
                "runtime step is unavailable: runtime orchestrator not configured for ConnectionNetwork"
            )
        return self._runtime_orchestrator.step(network=self, macro_step_time_s=macro_step_time_s)

    def queue_reconfiguration(self, request: object) -> str:
        if not isinstance(request, Mapping):
            raise ValueError(
                "invalid reconfiguration request: expected mapping with non-empty string field 'operation'"
            )
        request_mapping = cast(Mapping[object, object], request)
        operation = request_mapping.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError(
                "invalid reconfiguration request: expected mapping with non-empty string field 'operation'"
            )

        idempotency_key_raw = request_mapping.get("idempotency_key")
        idempotency_key: str | None
        if idempotency_key_raw is None:
            idempotency_key = None
        elif isinstance(idempotency_key_raw, str) and idempotency_key_raw.strip():
            idempotency_key = idempotency_key_raw.strip()
        else:
            raise ValueError(
                "invalid reconfiguration request: optional field 'idempotency_key' must be a non-empty string"
            )

        mutations_field = request_mapping.get("mutations")
        mutations: tuple[dict[str, object], ...]
        if mutations_field is None:
            mutations = (self._normalize_reconfiguration_mutation(request_mapping),)
        else:
            if isinstance(mutations_field, str) or not isinstance(mutations_field, Sequence):
                raise ValueError(
                    "invalid reconfiguration request: field 'mutations' must be a non-empty sequence of mapping objects"
                )
            normalized_mutations: list[dict[str, object]] = []
            for mutation in mutations_field:
                if not isinstance(mutation, Mapping):
                    raise ValueError(
                        "invalid reconfiguration request: each mutation must be a mapping with non-empty string field 'operation'"
                    )
                normalized_mutations.append(
                    self._normalize_reconfiguration_mutation(
                        cast(Mapping[object, object], mutation)
                    )
                )
            if not normalized_mutations:
                raise ValueError(
                    "invalid reconfiguration request: field 'mutations' must be a non-empty sequence of mapping objects"
                )
            mutations = tuple(normalized_mutations)

        request_id = f"rq_{self._next_reconfiguration_index:04d}"
        self._next_reconfiguration_index += 1
        self._queued_reconfigurations.append(
            _QueuedReconfigurationTransaction(
                request_id=request_id,
                idempotency_key=idempotency_key,
                mutations=mutations,
            )
        )
        return request_id

    def save_runtime_snapshot(self) -> Mapping[str, object]:
        if self._runtime_orchestrator is None:
            raise RuntimeError(
                "runtime snapshot is unavailable: runtime orchestrator not configured for ConnectionNetwork"
            )
        return self._runtime_orchestrator.save_runtime_snapshot(network=self)

    def resume_from_snapshot(self, *, snapshot: object) -> None:
        if not isinstance(snapshot, Mapping):
            raise ValueError("invalid snapshot payload: expected mapping")
        if self._runtime_orchestrator is None:
            raise RuntimeError(
                "runtime resume is unavailable: runtime orchestrator not configured for ConnectionNetwork"
            )
        self._runtime_orchestrator.resume_from_snapshot(network=self, snapshot=snapshot)

    def _add_boundary(
        self,
        *,
        boundary_id: str,
        direction: Literal["source", "sink"],
        port_name: str,
    ) -> None:
        self._validate_node_id(boundary_id)
        self._ensure_node_id_is_available(boundary_id)
        normalized_port = self._normalize_port_name(port_name)
        self._boundary_specs[boundary_id] = _BoundarySpec(
            boundary_id=boundary_id,
            direction=direction,
            port_name=normalized_port,
        )

    def _remove_boundary(self, *, boundary_id: object) -> None:
        if not isinstance(boundary_id, str) or not boundary_id.strip():
            raise ValueError("remove_boundary requires non-empty string field 'boundary_id'")
        boundary_id = boundary_id.strip()
        if boundary_id not in self._boundary_specs:
            raise ValueError(f"unknown boundary '{boundary_id}'")

        incident_edges = tuple(
            sorted(
                edge.edge_id
                for edge in self._edges
                if edge.source.node_id == boundary_id or edge.target.node_id == boundary_id
            )
        )
        if incident_edges:
            raise ValueError(
                f"cannot remove boundary '{boundary_id}': connected edges exist {incident_edges!r}"
            )
        del self._boundary_specs[boundary_id]

    def _add_process_port(
        self,
        *,
        process_id: object,
        direction: object,
        port_name: object,
    ) -> None:
        if not isinstance(process_id, str) or not process_id.strip():
            raise ValueError("add_process_port requires non-empty string field 'process_id'")
        process_id = process_id.strip()
        spec = self._process_specs.get(process_id)
        if spec is None:
            raise ValueError(f"unknown process '{process_id}'")

        if not isinstance(direction, str) or direction not in ("inlet", "outlet"):
            raise ValueError(
                "add_process_port requires field 'direction' to be 'inlet' or 'outlet'"
            )
        normalized_port_name = self._normalize_port_name(port_name)

        known_ports = set(spec.inlet_ports) | set(spec.outlet_ports)
        if normalized_port_name in known_ports:
            raise ValueError(
                f"duplicate process endpoint '{process_id}.{normalized_port_name}' is not allowed"
            )

        if direction == "inlet":
            self._process_specs[process_id] = _ProcessSpec(
                process_id=spec.process_id,
                inlet_ports=(*spec.inlet_ports, normalized_port_name),
                outlet_ports=spec.outlet_ports,
            )
            return

        self._process_specs[process_id] = _ProcessSpec(
            process_id=spec.process_id,
            inlet_ports=spec.inlet_ports,
            outlet_ports=(*spec.outlet_ports, normalized_port_name),
        )

    def _remove_process_port(
        self,
        *,
        process_id: object,
        direction: object,
        port_name: object,
    ) -> None:
        if not isinstance(process_id, str) or not process_id.strip():
            raise ValueError("remove_process_port requires non-empty string field 'process_id'")
        process_id = process_id.strip()
        spec = self._process_specs.get(process_id)
        if spec is None:
            raise ValueError(f"unknown process '{process_id}'")

        if not isinstance(direction, str) or direction not in ("inlet", "outlet"):
            raise ValueError(
                "remove_process_port requires field 'direction' to be 'inlet' or 'outlet'"
            )
        normalized_port_name = self._normalize_port_name(port_name)
        target_ports = spec.inlet_ports if direction == "inlet" else spec.outlet_ports
        if normalized_port_name not in target_ports:
            raise ValueError(
                f"unknown process endpoint '{process_id}.{normalized_port_name}' for direction '{direction}'"
            )

        incident_edges = tuple(
            sorted(
                edge.edge_id
                for edge in self._edges
                if (edge.source.node_id, edge.source.port_name)
                == (process_id, normalized_port_name)
                or (edge.target.node_id, edge.target.port_name)
                == (process_id, normalized_port_name)
            )
        )
        if incident_edges:
            raise ValueError(
                f"cannot remove process endpoint '{process_id}.{normalized_port_name}': connected edges exist {incident_edges!r}"
            )

        if direction == "inlet":
            updated_inlets = tuple(
                item for item in spec.inlet_ports if item != normalized_port_name
            )
            self._process_specs[process_id] = _ProcessSpec(
                process_id=spec.process_id,
                inlet_ports=updated_inlets,
                outlet_ports=spec.outlet_ports,
            )
            return

        updated_outlets = tuple(item for item in spec.outlet_ports if item != normalized_port_name)
        self._process_specs[process_id] = _ProcessSpec(
            process_id=spec.process_id,
            inlet_ports=spec.inlet_ports,
            outlet_ports=updated_outlets,
        )

    def _remove_connection(
        self,
        *,
        edge_id: object | None = None,
        source: object | None = None,
        target: object | None = None,
    ) -> str:
        if edge_id is not None:
            if not isinstance(edge_id, str) or not edge_id.strip():
                raise ValueError("remove_connection field 'edge_id' must be a non-empty string")
            normalized_edge_id = edge_id.strip()
            for index, edge in enumerate(self._edges):
                if edge.edge_id != normalized_edge_id:
                    continue
                del self._edges[index]
                edge_key = (
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name,
                )
                _ = self._edge_lookup.pop(edge_key, None)
                return normalized_edge_id
            raise ValueError(f"unknown connection edge_id '{normalized_edge_id}'")

        if source is None or target is None:
            raise ValueError(
                "remove_connection requires either field 'edge_id' or both fields 'source' and 'target'"
            )

        source_ref = self._parse_endpoint(source)
        target_ref = self._parse_endpoint(target)
        edge_key = (
            source_ref.node_id,
            source_ref.port_name,
            target_ref.node_id,
            target_ref.port_name,
        )
        existing_edge_id = self._edge_lookup.get(edge_key)
        if existing_edge_id is None:
            edge_name = (
                f"{source_ref.node_id}.{source_ref.port_name}"
                f"->{target_ref.node_id}.{target_ref.port_name}"
            )
            raise ValueError(f"unknown connection '{edge_name}'")

        for index, edge in enumerate(self._edges):
            if edge.edge_id != existing_edge_id:
                continue
            del self._edges[index]
            _ = self._edge_lookup.pop(edge_key, None)
            return existing_edge_id

        raise ValueError(f"unknown connection edge_id '{existing_edge_id}'")

    def _rewire_connection(
        self,
        *,
        edge_id: object,
        source: object,
        target: object,
    ) -> None:
        if not isinstance(edge_id, str) or not edge_id.strip():
            raise ValueError("rewire_connection field 'edge_id' must be a non-empty string")
        normalized_edge_id = edge_id.strip()
        if not isinstance(source, str) or not source.strip():
            raise ValueError("rewire_connection field 'source' must be a non-empty string")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("rewire_connection field 'target' must be a non-empty string")
        normalized_source = source.strip()
        normalized_target = target.strip()

        original_index: int | None = None
        original_edge: ConnectionEdge | None = None
        for index, edge in enumerate(self._edges):
            if edge.edge_id == normalized_edge_id:
                original_index = index
                original_edge = edge
                break
        if original_index is None or original_edge is None:
            raise ValueError(f"unknown connection edge_id '{normalized_edge_id}'")

        original_key = (
            original_edge.source.node_id,
            original_edge.source.port_name,
            original_edge.target.node_id,
            original_edge.target.port_name,
        )
        del self._edges[original_index]
        _ = self._edge_lookup.pop(original_key, None)
        try:
            _ = self.connect(normalized_source, normalized_target, edge_id=normalized_edge_id)
        except Exception:
            self._edges.insert(original_index, original_edge)
            self._edge_lookup[original_key] = normalized_edge_id
            raise

    def _capture_reconfiguration_state(self) -> _NetworkReconfigurationState:
        return _NetworkReconfigurationState(
            process_specs=dict(self._process_specs),
            boundary_specs=dict(self._boundary_specs),
            edges=list(self._edges),
            edge_lookup=dict(self._edge_lookup),
            next_edge_index=self._next_edge_index,
            next_reconfiguration_index=self._next_reconfiguration_index,
            next_revision_index=self._next_revision_index,
            queued_reconfigurations=list(self._queued_reconfigurations),
            compiled=self._compiled,
        )

    def _restore_reconfiguration_state(self, state: _NetworkReconfigurationState) -> None:
        self._process_specs = dict(state.process_specs)
        self._boundary_specs = dict(state.boundary_specs)
        self._edges = list(state.edges)
        self._edge_lookup = dict(state.edge_lookup)
        self._next_edge_index = state.next_edge_index
        self._next_reconfiguration_index = state.next_reconfiguration_index
        self._next_revision_index = state.next_revision_index
        self._queued_reconfigurations = list(state.queued_reconfigurations)
        self._compiled = state.compiled

    def _drain_reconfiguration_queue(self) -> tuple[_QueuedReconfigurationTransaction, ...]:
        queue = tuple(self._queued_reconfigurations)
        self._queued_reconfigurations = []
        return queue

    def _normalize_reconfiguration_mutation(
        self, request_mapping: Mapping[object, object]
    ) -> dict[str, object]:
        operation = request_mapping.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError(
                "invalid reconfiguration request: each mutation must be a mapping with non-empty string field 'operation'"
            )
        return {str(key): value for key, value in request_mapping.items()}

    def _parse_endpoint(self, endpoint: object) -> PortReference:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("invalid endpoint: expected non-empty string")

        normalized_endpoint = endpoint.strip()
        parts = normalized_endpoint.split(".")
        if len(parts) == 2:
            node_id = parts[0].strip()
            port_name = parts[1].strip()
            if not node_id or not port_name:
                raise ValueError(
                    f"invalid endpoint '{endpoint}': expected '<node_id>.<port_name>' format"
                )
            node_kind = self._node_kind(node_id)
            if node_kind is None:
                raise ValueError(f"unknown endpoint node '{node_id}'")
            self._assert_port_exists(node_id=node_id, port_name=port_name)
            return PortReference(node_id=node_id, port_name=port_name)

        if len(parts) != 1:
            raise ValueError(
                f"invalid endpoint '{endpoint}': expected '<node_id>.<port_name>' or '<boundary_id>'"
            )

        node_id = parts[0].strip()
        if not node_id:
            raise ValueError(
                f"invalid endpoint '{endpoint}': expected '<node_id>.<port_name>' or '<boundary_id>'"
            )

        node_kind = self._node_kind(node_id)
        if node_kind is None:
            raise ValueError(f"unknown endpoint node '{node_id}'")
        if node_kind == "process":
            raise ValueError(
                f"invalid endpoint '{endpoint}': process endpoints must use '<process_id>.<port_name>' format"
            )
        boundary = self._boundary_specs[node_id]
        return PortReference(node_id=node_id, port_name=boundary.port_name)

    def _assert_port_exists(self, *, node_id: str, port_name: str) -> None:
        process_spec = self._process_specs.get(node_id)
        if process_spec is not None:
            known_ports = set(process_spec.inlet_ports) | set(process_spec.outlet_ports)
            if port_name not in known_ports:
                raise ValueError(
                    f"unknown process endpoint '{node_id}.{port_name}': available ports are {sorted(known_ports)!r}"
                )
            return

        boundary_spec = self._boundary_specs.get(node_id)
        if boundary_spec is not None and port_name != boundary_spec.port_name:
            raise ValueError(
                f"unknown boundary endpoint '{node_id}.{port_name}': expected port '{boundary_spec.port_name}'"
            )

    def _node_kind(self, node_id: str) -> str | None:
        if node_id in self._process_specs:
            return "process"
        if node_id in self._boundary_specs:
            return "boundary"
        return None

    def _validate_node_id(self, node_id: object) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("node id must be a non-empty string")

    def _validate_edge_id(self, edge_id: object) -> None:
        if not isinstance(edge_id, str) or not edge_id.strip():
            raise ValueError("edge id must be a non-empty string")

    def _ensure_node_id_is_available(self, node_id: str) -> None:
        if self._node_kind(node_id) is not None:
            raise ValueError(f"duplicate node id '{node_id}' is not allowed")

    def _normalize_port_names(
        self,
        *,
        node_id: str,
        port_names: Sequence[str],
        role: str,
    ) -> tuple[str, ...]:
        normalized = tuple(self._normalize_port_name(port_name) for port_name in port_names)
        duplicates = sorted(
            {port_name for port_name in normalized if normalized.count(port_name) > 1}
        )
        if duplicates:
            raise ValueError(
                f"duplicate {role} ports for process '{node_id}' are not allowed: {duplicates!r}"
            )
        return normalized

    def _normalize_port_name(self, port_name: object) -> str:
        if not isinstance(port_name, str) or not port_name.strip():
            raise ValueError("port name must be a non-empty string")
        return port_name.strip()


__all__ = ["CompiledConnectionNetwork", "ConnectionNetwork"]
