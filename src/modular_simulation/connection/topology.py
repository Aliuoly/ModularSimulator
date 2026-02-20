from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProcessNode:
    process_id: str
    inlet_ports: tuple[str, ...]
    outlet_ports: tuple[str, ...]

    def endpoints(self) -> tuple[str, ...]:
        return tuple((*self.inlet_ports, *self.outlet_ports))


@dataclass(frozen=True)
class BoundaryNode:
    boundary_id: str
    port_name: str
    direction: Literal["source", "sink"]

    def __post_init__(self) -> None:
        if self.direction not in ("source", "sink"):
            raise ValueError(
                f"boundary {self.boundary_id!r} direction must be 'source' or 'sink', got {self.direction!r}"
            )

    def endpoints(self) -> tuple[str, ...]:
        return (self.port_name,)


@dataclass(frozen=True)
class JunctionNode:
    junction_id: str
    port_names: tuple[str, ...]

    def endpoints(self) -> tuple[str, ...]:
        return self.port_names


@dataclass(frozen=True)
class PortReference:
    node_id: str
    port_name: str


@dataclass(frozen=True)
class ConnectionEdge:
    edge_id: str
    source: PortReference
    target: PortReference


@dataclass(frozen=True)
class TopologyGraph:
    process_nodes: tuple[ProcessNode, ...]
    boundary_nodes: tuple[BoundaryNode, ...]
    junction_nodes: tuple[JunctionNode, ...]
    edges: tuple[ConnectionEdge, ...]

    def __post_init__(self) -> None:
        self._validate_duplicate_ids()
        endpoint_index = self._build_endpoint_index()
        self._validate_unknown_endpoints(endpoint_index)
        self._validate_duplicate_edges()
        self._validate_invalid_self_links()

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._node_ids()))

    @property
    def edge_ids(self) -> tuple[str, ...]:
        return tuple(sorted(edge.edge_id for edge in self.edges))

    def _node_ids(self) -> tuple[str, ...]:
        process_ids = tuple(node.process_id for node in self.process_nodes)
        boundary_ids = tuple(node.boundary_id for node in self.boundary_nodes)
        junction_ids = tuple(node.junction_id for node in self.junction_nodes)
        return (*process_ids, *boundary_ids, *junction_ids)

    def _validate_duplicate_ids(self) -> None:
        duplicate_ids: list[str] = []

        node_counts = Counter(self._node_ids())
        duplicate_ids.extend(f"node:{name}" for name, count in node_counts.items() if count > 1)

        edge_counts = Counter(edge.edge_id for edge in self.edges)
        duplicate_ids.extend(f"edge:{name}" for name, count in edge_counts.items() if count > 1)

        if duplicate_ids:
            raise ValueError(f"duplicate ids are not allowed: {sorted(duplicate_ids)!r}")

    def _build_endpoint_index(self) -> set[tuple[str, str]]:
        endpoint_index: set[tuple[str, str]] = set()
        for process in self.process_nodes:
            for port_name in process.endpoints():
                endpoint_index.add((process.process_id, port_name))
        for boundary in self.boundary_nodes:
            for port_name in boundary.endpoints():
                endpoint_index.add((boundary.boundary_id, port_name))
        for junction in self.junction_nodes:
            for port_name in junction.endpoints():
                endpoint_index.add((junction.junction_id, port_name))
        return endpoint_index

    def _validate_unknown_endpoints(self, endpoint_index: set[tuple[str, str]]) -> None:
        known_nodes = set(self.node_ids)
        errors: list[str] = []
        for edge in self.edges:
            if (edge.source.node_id, edge.source.port_name) not in endpoint_index:
                detail = (
                    "node not found" if edge.source.node_id not in known_nodes else "port not found"
                )
                errors.append(
                    f"{edge.edge_id}:source={edge.source.node_id}.{edge.source.port_name} ({detail})"
                )
            if (edge.target.node_id, edge.target.port_name) not in endpoint_index:
                detail = (
                    "node not found" if edge.target.node_id not in known_nodes else "port not found"
                )
                errors.append(
                    f"{edge.edge_id}:target={edge.target.node_id}.{edge.target.port_name} ({detail})"
                )

        if errors:
            raise ValueError(f"unknown connection endpoints: {sorted(errors)!r}")

    def _validate_duplicate_edges(self) -> None:
        edge_keys = Counter(
            (edge.source.node_id, edge.source.port_name, edge.target.node_id, edge.target.port_name)
            for edge in self.edges
        )
        duplicates = [
            f"{source_node}.{source_port}->{target_node}.{target_port}"
            for (source_node, source_port, target_node, target_port), count in edge_keys.items()
            if count > 1
        ]
        if duplicates:
            raise ValueError(f"duplicate edges are not allowed: {sorted(duplicates)!r}")

    def _validate_invalid_self_links(self) -> None:
        self_links = [
            f"{edge.edge_id}:{edge.source.node_id}.{edge.source.port_name}"
            for edge in self.edges
            if edge.source.node_id == edge.target.node_id
            and edge.source.port_name == edge.target.port_name
        ]
        if self_links:
            raise ValueError(f"invalid self-links are not allowed: {sorted(self_links)!r}")


__all__ = [
    "BoundaryNode",
    "ConnectionEdge",
    "JunctionNode",
    "PortReference",
    "ProcessNode",
    "TopologyGraph",
]
