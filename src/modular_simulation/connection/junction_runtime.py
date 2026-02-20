from __future__ import annotations

from collections.abc import Mapping

from modular_simulation.connection.junction import (
    NEAR_ZERO_INCOMING_FLOW_EPSILON,
    JunctionMixingResult,
    mix_junction_state,
)
from modular_simulation.connection.state import MaterialState, PortCondition
from modular_simulation.connection.topology import TopologyGraph


def mix_junction_runtime_states(
    *,
    topology: TopologyGraph,
    edge_states: Mapping[str, MaterialState],
    edge_flow_rates: Mapping[str, float],
    previous_junction_states: Mapping[str, MaterialState],
    near_zero_incoming_flow_epsilon: float = NEAR_ZERO_INCOMING_FLOW_EPSILON,
) -> dict[str, JunctionMixingResult]:
    edge_by_id = {edge.edge_id: edge for edge in topology.edges}
    results: dict[str, JunctionMixingResult] = {}

    for junction_id in sorted(junction.junction_id for junction in topology.junction_nodes):
        previous_state = previous_junction_states.get(junction_id)
        if previous_state is None:
            raise ValueError(f"missing previous junction state for '{junction_id}'")

        missing_data: list[str] = []
        junction_port_conditions: dict[str, PortCondition] = {}
        for edge_id in sorted(edge_by_id):
            edge = edge_by_id[edge_id]
            if edge.target.node_id == junction_id:
                junction_flow_sign = 1.0
            elif edge.source.node_id == junction_id:
                junction_flow_sign = -1.0
            else:
                continue

            edge_state = edge_states.get(edge_id)
            if edge_state is None:
                missing_data.append(f"{edge_id}:state")
            edge_flow_rate = edge_flow_rates.get(edge_id)
            if edge_flow_rate is None:
                missing_data.append(f"{edge_id}:flow_rate")
            if edge_state is None or edge_flow_rate is None:
                continue

            junction_port_conditions[edge_id] = PortCondition(
                state=edge_state,
                through_molar_flow_rate=junction_flow_sign * edge_flow_rate,
            )

        if missing_data:
            details = ", ".join(sorted(missing_data))
            raise ValueError(f"missing runtime edge data for junction '{junction_id}': {details}")

        results[junction_id] = mix_junction_state(
            incoming_port_conditions=junction_port_conditions,
            previous_state=previous_state,
            near_zero_incoming_flow_epsilon=near_zero_incoming_flow_epsilon,
        )

    return results


__all__ = ["mix_junction_runtime_states"]
