from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from modular_simulation.connection.topology import TopologyGraph
from modular_simulation.connection.transport import (
    LagTransportState,
    LagTransportUpdateResult,
    update_lag_transport_state,
)

_COMPOSITION_ABS_TOL = 1.0e-9


@dataclass(frozen=True)
class TransportRuntimeStepResult:
    edge_updates: Mapping[str, LagTransportUpdateResult]
    edge_states: Mapping[str, LagTransportState]
    previous_edge_flow_rates: Mapping[str, float]


def update_transport_runtime_states(
    *,
    topology: TopologyGraph,
    current_edge_states: Mapping[str, LagTransportState],
    advected_edge_states: Mapping[str, LagTransportState],
    edge_flow_rates: Mapping[str, float],
    previous_edge_flow_rates: Mapping[str, float] | None,
    dt: float,
    lag_time_constant_s: float,
    near_zero_flow_epsilon: float = 1.0e-12,
    flow_smoothing_flow_rate: float = 1.0e-9,
) -> TransportRuntimeStepResult:
    missing_data: list[str] = []
    edge_updates: dict[str, LagTransportUpdateResult] = {}
    edge_states: dict[str, LagTransportState] = {}
    next_previous_edge_flow_rates: dict[str, float] = {}

    for edge_id in sorted(edge.edge_id for edge in topology.edges):
        current_state = current_edge_states.get(edge_id)
        if current_state is None:
            missing_data.append(f"{edge_id}:current_state")

        advected_state = advected_edge_states.get(edge_id)
        if advected_state is None:
            missing_data.append(f"{edge_id}:advected_state")

        through_molar_flow_rate = edge_flow_rates.get(edge_id)
        if through_molar_flow_rate is None:
            missing_data.append(f"{edge_id}:flow_rate")

        if current_state is None or advected_state is None or through_molar_flow_rate is None:
            continue

        previous_through_molar_flow_rate: float | None = None
        if previous_edge_flow_rates is not None:
            previous_through_molar_flow_rate = previous_edge_flow_rates.get(edge_id)

        update_result = update_lag_transport_state(
            current_state=current_state,
            advected_state=advected_state,
            dt=dt,
            lag_time_constant_s=lag_time_constant_s,
            through_molar_flow_rate=through_molar_flow_rate,
            previous_through_molar_flow_rate=previous_through_molar_flow_rate,
            near_zero_flow_epsilon=near_zero_flow_epsilon,
            flow_smoothing_flow_rate=flow_smoothing_flow_rate,
        )
        _assert_runtime_invariants(edge_id=edge_id, update_result=update_result)

        edge_updates[edge_id] = update_result
        edge_states[edge_id] = update_result.state
        next_previous_edge_flow_rates[edge_id] = through_molar_flow_rate

    if missing_data:
        details = ", ".join(missing_data)
        raise ValueError(f"missing runtime edge data: {details}")

    return TransportRuntimeStepResult(
        edge_updates=edge_updates,
        edge_states=edge_states,
        previous_edge_flow_rates=next_previous_edge_flow_rates,
    )


def _assert_runtime_invariants(*, edge_id: str, update_result: LagTransportUpdateResult) -> None:
    if not isfinite(update_result.update_fraction):
        raise ValueError(f"transport runtime produced non-finite update_fraction for '{edge_id}'")
    if update_result.update_fraction < 0.0 or update_result.update_fraction > 1.0:
        raise ValueError(f"transport runtime produced out-of-range update_fraction for '{edge_id}'")

    if not isfinite(update_result.flow_scale):
        raise ValueError(f"transport runtime produced non-finite flow_scale for '{edge_id}'")
    if update_result.flow_scale < 0.0 or update_result.flow_scale > 1.0:
        raise ValueError(f"transport runtime produced out-of-range flow_scale for '{edge_id}'")

    state = update_result.state
    if not isfinite(state.temperature):
        raise ValueError(f"transport runtime produced non-finite temperature for '{edge_id}'")

    composition_sum = 0.0
    for idx, fraction in enumerate(state.composition):
        if not isfinite(fraction):
            raise ValueError(
                f"transport runtime produced non-finite composition value for '{edge_id}' at index {idx}"
            )
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(
                f"transport runtime produced out-of-range composition value for '{edge_id}' at index {idx}"
            )
        composition_sum += fraction

    if abs(composition_sum - 1.0) > _COMPOSITION_ABS_TOL:
        raise ValueError(f"transport runtime produced non-normalized composition for '{edge_id}'")


__all__ = ["TransportRuntimeStepResult", "update_transport_runtime_states"]
