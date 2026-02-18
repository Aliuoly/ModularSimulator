from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final, Mapping

from modular_simulation.connection.state import MaterialState, PortCondition

NEAR_ZERO_INCOMING_FLOW_EPSILON: Final[float] = 1.0e-12


@dataclass(frozen=True)
class JunctionMixingResult:
    state: MaterialState
    total_incoming_flow_rate: float
    used_fallback: bool


def mix_junction_state(
    *,
    incoming_port_conditions: Mapping[str, PortCondition],
    previous_state: MaterialState,
    near_zero_incoming_flow_epsilon: float = NEAR_ZERO_INCOMING_FLOW_EPSILON,
) -> JunctionMixingResult:
    if near_zero_incoming_flow_epsilon <= 0.0:
        raise ValueError("near_zero_incoming_flow_epsilon must be positive")
    if not isfinite(near_zero_incoming_flow_epsilon):
        raise ValueError("near_zero_incoming_flow_epsilon must be finite")

    if not incoming_port_conditions:
        return JunctionMixingResult(
            state=previous_state,
            total_incoming_flow_rate=0.0,
            used_fallback=True,
        )

    composition_len = len(previous_state.mole_fractions)
    incoming_contributions: list[tuple[float, MaterialState]] = []

    for edge_id in sorted(incoming_port_conditions):
        port = incoming_port_conditions[edge_id]
        flow = port.through_molar_flow_rate
        if not isfinite(flow):
            raise ValueError("through_molar_flow_rate must be finite")
        if len(port.state.mole_fractions) != composition_len:
            raise ValueError("incoming and previous_state mole fraction lengths must match")
        if flow > 0.0:
            incoming_contributions.append((flow, port.state))

    total_incoming_flow_rate = sum(flow for flow, _ in incoming_contributions)
    if total_incoming_flow_rate <= near_zero_incoming_flow_epsilon:
        return JunctionMixingResult(
            state=previous_state,
            total_incoming_flow_rate=total_incoming_flow_rate,
            used_fallback=True,
        )

    mixed_pressure = (
        sum(flow * state.pressure for flow, state in incoming_contributions)
        / total_incoming_flow_rate
    )
    mixed_temperature = (
        sum(flow * state.temperature for flow, state in incoming_contributions)
        / total_incoming_flow_rate
    )
    mixed_mole_fractions = tuple(
        sum(flow * state.mole_fractions[index] for flow, state in incoming_contributions)
        / total_incoming_flow_rate
        for index in range(composition_len)
    )

    mixed_state = MaterialState(
        pressure=mixed_pressure,
        temperature=mixed_temperature,
        mole_fractions=mixed_mole_fractions,
    )
    return JunctionMixingResult(
        state=mixed_state,
        total_incoming_flow_rate=total_incoming_flow_rate,
        used_fallback=False,
    )


__all__ = [
    "JunctionMixingResult",
    "mix_junction_state",
]
