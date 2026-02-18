from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Protocol

HydraulicSolveResult = Any
JunctionMixingResult = Any
PortCondition = Any
LagTransportUpdateResult = Any

MACRO_COUPLING_SEQUENCE: tuple[str, str, str] = (
    "hydraulics_solve",
    "transport_update",
    "boundary_state_propagation",
)


@dataclass(frozen=True)
class TransportAndMixingUpdate:
    transport_results: Mapping[str, LagTransportUpdateResult]
    junction_results: Mapping[str, JunctionMixingResult] = field(default_factory=dict)


@dataclass(frozen=True)
class MacroStepBookkeeping:
    macro_step_time_s: float
    macro_step_index: int | None
    executed_sequence: tuple[str, str, str]
    updated_port_keys: tuple[str, ...]
    transport_fallback_keys: tuple[str, ...]
    junction_fallback_keys: tuple[str, ...]
    used_fallback: bool
    picard_gate_enabled: bool = False
    picard_iterations_used: int = 0
    picard_converged: bool = True
    picard_last_residual: float | None = None


@dataclass(frozen=True)
class MacroCouplingStepResult:
    hydraulic_result: HydraulicSolveResult
    transport_and_mixing_update: TransportAndMixingUpdate
    port_conditions: Mapping[str, PortCondition]
    bookkeeping: MacroStepBookkeeping


@dataclass(frozen=True)
class PicardIterationGateConfig:
    residual_threshold: float
    max_iterations: int
    tolerance: float
    enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if not isfinite(self.residual_threshold) or self.residual_threshold < 0.0:
            raise ValueError("residual_threshold must be finite and non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")


class HydraulicSolveStep(Protocol):
    def __call__(self, *, macro_step_time_s: float) -> HydraulicSolveResult: ...


class TransportUpdateStep(Protocol):
    def __call__(
        self,
        *,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate: ...


class BoundaryPropagationStep(Protocol):
    def __call__(
        self,
        *,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> Mapping[str, PortCondition]: ...


class ProcessPortConditionWriteStep(Protocol):
    def __call__(
        self,
        *,
        port_conditions: Mapping[str, PortCondition],
        macro_step_time_s: float,
    ) -> None: ...


def run_macro_coupling_step(
    *,
    macro_step_time_s: float,
    hydraulic_solve_step: HydraulicSolveStep,
    transport_update_step: TransportUpdateStep,
    boundary_propagation_step: BoundaryPropagationStep,
    macro_step_index: int | None = None,
    picard_gate_config: PicardIterationGateConfig | None = None,
) -> MacroCouplingStepResult:
    if macro_step_time_s < 0.0:
        raise ValueError("macro_step_time_s must be non-negative")
    if not isfinite(macro_step_time_s):
        raise ValueError("macro_step_time_s must be finite")

    def _run_sequence_once() -> tuple[
        HydraulicSolveResult,
        TransportAndMixingUpdate,
        Mapping[str, PortCondition],
    ]:
        hydraulic_result = hydraulic_solve_step(macro_step_time_s=macro_step_time_s)
        transport_and_mixing_update = transport_update_step(
            hydraulic_result=hydraulic_result,
            macro_step_time_s=macro_step_time_s,
        )
        port_conditions = boundary_propagation_step(
            hydraulic_result=hydraulic_result,
            transport_and_mixing_update=transport_and_mixing_update,
            macro_step_time_s=macro_step_time_s,
        )
        return hydraulic_result, transport_and_mixing_update, port_conditions

    hydraulic_result, transport_and_mixing_update, port_conditions = _run_sequence_once()

    gate_residual = _extract_gate_residual(hydraulic_result)
    picard_gate_enabled = (
        picard_gate_config is not None
        and picard_gate_config.enabled
        and gate_residual > picard_gate_config.residual_threshold
    )
    picard_iterations_used = 0
    picard_converged = True
    picard_last_residual: float | None = None

    if picard_gate_enabled and picard_gate_config is not None:
        previous_port_conditions = port_conditions
        picard_converged = False
        for iteration in range(1, picard_gate_config.max_iterations + 1):
            hydraulic_result, transport_and_mixing_update, port_conditions = _run_sequence_once()
            picard_iterations_used = iteration
            picard_last_residual = _compute_coupling_residual(
                previous_port_conditions=previous_port_conditions,
                current_port_conditions=port_conditions,
            )
            if picard_last_residual <= picard_gate_config.tolerance:
                picard_converged = True
                break
            previous_port_conditions = port_conditions

    transport_fallback_keys = tuple(
        sorted(
            edge_key
            for edge_key, update_result in transport_and_mixing_update.transport_results.items()
            if update_result.held_for_near_zero_flow
        )
    )
    junction_fallback_keys = tuple(
        sorted(
            junction_key
            for junction_key, update_result in transport_and_mixing_update.junction_results.items()
            if update_result.used_fallback
        )
    )
    used_fallback = bool(transport_fallback_keys or junction_fallback_keys)

    bookkeeping = MacroStepBookkeeping(
        macro_step_time_s=macro_step_time_s,
        macro_step_index=macro_step_index,
        executed_sequence=MACRO_COUPLING_SEQUENCE,
        updated_port_keys=tuple(sorted(port_conditions.keys())),
        transport_fallback_keys=transport_fallback_keys,
        junction_fallback_keys=junction_fallback_keys,
        used_fallback=used_fallback,
        picard_gate_enabled=picard_gate_enabled,
        picard_iterations_used=picard_iterations_used,
        picard_converged=picard_converged,
        picard_last_residual=picard_last_residual,
    )
    return MacroCouplingStepResult(
        hydraulic_result=hydraulic_result,
        transport_and_mixing_update=transport_and_mixing_update,
        port_conditions=port_conditions,
        bookkeeping=bookkeeping,
    )


def _extract_gate_residual(hydraulic_result: HydraulicSolveResult) -> float:
    residual_norm = getattr(hydraulic_result, "residual_norm", 0.0)
    try:
        residual = float(residual_norm)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(residual) or residual < 0.0:
        return 0.0
    return residual


def _compute_coupling_residual(
    *,
    previous_port_conditions: Mapping[str, PortCondition],
    current_port_conditions: Mapping[str, PortCondition],
) -> float:
    all_keys = sorted(set(previous_port_conditions.keys()) | set(current_port_conditions.keys()))
    residual = 0.0
    for key in all_keys:
        previous = previous_port_conditions.get(key)
        current = current_port_conditions.get(key)
        if previous is None or current is None:
            return float("inf")
        residual = max(residual, _port_condition_delta(previous, current))
    return residual


def _port_condition_delta(previous: PortCondition, current: PortCondition) -> float:
    previous_flow = _as_finite_float(getattr(previous, "through_molar_flow_rate", 0.0))
    current_flow = _as_finite_float(getattr(current, "through_molar_flow_rate", 0.0))
    residual = abs(current_flow - previous_flow)

    previous_state = getattr(previous, "state", None)
    current_state = getattr(current, "state", None)
    if previous_state is None or current_state is None:
        return residual

    residual = max(
        residual,
        abs(
            _as_finite_float(getattr(current_state, "pressure", 0.0))
            - _as_finite_float(getattr(previous_state, "pressure", 0.0))
        ),
    )
    residual = max(
        residual,
        abs(
            _as_finite_float(getattr(current_state, "temperature", 0.0))
            - _as_finite_float(getattr(previous_state, "temperature", 0.0))
        ),
    )

    previous_comp = getattr(previous_state, "mole_fractions", ())
    current_comp = getattr(current_state, "mole_fractions", ())
    if len(previous_comp) != len(current_comp):
        return float("inf")
    for previous_frac, current_frac in zip(previous_comp, current_comp, strict=True):
        residual = max(
            residual, abs(_as_finite_float(current_frac) - _as_finite_float(previous_frac))
        )
    return residual


def _as_finite_float(value: Any) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(scalar):
        return 0.0
    return scalar


__all__ = [
    "MACRO_COUPLING_SEQUENCE",
    "BoundaryPropagationStep",
    "HydraulicSolveStep",
    "MacroCouplingStepResult",
    "MacroStepBookkeeping",
    "PicardIterationGateConfig",
    "ProcessPortConditionWriteStep",
    "TransportAndMixingUpdate",
    "TransportUpdateStep",
    "run_macro_coupling_step",
]
