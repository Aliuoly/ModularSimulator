from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, isfinite
from typing import Final, Protocol

COMPOSITION_ABS_TOL: Final[float] = 1.0e-9


def _validate_composition(*, composition: tuple[float, ...], name: str) -> tuple[float, ...]:
    if not composition:
        raise ValueError(f"{name} must contain at least one component")
    for fraction in composition:
        if not isfinite(fraction):
            raise ValueError(f"{name} values must be finite")
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(f"{name} values must be between 0.0 and 1.0")
    if abs(sum(composition) - 1.0) > COMPOSITION_ABS_TOL:
        raise ValueError(f"{name} must sum to 1.0")
    return composition


@dataclass(frozen=True)
class LagTransportState:
    composition: tuple[float, ...]
    temperature: float

    def __post_init__(self) -> None:
        _ = _validate_composition(composition=self.composition, name="composition")
        if not isfinite(self.temperature):
            raise ValueError("temperature must be finite")


@dataclass(frozen=True)
class LagTransportUpdateResult:
    state: LagTransportState
    update_fraction: float
    flow_scale: float
    flow_sign_changed: bool
    held_for_near_zero_flow: bool


@dataclass(frozen=True)
class AdvancedConservativeTransportScaffoldState:
    model: str = "advanced_conservative"
    implemented: bool = False
    message: str = (
        "Advanced conservative transport is scaffolded only and is not implemented in T3.5."
    )


@dataclass(frozen=True)
class TransportModelSelection:
    model: str
    parameters: Mapping[str, float] | None = None


def _is_non_zero_flow(*, flow: float, epsilon: float) -> bool:
    return abs(flow) > epsilon


def _compute_flow_sign_changed(
    *,
    through_molar_flow_rate: float,
    previous_through_molar_flow_rate: float | None,
    near_zero_flow_epsilon: float,
) -> bool:
    if previous_through_molar_flow_rate is None:
        return False

    prev_non_zero = _is_non_zero_flow(
        flow=previous_through_molar_flow_rate,
        epsilon=near_zero_flow_epsilon,
    )
    curr_non_zero = _is_non_zero_flow(
        flow=through_molar_flow_rate,
        epsilon=near_zero_flow_epsilon,
    )
    return (
        prev_non_zero
        and curr_non_zero
        and previous_through_molar_flow_rate * through_molar_flow_rate < 0.0
    )


class TransportModel(Protocol):
    @property
    def model(self) -> str: ...

    def update(
        self,
        *,
        current_state: LagTransportState,
        advected_state: LagTransportState,
        dt: float,
        through_molar_flow_rate: float,
        previous_through_molar_flow_rate: float | None = None,
    ) -> LagTransportUpdateResult: ...


@dataclass(frozen=True)
class MVPLagTransportModel:
    lag_time_constant_s: float
    near_zero_flow_epsilon: float = 1.0e-12
    flow_smoothing_flow_rate: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.lag_time_constant_s <= 0.0:
            raise ValueError("lag_time_constant_s must be positive")
        if not isfinite(self.lag_time_constant_s):
            raise ValueError("lag_time_constant_s must be finite")
        if self.near_zero_flow_epsilon <= 0.0:
            raise ValueError("near_zero_flow_epsilon must be positive")
        if not isfinite(self.near_zero_flow_epsilon):
            raise ValueError("near_zero_flow_epsilon must be finite")
        if self.flow_smoothing_flow_rate <= 0.0:
            raise ValueError("flow_smoothing_flow_rate must be positive")
        if not isfinite(self.flow_smoothing_flow_rate):
            raise ValueError("flow_smoothing_flow_rate must be finite")

    @property
    def model(self) -> str:
        return "mvp_lag"

    def update(
        self,
        *,
        current_state: LagTransportState,
        advected_state: LagTransportState,
        dt: float,
        through_molar_flow_rate: float,
        previous_through_molar_flow_rate: float | None = None,
    ) -> LagTransportUpdateResult:
        return update_lag_transport_state(
            current_state=current_state,
            advected_state=advected_state,
            dt=dt,
            lag_time_constant_s=self.lag_time_constant_s,
            through_molar_flow_rate=through_molar_flow_rate,
            previous_through_molar_flow_rate=previous_through_molar_flow_rate,
            near_zero_flow_epsilon=self.near_zero_flow_epsilon,
            flow_smoothing_flow_rate=self.flow_smoothing_flow_rate,
        )


@dataclass(frozen=True)
class AdvancedConservativeTransportModelScaffold:
    scaffold_state: AdvancedConservativeTransportScaffoldState = (
        AdvancedConservativeTransportScaffoldState()
    )

    @property
    def model(self) -> str:
        return "advanced_conservative"

    def update(
        self,
        *,
        current_state: LagTransportState,
        advected_state: LagTransportState,
        dt: float,
        through_molar_flow_rate: float,
        previous_through_molar_flow_rate: float | None = None,
    ) -> LagTransportUpdateResult:
        del current_state, advected_state, dt
        _ = _compute_flow_sign_changed(
            through_molar_flow_rate=through_molar_flow_rate,
            previous_through_molar_flow_rate=previous_through_molar_flow_rate,
            near_zero_flow_epsilon=1.0e-12,
        )
        raise NotImplementedError(self.scaffold_state.message)


def select_transport_model(config: TransportModelSelection) -> TransportModel:
    parameters = dict(config.parameters or {})

    if config.model == "mvp_lag":
        return MVPLagTransportModel(**parameters)
    if config.model == "advanced_conservative":
        if parameters:
            raise ValueError(
                "advanced_conservative scaffold does not accept parameters yet. "
                + "Use model='advanced_conservative' with no parameters."
            )
        return AdvancedConservativeTransportModelScaffold()
    supported_models = "mvp_lag, advanced_conservative"
    raise ValueError(
        f"Unsupported transport model '{config.model}'. Supported models: {supported_models}."
    )


def update_lag_transport_state(
    *,
    current_state: LagTransportState,
    advected_state: LagTransportState,
    dt: float,
    lag_time_constant_s: float,
    through_molar_flow_rate: float,
    previous_through_molar_flow_rate: float | None = None,
    near_zero_flow_epsilon: float = 1.0e-12,
    flow_smoothing_flow_rate: float = 1.0e-9,
) -> LagTransportUpdateResult:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not isfinite(dt):
        raise ValueError("dt must be finite")
    if lag_time_constant_s <= 0.0:
        raise ValueError("lag_time_constant_s must be positive")
    if not isfinite(lag_time_constant_s):
        raise ValueError("lag_time_constant_s must be finite")
    if near_zero_flow_epsilon <= 0.0:
        raise ValueError("near_zero_flow_epsilon must be positive")
    if not isfinite(near_zero_flow_epsilon):
        raise ValueError("near_zero_flow_epsilon must be finite")
    if flow_smoothing_flow_rate <= 0.0:
        raise ValueError("flow_smoothing_flow_rate must be positive")
    if not isfinite(flow_smoothing_flow_rate):
        raise ValueError("flow_smoothing_flow_rate must be finite")
    if not isfinite(through_molar_flow_rate):
        raise ValueError("through_molar_flow_rate must be finite")
    if previous_through_molar_flow_rate is not None and not isfinite(
        previous_through_molar_flow_rate
    ):
        raise ValueError("previous_through_molar_flow_rate must be finite when provided")
    if len(current_state.composition) != len(advected_state.composition):
        raise ValueError("current_state and advected_state composition lengths must match")

    held_for_near_zero_flow = not _is_non_zero_flow(
        flow=through_molar_flow_rate,
        epsilon=near_zero_flow_epsilon,
    )

    flow_sign_changed = _compute_flow_sign_changed(
        through_molar_flow_rate=through_molar_flow_rate,
        previous_through_molar_flow_rate=previous_through_molar_flow_rate,
        near_zero_flow_epsilon=near_zero_flow_epsilon,
    )

    if held_for_near_zero_flow:
        return LagTransportUpdateResult(
            state=current_state,
            update_fraction=0.0,
            flow_scale=0.0,
            flow_sign_changed=flow_sign_changed,
            held_for_near_zero_flow=True,
        )

    flow_magnitude = abs(through_molar_flow_rate)
    flow_scale = flow_magnitude / (flow_magnitude + flow_smoothing_flow_rate)
    base_fraction = 1.0 - exp(-dt / lag_time_constant_s)
    update_fraction = max(0.0, min(1.0, flow_scale * base_fraction))

    composition = tuple(
        current_fraction + update_fraction * (advected_fraction - current_fraction)
        for current_fraction, advected_fraction in zip(
            current_state.composition,
            advected_state.composition,
            strict=True,
        )
    )
    temperature = current_state.temperature + update_fraction * (
        advected_state.temperature - current_state.temperature
    )

    next_state = LagTransportState(composition=composition, temperature=temperature)
    return LagTransportUpdateResult(
        state=next_state,
        update_fraction=update_fraction,
        flow_scale=flow_scale,
        flow_sign_changed=flow_sign_changed,
        held_for_near_zero_flow=False,
    )


__all__ = [
    "AdvancedConservativeTransportModelScaffold",
    "AdvancedConservativeTransportScaffoldState",
    "LagTransportState",
    "LagTransportUpdateResult",
    "MVPLagTransportModel",
    "TransportModel",
    "TransportModelSelection",
    "select_transport_model",
    "update_lag_transport_state",
]
