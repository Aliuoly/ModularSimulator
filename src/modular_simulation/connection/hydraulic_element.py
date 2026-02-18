from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from collections.abc import Callable, Mapping

ElementValues = Mapping[str, float]
JacobianKey = tuple[str, str]
JacobianValues = Mapping[JacobianKey, float]


@dataclass(frozen=True)
class ElementParameterSpec:
    names: tuple[str, ...]


@dataclass(frozen=True)
class ElementUnknownSpec:
    names: tuple[str, ...]


@dataclass(frozen=True)
class ElementOutputSpec:
    residual_names: tuple[str, ...]


@runtime_checkable
class HydraulicElement(Protocol):
    parameter_spec: ElementParameterSpec
    unknown_spec: ElementUnknownSpec
    output_spec: ElementOutputSpec

    def residuals(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> ElementValues: ...

    def jacobian(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> JacobianValues: ...


def smooth_signed_quadratic_flow_term(*, m_dot: float, delta: float) -> float:
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    return m_dot * (m_dot * m_dot + delta * delta) ** 0.5


def smooth_signed_quadratic_flow_term_derivative(*, m_dot: float, delta: float) -> float:
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    denominator = (m_dot * m_dot + delta * delta) ** 0.5
    return (2.0 * m_dot * m_dot + delta * delta) / denominator


@dataclass(frozen=True)
class _SmoothHydraulicResistanceElement:
    resistance_parameter_name: str

    unknown_spec: ElementUnknownSpec = ElementUnknownSpec(names=("flow_rate",))
    output_spec: ElementOutputSpec = ElementOutputSpec(residual_names=("head_balance",))

    @property
    def parameter_spec(self) -> ElementParameterSpec:
        return ElementParameterSpec(names=(self.resistance_parameter_name, "delta"))

    def residuals(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> ElementValues:
        m_dot = unknowns["flow_rate"]
        resistance = parameters[self.resistance_parameter_name]
        delta = parameters["delta"]
        smooth_loss_term = smooth_signed_quadratic_flow_term(m_dot=m_dot, delta=delta)

        return {
            "head_balance": (
                inputs["upstream_pressure"]
                - inputs["downstream_pressure"]
                - resistance * smooth_loss_term
            )
        }

    def jacobian(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> JacobianValues:
        del inputs
        m_dot = unknowns["flow_rate"]
        resistance = parameters[self.resistance_parameter_name]
        delta = parameters["delta"]
        d_loss_dm_dot = smooth_signed_quadratic_flow_term_derivative(m_dot=m_dot, delta=delta)

        return {("head_balance", "flow_rate"): -resistance * d_loss_dm_dot}


@dataclass(frozen=True)
class PipeHydraulicElement(_SmoothHydraulicResistanceElement):
    resistance_parameter_name: str = "pipe_resistance"


@dataclass(frozen=True)
class ValveHydraulicElement(_SmoothHydraulicResistanceElement):
    resistance_parameter_name: str = "valve_resistance"


@dataclass(frozen=True)
class PumpHydraulicElement:
    dp_curve: Callable[[float, float], float] | None = None
    d_dp_d_mdot: Callable[[float, float], float] | None = None

    parameter_spec: ElementParameterSpec = ElementParameterSpec(names=("pump_speed",))
    unknown_spec: ElementUnknownSpec = ElementUnknownSpec(names=("flow_rate",))
    output_spec: ElementOutputSpec = ElementOutputSpec(residual_names=("head_balance",))

    def __post_init__(self) -> None:
        if self.dp_curve is None:
            raise ValueError("dp_curve callback is required")
        if not callable(self.dp_curve):
            raise TypeError("dp_curve must be callable")

        if self.d_dp_d_mdot is None:
            raise ValueError("d_dp_d_mdot callback is required")
        if not callable(self.d_dp_d_mdot):
            raise TypeError("d_dp_d_mdot must be callable")

    def residuals(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> ElementValues:
        dp_curve = self.dp_curve
        if dp_curve is None:
            raise RuntimeError("dp_curve callback must be set")
        m_dot = unknowns["flow_rate"]
        pump_speed = parameters["pump_speed"]
        pressure_rise = dp_curve(m_dot, pump_speed)

        return {
            "head_balance": (
                inputs["upstream_pressure"] - inputs["downstream_pressure"] + pressure_rise
            )
        }

    def jacobian(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> JacobianValues:
        del inputs
        d_dp_d_mdot = self.d_dp_d_mdot
        if d_dp_d_mdot is None:
            raise RuntimeError("d_dp_d_mdot callback must be set")
        m_dot = unknowns["flow_rate"]
        pump_speed = parameters["pump_speed"]
        derivative = d_dp_d_mdot(m_dot, pump_speed)

        return {("head_balance", "flow_rate"): derivative}
