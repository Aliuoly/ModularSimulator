from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Callable, Protocol, cast

import pytest

hydraulic_element = importlib.import_module("modular_simulation.connection.hydraulic_element")


class _SmoothHydraulicElement(Protocol):
    def residuals(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> Mapping[str, float]: ...

    def jacobian(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> Mapping[tuple[str, str], float]: ...


PipeHydraulicElement = cast(type[_SmoothHydraulicElement], hydraulic_element.PipeHydraulicElement)
ValveHydraulicElement = cast(type[_SmoothHydraulicElement], hydraulic_element.ValveHydraulicElement)
smooth_signed_quadratic_flow_term = cast(
    Callable[..., float], hydraulic_element.smooth_signed_quadratic_flow_term
)
smooth_signed_quadratic_flow_term_derivative = cast(
    Callable[..., float], hydraulic_element.smooth_signed_quadratic_flow_term_derivative
)


@pytest.mark.parametrize(
    ("element", "resistance_parameter_name"),
    [
        (PipeHydraulicElement(), "pipe_resistance"),
        (ValveHydraulicElement(), "valve_resistance"),
    ],
)
def test_residual_is_continuous_around_zero_flow(
    element: _SmoothHydraulicElement,
    resistance_parameter_name: str,
) -> None:
    baseline = 230_000.0 - 125_000.0
    epsilon = 1.0e-9
    parameters = MappingProxyType({resistance_parameter_name: 9_500.0, "delta": 1.0e-3})
    inputs = MappingProxyType({"upstream_pressure": 230_000.0, "downstream_pressure": 125_000.0})

    residual_pos = element.residuals(
        inputs=inputs,
        unknowns=MappingProxyType({"flow_rate": epsilon}),
        parameters=parameters,
    )["head_balance"]
    residual_neg = element.residuals(
        inputs=inputs,
        unknowns=MappingProxyType({"flow_rate": -epsilon}),
        parameters=parameters,
    )["head_balance"]

    assert residual_pos == pytest.approx(baseline, abs=1.0e-5)
    assert residual_neg == pytest.approx(baseline, abs=1.0e-5)
    assert residual_pos == pytest.approx(residual_neg, abs=1.0e-7)


@pytest.mark.parametrize(
    ("element", "resistance_parameter_name"),
    [
        (PipeHydraulicElement(), "pipe_resistance"),
        (ValveHydraulicElement(), "valve_resistance"),
    ],
)
def test_jacobian_sign_and_magnitude_near_zero_and_away(
    element: _SmoothHydraulicElement,
    resistance_parameter_name: str,
) -> None:
    resistance = 7_000.0
    delta = 2.0e-3
    parameters = MappingProxyType({resistance_parameter_name: resistance, "delta": delta})
    inputs = MappingProxyType({"upstream_pressure": 210_000.0, "downstream_pressure": 140_000.0})

    jac_zero = element.jacobian(
        inputs=inputs,
        unknowns=MappingProxyType({"flow_rate": 0.0}),
        parameters=parameters,
    )[("head_balance", "flow_rate")]
    jac_pos_eps = element.jacobian(
        inputs=inputs,
        unknowns=MappingProxyType({"flow_rate": 1.0e-9}),
        parameters=parameters,
    )[("head_balance", "flow_rate")]
    jac_neg_eps = element.jacobian(
        inputs=inputs,
        unknowns=MappingProxyType({"flow_rate": -1.0e-9}),
        parameters=parameters,
    )[("head_balance", "flow_rate")]
    jac_far = element.jacobian(
        inputs=inputs,
        unknowns=MappingProxyType({"flow_rate": 2.0}),
        parameters=parameters,
    )[("head_balance", "flow_rate")]

    assert jac_zero < 0.0
    assert jac_pos_eps < 0.0
    assert jac_neg_eps < 0.0
    assert jac_zero == pytest.approx(-(resistance * delta), rel=1.0e-12)
    assert jac_pos_eps == pytest.approx(jac_neg_eps, abs=1.0e-3)
    assert abs(jac_far) > abs(jac_zero)


def test_pipe_and_valve_share_the_same_smooth_law_formulation() -> None:
    pipe = PipeHydraulicElement()
    valve = ValveHydraulicElement()
    inputs = MappingProxyType({"upstream_pressure": 250_000.0, "downstream_pressure": 180_000.0})
    unknowns = MappingProxyType({"flow_rate": -0.4})
    delta = 0.02
    resistance = 12_000.0

    pipe_parameters = MappingProxyType({"pipe_resistance": resistance, "delta": delta})
    valve_parameters = MappingProxyType({"valve_resistance": resistance, "delta": delta})

    pipe_residual = pipe.residuals(inputs=inputs, unknowns=unknowns, parameters=pipe_parameters)[
        "head_balance"
    ]
    valve_residual = valve.residuals(inputs=inputs, unknowns=unknowns, parameters=valve_parameters)[
        "head_balance"
    ]
    pipe_jacobian = pipe.jacobian(inputs=inputs, unknowns=unknowns, parameters=pipe_parameters)[
        ("head_balance", "flow_rate")
    ]
    valve_jacobian = valve.jacobian(inputs=inputs, unknowns=unknowns, parameters=valve_parameters)[
        ("head_balance", "flow_rate")
    ]

    smooth_term = smooth_signed_quadratic_flow_term(m_dot=-0.4, delta=delta)
    smooth_derivative = smooth_signed_quadratic_flow_term_derivative(m_dot=-0.4, delta=delta)
    expected_residual = 250_000.0 - 180_000.0 - resistance * smooth_term
    expected_jacobian = -(resistance * smooth_derivative)

    assert pipe_residual == pytest.approx(expected_residual)
    assert valve_residual == pytest.approx(expected_residual)
    assert pipe_jacobian == pytest.approx(expected_jacobian)
    assert valve_jacobian == pytest.approx(expected_jacobian)
