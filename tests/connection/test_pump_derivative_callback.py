from __future__ import annotations

import importlib
from types import MappingProxyType

import pytest

hydraulic_element = importlib.import_module("modular_simulation.connection.hydraulic_element")
PumpHydraulicElement = hydraulic_element.PumpHydraulicElement


def test_pump_residual_and_jacobian_use_analytic_callbacks() -> None:
    callback_args: list[tuple[str, float, float]] = []

    def dp_curve(mdot: float, speed: float) -> float:
        callback_args.append(("dp", mdot, speed))
        return 15_000.0 + 8.0 * speed - 2.0 * mdot

    def d_dp_d_mdot(mdot: float, speed: float) -> float:
        callback_args.append(("d", mdot, speed))
        return -2.0

    element = PumpHydraulicElement(dp_curve=dp_curve, d_dp_d_mdot=d_dp_d_mdot)

    inputs = MappingProxyType({"upstream_pressure": 180_000.0, "downstream_pressure": 220_000.0})
    unknowns = MappingProxyType({"flow_rate": 3.0})
    parameters = MappingProxyType({"pump_speed": 1_000.0})

    residual = element.residuals(inputs=inputs, unknowns=unknowns, parameters=parameters)[
        "head_balance"
    ]
    jacobian = element.jacobian(inputs=inputs, unknowns=unknowns, parameters=parameters)[
        ("head_balance", "flow_rate")
    ]

    assert residual == pytest.approx(180_000.0 - 220_000.0 + (15_000.0 + 8.0 * 1_000.0 - 2.0 * 3.0))
    assert jacobian == pytest.approx(-2.0)
    assert callback_args == [("dp", 3.0, 1_000.0), ("d", 3.0, 1_000.0)]


def test_jacobian_path_uses_derivative_callback_only() -> None:
    derivative_calls: list[tuple[float, float]] = []

    def dp_curve(mdot: float, speed: float) -> float:
        raise AssertionError("jacobian must not call dp_curve")

    def d_dp_d_mdot(mdot: float, speed: float) -> float:
        derivative_calls.append((mdot, speed))
        return 42.0

    element = PumpHydraulicElement(dp_curve=dp_curve, d_dp_d_mdot=d_dp_d_mdot)
    jacobian = element.jacobian(
        inputs=MappingProxyType({"upstream_pressure": 0.0, "downstream_pressure": 0.0}),
        unknowns=MappingProxyType({"flow_rate": -1.25}),
        parameters=MappingProxyType({"pump_speed": 900.0}),
    )

    assert jacobian[("head_balance", "flow_rate")] == pytest.approx(42.0)
    assert derivative_calls == [(-1.25, 900.0)]


def test_pump_requires_callable_callbacks() -> None:
    with pytest.raises(ValueError, match="dp_curve callback is required"):
        PumpHydraulicElement(d_dp_d_mdot=lambda mdot, speed: mdot + speed)

    with pytest.raises(ValueError, match="d_dp_d_mdot callback is required"):
        PumpHydraulicElement(dp_curve=lambda mdot, speed: mdot + speed)

    with pytest.raises(TypeError, match="dp_curve must be callable"):
        PumpHydraulicElement(dp_curve=123.0, d_dp_d_mdot=lambda mdot, speed: mdot + speed)

    with pytest.raises(TypeError, match="d_dp_d_mdot must be callable"):
        PumpHydraulicElement(dp_curve=lambda mdot, speed: mdot + speed, d_dp_d_mdot=123.0)
