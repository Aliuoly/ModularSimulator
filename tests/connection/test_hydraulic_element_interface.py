from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import MappingProxyType

hydraulic_element = importlib.import_module("modular_simulation.connection.hydraulic_element")
ElementOutputSpec = hydraulic_element.ElementOutputSpec
ElementParameterSpec = hydraulic_element.ElementParameterSpec
ElementUnknownSpec = hydraulic_element.ElementUnknownSpec
HydraulicElement = hydraulic_element.HydraulicElement


class DemoHydraulicElement:
    parameter_spec: ElementParameterSpec = ElementParameterSpec(names=("resistance", "pump_gain"))
    unknown_spec: ElementUnknownSpec = ElementUnknownSpec(names=("flow_rate",))
    output_spec: ElementOutputSpec = ElementOutputSpec(
        residual_names=("mass_balance", "head_balance")
    )

    def residuals(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> Mapping[str, float]:
        return {
            "mass_balance": unknowns["flow_rate"] - inputs["target_flow_rate"],
            "head_balance": (
                inputs["upstream_pressure"]
                - inputs["downstream_pressure"]
                - parameters["resistance"] * unknowns["flow_rate"]
                + parameters["pump_gain"] * inputs["pump_speed"]
            ),
        }

    def jacobian(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> Mapping[tuple[str, str], float]:
        del inputs, unknowns
        return {
            ("mass_balance", "flow_rate"): 1.0,
            ("head_balance", "flow_rate"): -parameters["resistance"],
        }


def test_hydraulic_element_protocol_runtime_contract() -> None:
    element = DemoHydraulicElement()

    assert isinstance(element, HydraulicElement)
    assert element.parameter_spec.names == ("resistance", "pump_gain")
    assert element.unknown_spec.names == ("flow_rate",)
    assert element.output_spec.residual_names == ("mass_balance", "head_balance")


def test_residual_and_jacobian_accept_generic_mapping_inputs() -> None:
    element = DemoHydraulicElement()

    inputs = MappingProxyType(
        {
            "upstream_pressure": 200_000.0,
            "downstream_pressure": 150_000.0,
            "target_flow_rate": 2.0,
            "pump_speed": 1_200.0,
        }
    )
    unknowns = MappingProxyType({"flow_rate": 3.0})
    parameters = MappingProxyType({"resistance": 10_000.0, "pump_gain": 40.0})

    residuals = element.residuals(inputs=inputs, unknowns=unknowns, parameters=parameters)
    jacobian = element.jacobian(inputs=inputs, unknowns=unknowns, parameters=parameters)

    assert residuals["mass_balance"] == 1.0
    assert residuals["head_balance"] == 68_000.0
    assert jacobian[("mass_balance", "flow_rate")] == 1.0
    assert jacobian[("head_balance", "flow_rate")] == -10_000.0


def test_jacobian_contract_uses_residual_and_unknown_name_pairs() -> None:
    element = DemoHydraulicElement()

    jacobian = element.jacobian(
        inputs=MappingProxyType(
            {
                "upstream_pressure": 200_000.0,
                "downstream_pressure": 150_000.0,
                "target_flow_rate": 2.0,
                "pump_speed": 1_200.0,
            }
        ),
        unknowns=MappingProxyType({"flow_rate": 3.0}),
        parameters=MappingProxyType({"resistance": 10_000.0, "pump_gain": 40.0}),
    )

    assert set(jacobian.keys()) == {
        ("mass_balance", "flow_rate"),
        ("head_balance", "flow_rate"),
    }
    assert all(isinstance(value, float) for value in jacobian.values())
