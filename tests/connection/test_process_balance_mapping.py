from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

import pytest

from modular_simulation.connection.network import ConnectionNetwork
from modular_simulation.connection.process_binding import ProcessBinding


def _compiled_reactor_binding() -> ProcessBinding:
    network = ConnectionNetwork()
    network.add_process("reactor", inlet_ports=("feed",), outlet_ports=("product",))
    compiled = network.compile()
    return compiled.process_bindings["reactor"]


def test_binding_sign_projection_for_forward_process_flow() -> None:
    binding = _compiled_reactor_binding()
    binding.bind_inlets({"feed": 1.2})
    binding.bind_outlets(
        {
            "product": {
                "role": "outlet",
                "flow": 0.9,
                "composition": (0.8, 0.2),
            }
        }
    )

    outlet = binding.get_outlet("product")
    assert outlet["role"] == "outlet"
    assert outlet["flow"] == pytest.approx(0.9)
    assert outlet["normalized_inlet_flow"] == pytest.approx(-0.9)
    assert outlet["inlet_flow"] == pytest.approx(0.0)
    assert outlet["outlet_flow"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("role", "flow", "expected_inlet", "expected_outlet"),
    [
        ("inlet", 2.0, 2.0, 0.0),
        ("inlet", -2.0, 0.0, 2.0),
        ("outlet", 2.0, 0.0, 2.0),
        ("outlet", -2.0, 2.0, 0.0),
    ],
)
def test_binding_sign_normalization_is_stable_under_reversal(
    role: str,
    flow: float,
    expected_inlet: float,
    expected_outlet: float,
) -> None:
    binding = _compiled_reactor_binding()
    binding.bind_inlets({"feed": 0.3})
    binding.bind_outlets(
        {
            "product": {
                "role": role,
                "flow": flow,
                "composition": (0.5, 0.5),
            }
        }
    )

    first = binding.get_outlet("product")
    second = binding.get_outlet("product")

    assert first == second
    assert first["normalized_inlet_flow"] == pytest.approx(flow if role == "inlet" else -flow)
    assert first["inlet_flow"] == pytest.approx(expected_inlet)
    assert first["outlet_flow"] == pytest.approx(expected_outlet)


def test_binding_preserves_declared_composition_payload() -> None:
    binding = _compiled_reactor_binding()
    binding.bind_inlets({"feed": 0.6})
    binding.bind_outlets(
        {
            "product": {
                "role": "inlet",
                "flow": -0.7,
                "composition": (0.25, 0.75),
            }
        }
    )

    outlet = binding.get_outlet("product")
    assert outlet["composition"] == pytest.approx((0.25, 0.75))
    assert outlet["inlet_flow"] == pytest.approx(0.0)
    assert outlet["outlet_flow"] == pytest.approx(0.7)
