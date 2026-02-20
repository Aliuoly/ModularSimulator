from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

import pytest

from modular_simulation.connection.network import ConnectionNetwork
from modular_simulation.connection.process_binding import BindingError, ProcessBinding


def _compiled_reactor_binding() -> ProcessBinding:
    network = ConnectionNetwork()
    network.add_process("reactor", inlet_ports=("feed",), outlet_ports=("product",))
    compiled = network.compile()
    return compiled.process_bindings["reactor"]


def test_network_compile_exposes_explicit_process_binding() -> None:
    binding = _compiled_reactor_binding()

    binding.bind_inlets({"feed": 1.25})
    binding.bind_outlets(
        {
            "product": {
                "role": "outlet",
                "flow": 0.9,
                "composition": (0.4, 0.6),
            }
        }
    )

    assert binding.get_outlet("product") == {
        "role": "outlet",
        "flow": 0.9,
        "composition": (0.4, 0.6),
        "normalized_inlet_flow": -0.9,
        "inlet_flow": 0.0,
        "outlet_flow": 0.9,
    }


def test_binding_api_raises_actionable_errors_for_invalid_ports() -> None:
    binding = _compiled_reactor_binding()

    with pytest.raises(BindingError) as unknown_inlet_error:
        binding.set_inlet("invalid", 1.0)
    assert (
        str(unknown_inlet_error.value) == "Unknown inlet port 'invalid'. Expected one of: ['feed']"
    )

    with pytest.raises(BindingError) as unknown_outlet_error:
        _ = binding.get_outlet("missing")
    assert str(unknown_outlet_error.value) == (
        "Unknown outlet port 'missing'. Expected one of: ['product']"
    )


def test_binding_validation_requires_explicit_inlet_and_outlet_bindings() -> None:
    binding = _compiled_reactor_binding()

    with pytest.raises(BindingError) as unbound_inlet_error:
        binding.validate()
    assert str(unbound_inlet_error.value) == "Unbound inlet ports: ['feed']"

    binding.bind_inlets({"feed": 1.0})
    with pytest.raises(BindingError) as unbound_outlet_error:
        binding.validate()
    assert str(unbound_outlet_error.value) == "Unbound outlet ports: ['product']"
