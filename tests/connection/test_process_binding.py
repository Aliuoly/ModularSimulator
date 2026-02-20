import pytest
from modular_simulation.connection.process_binding import BindingError, ProcessBinding


def test_happy_binding_path():
    pb = ProcessBinding(inlet_ports=["in1", "in2"], outlet_ports=["out1", "out2"])
    pb.set_inlet("in1", 1.0)
    pb.bind_inlets({"in1": 1.0, "in2": -2.5})
    pb.bind_outlets(
        {
            "out1": {"role": "outlet", "flow": 3.14, "composition": (0.3, 0.7)},
            "out2": {"role": "inlet", "flow": -2.0, "composition": (0.4, 0.6)},
        }
    )
    outlet = pb.get_outlet("out1")
    assert outlet["role"] == "outlet"
    assert outlet["flow"] == 3.14
    assert outlet["composition"] == (0.3, 0.7)
    assert outlet["normalized_inlet_flow"] == -3.14
    assert outlet["inlet_flow"] == 0.0
    assert outlet["outlet_flow"] == 3.14
    pb.validate()


def test_unknown_inlet_port_error():
    pb = ProcessBinding(inlet_ports=["a"], outlet_ports=["b"])
    with pytest.raises(BindingError) as exc:
        pb.set_inlet("z", 1.0)
    assert str(exc.value) == "Unknown inlet port 'z'. Expected one of: ['a']"


def test_missing_inlet_binding_error():
    pb = ProcessBinding(inlet_ports=["x", "y"], outlet_ports=["u", "v"])
    with pytest.raises(BindingError) as exc:
        pb.bind_inlets({"x": 1.0})
    assert str(exc.value) == "Missing inlet bindings for ports: ['y']"


def test_initializer_mismatch_error():
    with pytest.raises(BindingError) as exc:
        _ = ProcessBinding(inlet_ports=["i1", "i2", "i3"], outlet_ports=["o1", "o2"])
    assert "Inlet/outlet port count mismatch" in str(exc.value)


def test_get_outlet_unbound_error():
    pb = ProcessBinding(inlet_ports=["in"], outlet_ports=["out"])
    pb.bind_inlets({"in": 1.0})
    with pytest.raises(BindingError) as exc:
        _ = pb.get_outlet("out")
    assert str(exc.value) == "Outlet 'out' has no binding yet."


def test_missing_outlet_binding_field_error_is_deterministic():
    pb = ProcessBinding(inlet_ports=["in"], outlet_ports=["out"])
    pb.bind_inlets({"in": 0.5})

    with pytest.raises(BindingError) as exc:
        pb.bind_outlets({"out": {"role": "outlet", "flow": 1.2}})

    assert str(exc.value) == "Missing outlet binding field(s) for 'out': ['composition']"


def test_unknown_outlet_binding_field_error_is_deterministic():
    pb = ProcessBinding(inlet_ports=["in"], outlet_ports=["out"])
    pb.bind_inlets({"in": 0.5})

    with pytest.raises(BindingError) as exc:
        pb.bind_outlets(
            {
                "out": {
                    "role": "outlet",
                    "flow": 1.2,
                    "composition": (1.0,),
                    "temperature": 300.0,
                }
            }
        )

    assert str(exc.value) == "Unknown outlet binding field(s) for 'out': ['temperature']"


def test_incompatible_composition_lengths_error_is_deterministic():
    pb = ProcessBinding(inlet_ports=["in1", "in2"], outlet_ports=["out1", "out2"])
    pb.bind_inlets({"in1": 0.5, "in2": 0.6})

    with pytest.raises(BindingError) as exc:
        pb.bind_outlets(
            {
                "out1": {"role": "outlet", "flow": 1.2, "composition": (0.5, 0.5)},
                "out2": {"role": "inlet", "flow": -0.2, "composition": (1.0,)},
            }
        )

    assert (
        str(exc.value)
        == "Incompatible composition lengths across outlet bindings: {'out1': 2, 'out2': 1}"
    )


@pytest.mark.parametrize(
    ("role", "flow", "expected_inlet", "expected_outlet"),
    [
        ("inlet", 2.0, 2.0, 0.0),
        ("inlet", -2.0, 0.0, 2.0),
        ("outlet", 2.0, 0.0, 2.0),
        ("outlet", -2.0, 2.0, 0.0),
    ],
)
def test_sign_normalization_matches_legacy_balance_mapping(
    role: str,
    flow: float,
    expected_inlet: float,
    expected_outlet: float,
):
    pb = ProcessBinding(inlet_ports=["in"], outlet_ports=["out"])
    pb.bind_inlets({"in": 0.1})
    pb.bind_outlets({"out": {"role": role, "flow": flow, "composition": (1.0,)}})

    outlet_binding = pb.get_outlet("out")
    assert outlet_binding["normalized_inlet_flow"] == (flow if role == "inlet" else -flow)
    assert outlet_binding["inlet_flow"] == expected_inlet
    assert outlet_binding["outlet_flow"] == expected_outlet
