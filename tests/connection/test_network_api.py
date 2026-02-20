from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from types import MappingProxyType

import pytest

from modular_simulation.connection.hydraulic_compile import HydraulicCompileLifecycle
from modular_simulation.connection.hydraulic_solver import (
    HydraulicSystemDefinition,
    LinearResidualEquation,
)
from modular_simulation.connection.network import ConnectionNetwork
from modular_simulation.connection.process_binding import ProcessBinding


def _linear_equation(
    residual_name: str,
    coefficients: dict[str, float],
    *,
    constant: float = 0.0,
) -> LinearResidualEquation:
    return LinearResidualEquation(
        residual_name=residual_name,
        coefficients=MappingProxyType(coefficients),
        constant=constant,
    )


def _valid_system() -> HydraulicSystemDefinition:
    return HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("ref", {"pressure": 1.0}, constant=-5.0),
            _linear_equation("balance", {"pressure": 1.0, "flow": -1.0}),
        ),
    )


def _network() -> ConnectionNetwork:
    network = ConnectionNetwork(
        compile_lifecycle=HydraulicCompileLifecycle(),
        hydraulic_system_builder=lambda topology: _valid_system(),
    )
    network.add_process("reactor", inlet_ports=("feed",), outlet_ports=("product",))
    network.add_boundary_source("feed_boundary")
    network.add_boundary_sink("product_boundary")
    network.connect("feed_boundary", "reactor.feed")
    network.connect("reactor.product", "product_boundary")
    return network


def test_public_api_entry_points_exist() -> None:
    network = ConnectionNetwork()

    assert callable(network.add_process)
    assert callable(network.add_boundary_source)
    assert callable(network.add_boundary_sink)
    assert callable(network.connect)
    assert callable(network.compile)
    assert callable(network.step)
    assert callable(network.queue_reconfiguration)
    assert callable(network.save_runtime_snapshot)
    assert callable(network.resume_from_snapshot)


def test_compile_delegates_to_topology_process_bindings_and_compile_artifact() -> None:
    compiled = _network().compile()

    assert compiled.topology.node_ids == ("feed_boundary", "product_boundary", "reactor")
    assert compiled.topology.edge_ids == ("edge_0001", "edge_0002")
    assert isinstance(compiled.process_bindings["reactor"], ProcessBinding)
    assert compiled.hydraulic is not None
    assert compiled.hydraulic.graph_revision == "graph_rev_0001"


def test_connect_rejects_malformed_endpoint_deterministically() -> None:
    network = ConnectionNetwork()
    network.add_process("reactor", inlet_ports=("feed",), outlet_ports=("product",))
    network.add_boundary_sink("sink")

    with pytest.raises(ValueError) as error:
        network.connect("reactor", "sink")

    assert str(error.value) == (
        "invalid endpoint 'reactor': process endpoints must use '<process_id>.<port_name>' format"
    )


def test_connect_duplicate_edge_has_actionable_error() -> None:
    network = ConnectionNetwork()
    network.add_process("reactor", inlet_ports=("feed",), outlet_ports=("product",))
    network.add_boundary_source("feed")

    network.connect("feed", "reactor.feed")

    with pytest.raises(ValueError) as error:
        network.connect("feed", "reactor.feed")

    assert str(error.value) == (
        "duplicate connection 'feed.outlet->reactor.feed' already exists as edge_id 'edge_0001'"
    )


def test_queue_reconfiguration_and_runtime_placeholders_are_deterministic() -> None:
    network = ConnectionNetwork()

    queued = network.queue_reconfiguration({"operation": "add_connection"})
    assert queued == "rq_0001"

    with pytest.raises(ValueError) as malformed_error:
        network.queue_reconfiguration({})
    assert str(malformed_error.value) == (
        "invalid reconfiguration request: expected mapping with non-empty string field 'operation'"
    )

    with pytest.raises(RuntimeError) as step_error:
        network.step(macro_step_time_s=1.0)
    assert str(step_error.value) == (
        "runtime step is unavailable: runtime orchestrator not configured for ConnectionNetwork"
    )

    with pytest.raises(RuntimeError) as snapshot_error:
        network.save_runtime_snapshot()
    assert str(snapshot_error.value) == (
        "runtime snapshot is unavailable: runtime orchestrator not configured for ConnectionNetwork"
    )

    with pytest.raises(ValueError) as payload_error:
        network.resume_from_snapshot(snapshot=[])
    assert str(payload_error.value) == "invalid snapshot payload: expected mapping"

    with pytest.raises(RuntimeError) as resume_error:
        network.resume_from_snapshot(snapshot={})
    assert str(resume_error.value) == (
        "runtime resume is unavailable: runtime orchestrator not configured for ConnectionNetwork"
    )
