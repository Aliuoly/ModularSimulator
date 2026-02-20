from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from collections.abc import Mapping
import pytest
from typing import cast

from modular_simulation.connection.hydraulic_compile import HydraulicCompileLifecycle
from modular_simulation.connection.hydraulic_solver import (
    HydraulicSystemDefinition,
    LinearResidualEquation,
)
from modular_simulation.connection.network import ConnectionNetwork
from modular_simulation.connection.runtime import (
    ConnectionRuntimeOrchestrator,
    ConnectionRuntimeStepResult,
)
from modular_simulation.connection.runtime_persistence import (
    RUNTIME_SNAPSHOT_SCHEMA_MARKER,
    RUNTIME_SNAPSHOT_SCHEMA_VERSION,
    restore_runtime_snapshot_contract,
    save_runtime_snapshot_contract,
    validate_runtime_snapshot_contract,
)


def _linear_equation(
    residual_name: str,
    coefficients: dict[str, float],
    *,
    constant: float = 0.0,
) -> LinearResidualEquation:
    return LinearResidualEquation(
        residual_name=residual_name,
        coefficients=coefficients,
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


def _network(runtime: ConnectionRuntimeOrchestrator) -> ConnectionNetwork:
    network = ConnectionNetwork(
        compile_lifecycle=HydraulicCompileLifecycle(),
        hydraulic_system_builder=lambda topology: _valid_system(),
        runtime_orchestrator=runtime,
    )
    network.add_process("reactor", inlet_ports=("feed",), outlet_ports=("product",))
    network.add_boundary_source("feed_boundary")
    network.add_boundary_sink("product_boundary")
    network.connect("feed_boundary", "reactor.feed")
    network.connect("reactor.product", "product_boundary")
    _ = network.compile()
    return network


def test_runtime_snapshot_round_trip_restores_queue_order_and_indexes() -> None:
    runtime = ConnectionRuntimeOrchestrator()
    network = _network(runtime)
    process_runtime_state: dict[str, object] = {
        "reactor": {"temperature": 345.0, "conversion": 0.34}
    }

    first_request = network.queue_reconfiguration(
        {
            "operation": "add_boundary_sink",
            "boundary_id": "purge_sink",
            "idempotency_key": "sink-key",
        }
    )
    second_request = network.queue_reconfiguration(
        {
            "operation": "batch_update",
            "mutations": [
                {
                    "operation": "add_process_port",
                    "process_id": "reactor",
                    "direction": "outlet",
                    "port_name": "purge",
                },
                {
                    "operation": "add_connection",
                    "source": "reactor.purge",
                    "target": "purge_sink",
                },
            ],
        }
    )
    assert (first_request, second_request) == ("rq_0001", "rq_0002")

    snapshot = save_runtime_snapshot_contract(
        runtime=runtime,
        network=network,
        process_runtime_state=process_runtime_state,
    )
    validated = validate_runtime_snapshot_contract(snapshot)
    assert validated.macro_step_index == 0
    assert validated.simulation_time_s == pytest.approx(0.0)

    runtime._simulation_time_s = 9.5
    runtime._macro_step_index = 19
    process_runtime_state["reactor"] = {"temperature": 300.0, "conversion": 0.10}
    _ = network.queue_reconfiguration(
        {
            "operation": "add_boundary_sink",
            "boundary_id": "waste_sink",
        }
    )

    restore_runtime_snapshot_contract(
        runtime=runtime,
        network=network,
        process_runtime_state_sink=process_runtime_state,
        snapshot=snapshot,
    )

    assert runtime._simulation_time_s == pytest.approx(0.0)
    assert runtime._macro_step_index == 0
    assert runtime._macro_step_active is False
    assert network._next_reconfiguration_index == 3
    assert network._next_revision_index == 2
    assert tuple(item.request_id for item in network._queued_reconfigurations) == (
        "rq_0001",
        "rq_0002",
    )
    assert process_runtime_state == {"reactor": {"conversion": 0.34, "temperature": 345.0}}


def test_runtime_resume_preserves_monotonic_time_and_step_index_continuity() -> None:
    runtime = ConnectionRuntimeOrchestrator()
    network = _network(runtime)
    process_runtime_state: dict[str, object] = {}

    _ = network.step(macro_step_time_s=1.5)
    _ = network.step(macro_step_time_s=0.5)
    snapshot = save_runtime_snapshot_contract(
        runtime=runtime,
        network=network,
        process_runtime_state=process_runtime_state,
    )

    _ = network.step(macro_step_time_s=0.25)

    restore_runtime_snapshot_contract(
        runtime=runtime,
        network=network,
        process_runtime_state_sink=process_runtime_state,
        snapshot=snapshot,
    )
    resumed = cast(ConnectionRuntimeStepResult, network.step(macro_step_time_s=0.25))

    assert resumed.simulation_time_s == pytest.approx(2.25)
    assert resumed.macro_step_index == 3


def test_save_runtime_snapshot_rejects_active_macro_step_and_allows_barrier_capture() -> None:
    runtime = ConnectionRuntimeOrchestrator()
    network = _network(runtime)

    runtime._macro_step_active = True
    with pytest.raises(
        RuntimeError,
        match="runtime snapshot capture is unavailable during active macro-step execution",
    ):
        _ = network.save_runtime_snapshot()

    runtime._macro_step_active = False
    snapshot = network.save_runtime_snapshot()
    assert snapshot["schema"] == "connection_runtime_v1"


def test_restore_rejects_invalid_schema_without_mutating_active_state() -> None:
    runtime = ConnectionRuntimeOrchestrator()
    network = _network(runtime)
    process_runtime_state: dict[str, object] = {"reactor": {"temperature": 320.0}}
    _ = network.queue_reconfiguration(
        {
            "operation": "add_boundary_sink",
            "boundary_id": "purge_sink",
        }
    )
    before_restore = save_runtime_snapshot_contract(
        runtime=runtime,
        network=network,
        process_runtime_state=process_runtime_state,
    )
    invalid_snapshot = dict(before_restore)
    invalid_snapshot["schema_version"] = 999

    with pytest.raises(
        ValueError,
        match="invalid runtime snapshot payload: field 'schema_version' does not match expected version",
    ):
        restore_runtime_snapshot_contract(
            runtime=runtime,
            network=network,
            process_runtime_state_sink=process_runtime_state,
            snapshot=invalid_snapshot,
        )

    after_restore = save_runtime_snapshot_contract(
        runtime=runtime,
        network=network,
        process_runtime_state=process_runtime_state,
    )
    assert after_restore == before_restore


def test_restore_requires_barrier_inactive_and_is_atomic_when_process_validation_fails() -> None:
    runtime = ConnectionRuntimeOrchestrator()
    network = _network(runtime)
    process_runtime_state: dict[str, object] = {"reactor": {"temperature": 350.0}}
    snapshot = save_runtime_snapshot_contract(
        runtime=runtime,
        network=network,
        process_runtime_state=process_runtime_state,
    )

    runtime._macro_step_active = True
    with pytest.raises(
        RuntimeError,
        match="runtime snapshot restore is unavailable during active macro-step execution",
    ):
        restore_runtime_snapshot_contract(
            runtime=runtime,
            network=network,
            process_runtime_state_sink=process_runtime_state,
            snapshot=snapshot,
        )
    runtime._macro_step_active = False

    before_restore = save_runtime_snapshot_contract(
        runtime=runtime,
        network=network,
        process_runtime_state=process_runtime_state,
    )

    def _raise_on_process_state(payload: Mapping[str, object]) -> Mapping[str, object]:
        del payload
        raise ValueError("process state is invalid")

    with pytest.raises(ValueError, match="process state is invalid"):
        restore_runtime_snapshot_contract(
            runtime=runtime,
            network=network,
            process_runtime_state_sink=process_runtime_state,
            snapshot=snapshot,
            process_runtime_validator=_raise_on_process_state,
        )

    after_restore = save_runtime_snapshot_contract(
        runtime=runtime,
        network=network,
        process_runtime_state=process_runtime_state,
    )
    assert after_restore == before_restore


def test_schema_marker_and_version_are_explicit_and_stable() -> None:
    runtime = ConnectionRuntimeOrchestrator()
    network = _network(runtime)
    snapshot = save_runtime_snapshot_contract(
        runtime=runtime,
        network=network,
        process_runtime_state={},
    )

    assert snapshot["schema_marker"] == RUNTIME_SNAPSHOT_SCHEMA_MARKER
    assert snapshot["schema_version"] == RUNTIME_SNAPSHOT_SCHEMA_VERSION
