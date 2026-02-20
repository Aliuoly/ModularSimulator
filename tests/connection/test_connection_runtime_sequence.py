from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from typing import cast

import numpy as np
import pytest

from modular_simulation.connection.connection_layer import (
    PicardIterationGateConfig,
    TransportAndMixingUpdate,
)
from modular_simulation.connection.hydraulic_compile import HydraulicCompileLifecycle
from modular_simulation.connection.hydraulic_solver import (
    HydraulicSolveResult,
    HydraulicSystemDefinition,
    LinearResidualEquation,
)
from modular_simulation.connection.junction import JunctionMixingResult
from modular_simulation.connection.network import ConnectionNetwork
from modular_simulation.connection.runtime import (
    ConnectionRuntimeOrchestrator,
    ConnectionRuntimeStepResult,
)
from modular_simulation.connection.state import MaterialState, PortCondition
from modular_simulation.connection.transport import LagTransportState, LagTransportUpdateResult


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


def _hydraulic_result(*, edge_flow: float, residual_norm: float = 1.0e-10) -> HydraulicSolveResult:
    return HydraulicSolveResult(
        converged=True,
        iterations=2,
        residual_norm=residual_norm,
        solution_vector=np.asarray([edge_flow, 5.0], dtype=np.float64),
        unknowns={"flow": edge_flow, "pressure": 5.0},
        unknown_index_map={"flow": 0, "pressure": 1},
        residual_index_map={"balance": 0, "ref": 1},
    )


def _material_state(*, pressure: float, temperature: float) -> MaterialState:
    return MaterialState(
        pressure=pressure,
        temperature=temperature,
        mole_fractions=(0.6, 0.4),
    )


def _transport_result(*, held_for_near_zero_flow: bool) -> LagTransportUpdateResult:
    return LagTransportUpdateResult(
        state=LagTransportState(composition=(0.6, 0.4), temperature=315.0),
        update_fraction=0.0 if held_for_near_zero_flow else 0.5,
        flow_scale=0.0 if held_for_near_zero_flow else 1.0,
        flow_sign_changed=False,
        held_for_near_zero_flow=held_for_near_zero_flow,
    )


def test_runtime_step_uses_fixed_macro_sequence_and_deterministic_bookkeeping() -> None:
    call_order: list[str] = []

    def hydraulic_solve_step(
        *,
        network: ConnectionNetwork,
        macro_step_time_s: float,
    ) -> HydraulicSolveResult:
        assert network._compiled is not None
        assert network._compiled.graph_revision == "graph_rev_0001"
        assert macro_step_time_s == pytest.approx(1.25)
        call_order.append("hydraulics_solve")
        return _hydraulic_result(edge_flow=0.8)

    def transport_and_mixing_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        assert network._compiled is not None
        assert hydraulic_result.unknowns["flow"] == pytest.approx(0.8)
        assert macro_step_time_s == pytest.approx(1.25)
        call_order.append("transport_update")
        return TransportAndMixingUpdate(
            transport_results={
                "edge_main": _transport_result(held_for_near_zero_flow=False),
                "edge_zero": _transport_result(held_for_near_zero_flow=True),
            },
            junction_results={
                "junction_a": JunctionMixingResult(
                    state=_material_state(pressure=101325.0, temperature=315.0),
                    total_incoming_flow_rate=0.0,
                    used_fallback=True,
                )
            },
        )

    def boundary_propagation_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        assert network._compiled is not None
        assert hydraulic_result.unknowns["flow"] == pytest.approx(0.8)
        assert "edge_main" in transport_and_mixing_update.transport_results
        assert macro_step_time_s == pytest.approx(1.25)
        call_order.append("boundary_state_propagation")
        return {
            "reactor_inlet": PortCondition(
                state=_material_state(pressure=120000.0, temperature=315.0),
                through_molar_flow_rate=0.8,
            ),
            "feed": PortCondition(
                state=_material_state(pressure=130000.0, temperature=320.0),
                through_molar_flow_rate=0.8,
            ),
        }

    runtime = ConnectionRuntimeOrchestrator(
        hydraulic_solve_step=hydraulic_solve_step,
        transport_and_mixing_step=transport_and_mixing_step,
        boundary_propagation_step=boundary_propagation_step,
    )
    network = _network(runtime)

    result = cast(ConnectionRuntimeStepResult, network.step(macro_step_time_s=1.25))

    assert call_order == [
        "hydraulics_solve",
        "transport_update",
        "boundary_state_propagation",
    ]
    assert result.macro_step_index == 1
    assert result.simulation_time_s == pytest.approx(1.25)
    assert result.graph_revision == "graph_rev_0001"
    assert result.bookkeeping.executed_sequence == (
        "hydraulics_solve",
        "transport_update",
        "boundary_state_propagation",
    )
    assert result.bookkeeping.updated_port_keys == ("feed", "reactor_inlet")
    assert result.bookkeeping.transport_fallback_keys == ("edge_zero",)
    assert result.bookkeeping.junction_fallback_keys == ("junction_a",)
    assert result.bookkeeping.used_fallback
    assert result.barrier_reconfiguration.applied_request_ids == ()
    assert result.barrier_reconfiguration.skipped_request_ids == ()


def test_runtime_step_preserves_picard_gate_metadata_deterministically() -> None:
    boundary_flows = iter((1.0, 0.40, 0.10, 0.1000000001))
    call_counts = {"hydraulics": 0, "transport": 0, "boundary": 0}

    def hydraulic_solve_step(
        *,
        network: ConnectionNetwork,
        macro_step_time_s: float,
    ) -> HydraulicSolveResult:
        del network, macro_step_time_s
        call_counts["hydraulics"] += 1
        return _hydraulic_result(edge_flow=1.0, residual_norm=5.0)

    def transport_and_mixing_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        del network, hydraulic_result, macro_step_time_s
        call_counts["transport"] += 1
        return TransportAndMixingUpdate(
            transport_results={"edge_main": _transport_result(held_for_near_zero_flow=False)}
        )

    def boundary_propagation_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        del network, hydraulic_result, transport_and_mixing_update, macro_step_time_s
        call_counts["boundary"] += 1
        return {
            "inlet": PortCondition(
                state=_material_state(pressure=101325.0, temperature=315.0),
                through_molar_flow_rate=next(boundary_flows),
            )
        }

    runtime = ConnectionRuntimeOrchestrator(
        hydraulic_solve_step=hydraulic_solve_step,
        transport_and_mixing_step=transport_and_mixing_step,
        boundary_propagation_step=boundary_propagation_step,
        picard_gate_config=PicardIterationGateConfig(
            enabled=True,
            residual_threshold=0.1,
            max_iterations=5,
            tolerance=1.0e-6,
        ),
    )
    network = _network(runtime)

    result = cast(ConnectionRuntimeStepResult, network.step(macro_step_time_s=2.0))

    assert call_counts == {"hydraulics": 4, "transport": 4, "boundary": 4}
    assert result.bookkeeping.picard_gate_enabled
    assert result.bookkeeping.picard_iterations_used == 3
    assert result.bookkeeping.picard_converged
    assert result.bookkeeping.picard_last_residual == pytest.approx(1.0e-10)
    assert result.bookkeeping.executed_sequence == (
        "hydraulics_solve",
        "transport_update",
        "boundary_state_propagation",
    )


def test_reconfiguration_commit_rejection_keeps_time_and_active_revision_unchanged() -> None:
    runtime = ConnectionRuntimeOrchestrator()
    network = _network(runtime)

    _ = network.queue_reconfiguration(
        {
            "operation": "batch_update",
            "mutations": [
                {
                    "operation": "add_connection",
                    "source": "unknown_source",
                    "target": "reactor.feed",
                }
            ],
        }
    )
    before_snapshot = network.save_runtime_snapshot()
    before_revision = network._compiled.graph_revision if network._compiled is not None else None

    with pytest.raises(
        RuntimeError,
        match=(
            "reconfiguration transaction rollback: request 'rq_0001' mutation 1 failed: "
            "unknown endpoint node 'unknown_source'"
        ),
    ):
        _ = network.step(macro_step_time_s=1.0)

    after_snapshot = network.save_runtime_snapshot()
    after_revision = network._compiled.graph_revision if network._compiled is not None else None

    assert after_snapshot == before_snapshot
    assert after_revision == before_revision
