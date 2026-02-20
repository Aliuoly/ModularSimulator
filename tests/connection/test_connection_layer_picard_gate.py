from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from modular_simulation.connection.connection_layer import (
    MACRO_COUPLING_SEQUENCE,
    PicardIterationGateConfig,
    TransportAndMixingUpdate,
)
from modular_simulation.connection.hydraulic_compile import HydraulicCompileLifecycle
from modular_simulation.connection.hydraulic_solver import (
    HydraulicSolveResult,
    HydraulicSystemDefinition,
    LinearResidualEquation,
)
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


def _hydraulic_result(*, edge_flow: float, residual_norm: float) -> HydraulicSolveResult:
    return HydraulicSolveResult(
        converged=True,
        iterations=2,
        residual_norm=residual_norm,
        solution_vector=np.asarray([edge_flow, 101325.0], dtype=np.float64),
        unknowns={"edge_flow": edge_flow, "node_pressure": 101325.0},
        unknown_index_map={"edge_flow": 0, "node_pressure": 1},
        residual_index_map={"mass_balance": 0, "pressure_reference": 1},
    )


def _transport_update() -> TransportAndMixingUpdate:
    return TransportAndMixingUpdate(
        transport_results={
            "edge_main": LagTransportUpdateResult(
                state=LagTransportState(composition=(0.6, 0.4), temperature=315.0),
                update_fraction=0.5,
                flow_scale=1.0,
                flow_sign_changed=False,
                held_for_near_zero_flow=False,
            )
        }
    )


def _port_condition(*, through_molar_flow_rate: float) -> PortCondition:
    return PortCondition(
        state=MaterialState(
            pressure=101325.0,
            temperature=315.0,
            mole_fractions=(0.6, 0.4),
        ),
        through_molar_flow_rate=through_molar_flow_rate,
    )


def test_gate_off_for_weak_coupling_keeps_single_pass() -> None:
    call_counts = {"hydraulics": 0, "transport": 0, "boundary": 0}

    def hydraulic_solve_step(
        *,
        network: ConnectionNetwork,
        macro_step_time_s: float,
    ) -> HydraulicSolveResult:
        del network, macro_step_time_s
        call_counts["hydraulics"] += 1
        return _hydraulic_result(edge_flow=0.25, residual_norm=1.0e-9)

    def transport_and_mixing_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        del network, hydraulic_result, macro_step_time_s
        call_counts["transport"] += 1
        return _transport_update()

    def boundary_propagation_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        del network, hydraulic_result, transport_and_mixing_update, macro_step_time_s
        call_counts["boundary"] += 1
        return {"inlet": _port_condition(through_molar_flow_rate=0.25)}

    runtime = ConnectionRuntimeOrchestrator(
        hydraulic_solve_step=hydraulic_solve_step,
        transport_and_mixing_step=transport_and_mixing_step,
        boundary_propagation_step=boundary_propagation_step,
        picard_gate_config=PicardIterationGateConfig(
            enabled=True,
            residual_threshold=1.0e-6,
            max_iterations=4,
            tolerance=1.0e-12,
        ),
    )
    network = _network(runtime)

    result = cast(ConnectionRuntimeStepResult, network.step(macro_step_time_s=1.0))

    assert call_counts == {"hydraulics": 1, "transport": 1, "boundary": 1}
    assert not result.bookkeeping.picard_gate_enabled
    assert result.bookkeeping.picard_iterations_used == 0
    assert result.bookkeeping.picard_converged
    assert result.bookkeeping.picard_last_residual is None


def test_gate_on_for_strong_coupling_iterates_until_tolerance() -> None:
    call_counts = {"hydraulics": 0, "transport": 0, "boundary": 0}
    boundary_flows = iter((1.0, 0.40, 0.10, 0.1000000001))

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
        return _transport_update()

    def boundary_propagation_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        del network, hydraulic_result, transport_and_mixing_update, macro_step_time_s
        call_counts["boundary"] += 1
        return {"inlet": _port_condition(through_molar_flow_rate=next(boundary_flows))}

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
    assert result.bookkeeping.executed_sequence == MACRO_COUPLING_SEQUENCE


def test_gate_stops_at_max_iterations_when_tolerance_not_reached() -> None:
    call_counts = {"hydraulics": 0, "transport": 0, "boundary": 0}
    boundary_flows = iter((1.0, 0.80, 0.70, 0.60))

    def hydraulic_solve_step(
        *,
        network: ConnectionNetwork,
        macro_step_time_s: float,
    ) -> HydraulicSolveResult:
        del network, macro_step_time_s
        call_counts["hydraulics"] += 1
        return _hydraulic_result(edge_flow=1.0, residual_norm=10.0)

    def transport_and_mixing_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        del network, hydraulic_result, macro_step_time_s
        call_counts["transport"] += 1
        return _transport_update()

    def boundary_propagation_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        del network, hydraulic_result, transport_and_mixing_update, macro_step_time_s
        call_counts["boundary"] += 1
        return {"inlet": _port_condition(through_molar_flow_rate=next(boundary_flows))}

    runtime = ConnectionRuntimeOrchestrator(
        hydraulic_solve_step=hydraulic_solve_step,
        transport_and_mixing_step=transport_and_mixing_step,
        boundary_propagation_step=boundary_propagation_step,
        picard_gate_config=PicardIterationGateConfig(
            enabled=True,
            residual_threshold=1.0,
            max_iterations=3,
            tolerance=1.0e-12,
        ),
    )
    network = _network(runtime)

    result = cast(ConnectionRuntimeStepResult, network.step(macro_step_time_s=3.0))

    assert call_counts == {"hydraulics": 4, "transport": 4, "boundary": 4}
    assert result.bookkeeping.picard_gate_enabled
    assert result.bookkeeping.picard_iterations_used == 3
    assert not result.bookkeeping.picard_converged
    assert result.bookkeeping.picard_last_residual == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: PicardIterationGateConfig(
                enabled=cast(bool, cast(object, "yes")),
                residual_threshold=0.0,
                max_iterations=1,
                tolerance=1.0e-8,
            ),
            "enabled must be a bool",
        ),
        (
            lambda: PicardIterationGateConfig(
                enabled=True,
                residual_threshold=-1.0,
                max_iterations=1,
                tolerance=1.0e-8,
            ),
            "residual_threshold must be finite and non-negative",
        ),
        (
            lambda: PicardIterationGateConfig(
                enabled=True,
                residual_threshold=0.0,
                max_iterations=0,
                tolerance=1.0e-8,
            ),
            "max_iterations must be positive",
        ),
        (
            lambda: PicardIterationGateConfig(
                enabled=True,
                residual_threshold=0.0,
                max_iterations=1,
                tolerance=0.0,
            ),
            "tolerance must be finite and positive",
        ),
    ],
)
def test_invalid_picard_gate_config_raises_value_error(
    factory: Callable[[], PicardIterationGateConfig], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = factory()
