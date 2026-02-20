from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from typing import cast

import numpy as np
import pytest

from modular_simulation.connection.connection_layer import (
    MACRO_COUPLING_SEQUENCE,
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


def _hydraulic_result(*, edge_flow: float, node_pressure: float = 101325.0) -> HydraulicSolveResult:
    return HydraulicSolveResult(
        converged=True,
        iterations=2,
        residual_norm=1.0e-10,
        solution_vector=np.asarray([edge_flow, node_pressure], dtype=np.float64),
        unknowns={"edge_flow": edge_flow, "node_pressure": node_pressure},
        unknown_index_map={"edge_flow": 0, "node_pressure": 1},
        residual_index_map={"mass_balance": 0, "pressure_reference": 1},
    )


def _transport_update_result(
    *,
    composition: tuple[float, ...],
    temperature: float,
    held_for_near_zero_flow: bool,
    update_fraction: float = 0.5,
    flow_scale: float = 1.0,
) -> LagTransportUpdateResult:
    return LagTransportUpdateResult(
        state=LagTransportState(composition=composition, temperature=temperature),
        update_fraction=update_fraction,
        flow_scale=flow_scale,
        flow_sign_changed=False,
        held_for_near_zero_flow=held_for_near_zero_flow,
    )


def _material_state(
    *,
    pressure: float,
    temperature: float,
    mole_fractions: tuple[float, ...] = (0.6, 0.4),
) -> MaterialState:
    return MaterialState(
        pressure=pressure,
        temperature=temperature,
        mole_fractions=mole_fractions,
    )


def test_macro_sequence_ordering_and_outputs_are_deterministic() -> None:
    call_order: list[str] = []
    boundary_outputs: list[dict[str, PortCondition]] = []

    def hydraulic_solve_step(
        *,
        network: ConnectionNetwork,
        macro_step_time_s: float,
    ) -> HydraulicSolveResult:
        assert network._compiled is not None
        assert macro_step_time_s == pytest.approx(12.0)
        call_order.append("hydraulics_solve")
        return _hydraulic_result(edge_flow=0.75)

    def transport_and_mixing_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        assert network._compiled is not None
        assert macro_step_time_s == pytest.approx(12.0)
        assert hydraulic_result.unknowns["edge_flow"] == pytest.approx(0.75)
        call_order.append("transport_update")
        return TransportAndMixingUpdate(
            transport_results={
                "edge_b": _transport_update_result(
                    composition=(0.2, 0.8),
                    temperature=340.0,
                    held_for_near_zero_flow=False,
                ),
                "edge_a": _transport_update_result(
                    composition=(0.7, 0.3),
                    temperature=315.0,
                    held_for_near_zero_flow=False,
                ),
            }
        )

    def boundary_propagation_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        assert network._compiled is not None
        assert macro_step_time_s == pytest.approx(12.0)
        assert hydraulic_result.unknowns["edge_flow"] == pytest.approx(0.75)
        assert tuple(transport_and_mixing_update.transport_results.keys()) == ("edge_b", "edge_a")
        call_order.append("boundary_state_propagation")
        output = {
            "reactor_inlet": PortCondition(
                state=_material_state(
                    pressure=120000.0, temperature=315.0, mole_fractions=(0.7, 0.3)
                ),
                through_molar_flow_rate=hydraulic_result.unknowns["edge_flow"],
                macro_step_time_s=macro_step_time_s,
            )
        }
        boundary_outputs.append(output)
        return output

    runtime = ConnectionRuntimeOrchestrator(
        hydraulic_solve_step=hydraulic_solve_step,
        transport_and_mixing_step=transport_and_mixing_step,
        boundary_propagation_step=boundary_propagation_step,
    )
    network = _network(runtime)

    result_a = cast(ConnectionRuntimeStepResult, network.step(macro_step_time_s=12.0))
    result_b = cast(ConnectionRuntimeStepResult, network.step(macro_step_time_s=12.0))

    assert call_order == [
        "hydraulics_solve",
        "transport_update",
        "boundary_state_propagation",
        "hydraulics_solve",
        "transport_update",
        "boundary_state_propagation",
    ]
    assert result_a.bookkeeping.macro_step_index == 1
    assert result_b.bookkeeping.macro_step_index == 2
    assert result_a.bookkeeping.executed_sequence == MACRO_COUPLING_SEQUENCE
    assert result_a.bookkeeping.executed_sequence == result_b.bookkeeping.executed_sequence
    assert result_a.bookkeeping.updated_port_keys == result_b.bookkeeping.updated_port_keys
    assert (
        result_a.bookkeeping.transport_fallback_keys == result_b.bookkeeping.transport_fallback_keys
    )
    assert (
        result_a.bookkeeping.junction_fallback_keys == result_b.bookkeeping.junction_fallback_keys
    )
    assert result_a.bookkeeping.used_fallback == result_b.bookkeeping.used_fallback
    assert boundary_outputs[0] == boundary_outputs[1]


def test_macro_step_uses_current_hydraulics_result_for_transport_and_boundaries() -> None:
    edge_flows = iter((0.20, 0.85))
    transport_update_fractions: list[float] = []
    boundary_flow_rates: list[float] = []
    boundary_temperatures: list[float] = []

    def hydraulic_solve_step(
        *,
        network: ConnectionNetwork,
        macro_step_time_s: float,
    ) -> HydraulicSolveResult:
        del network
        del macro_step_time_s
        return _hydraulic_result(edge_flow=next(edge_flows))

    def transport_and_mixing_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        del network
        del macro_step_time_s
        edge_flow = hydraulic_result.unknowns["edge_flow"]
        transport_update_fractions.append(edge_flow)
        return TransportAndMixingUpdate(
            transport_results={
                "edge_main": _transport_update_result(
                    composition=(0.5, 0.5),
                    temperature=300.0 + 50.0 * edge_flow,
                    update_fraction=edge_flow,
                    flow_scale=1.0,
                    held_for_near_zero_flow=False,
                )
            }
        )

    def boundary_propagation_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        del network
        del macro_step_time_s
        edge_flow = hydraulic_result.unknowns["edge_flow"]
        edge_update = cast(
            LagTransportUpdateResult,
            transport_and_mixing_update.transport_results["edge_main"],
        )
        edge_state = edge_update.state
        boundary_flow_rates.append(edge_flow)
        boundary_temperatures.append(edge_state.temperature)
        return {
            "feed": PortCondition(
                state=_material_state(
                    pressure=101325.0 + 1000.0 * edge_flow,
                    temperature=edge_state.temperature,
                    mole_fractions=edge_state.composition,
                ),
                through_molar_flow_rate=edge_flow,
            )
        }

    runtime = ConnectionRuntimeOrchestrator(
        hydraulic_solve_step=hydraulic_solve_step,
        transport_and_mixing_step=transport_and_mixing_step,
        boundary_propagation_step=boundary_propagation_step,
    )
    network = _network(runtime)

    first = cast(ConnectionRuntimeStepResult, network.step(macro_step_time_s=0.1))
    second = cast(ConnectionRuntimeStepResult, network.step(macro_step_time_s=0.2))

    assert first.macro_step_index == 1
    assert second.macro_step_index == 2
    assert transport_update_fractions == pytest.approx([0.20, 0.85])
    assert boundary_flow_rates == pytest.approx([0.20, 0.85])
    assert boundary_temperatures == pytest.approx([310.0, 342.5])


def test_macro_step_bookkeeping_marks_transport_and_junction_fallbacks() -> None:
    def hydraulic_solve_step(
        *,
        network: ConnectionNetwork,
        macro_step_time_s: float,
    ) -> HydraulicSolveResult:
        del network
        del macro_step_time_s
        return _hydraulic_result(edge_flow=0.0)

    def transport_and_mixing_step(
        *,
        network: ConnectionNetwork,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        del network
        del hydraulic_result, macro_step_time_s
        return TransportAndMixingUpdate(
            transport_results={
                "edge_zero": _transport_update_result(
                    composition=(0.4, 0.6),
                    temperature=305.0,
                    held_for_near_zero_flow=True,
                    update_fraction=0.0,
                    flow_scale=0.0,
                )
            },
            junction_results={
                "junction_a": JunctionMixingResult(
                    state=_material_state(
                        pressure=105000.0,
                        temperature=305.0,
                        mole_fractions=(0.4, 0.6),
                    ),
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
        del network
        del hydraulic_result, transport_and_mixing_update, macro_step_time_s
        return {
            "recycle": PortCondition(
                state=_material_state(
                    pressure=105000.0, temperature=305.0, mole_fractions=(0.4, 0.6)
                ),
                through_molar_flow_rate=0.0,
            )
        }

    runtime = ConnectionRuntimeOrchestrator(
        hydraulic_solve_step=hydraulic_solve_step,
        transport_and_mixing_step=transport_and_mixing_step,
        boundary_propagation_step=boundary_propagation_step,
    )
    network = _network(runtime)

    result = cast(ConnectionRuntimeStepResult, network.step(macro_step_time_s=5.0))

    assert result.bookkeeping.executed_sequence == MACRO_COUPLING_SEQUENCE
    assert result.bookkeeping.transport_fallback_keys == ("edge_zero",)
    assert result.bookkeeping.junction_fallback_keys == ("junction_a",)
    assert result.bookkeeping.used_fallback
