from __future__ import annotations

import importlib

import numpy as np
import pytest

connection_layer = importlib.import_module("modular_simulation.connection.connection_layer")
hydraulic_solver = importlib.import_module("modular_simulation.connection.hydraulic_solver")
junction = importlib.import_module("modular_simulation.connection.junction")
state = importlib.import_module("modular_simulation.connection.state")
transport = importlib.import_module("modular_simulation.connection.transport")

HydraulicSolveResult = hydraulic_solver.HydraulicSolveResult
JunctionMixingResult = junction.JunctionMixingResult
LagTransportState = transport.LagTransportState
LagTransportUpdateResult = transport.LagTransportUpdateResult
MACRO_COUPLING_SEQUENCE = connection_layer.MACRO_COUPLING_SEQUENCE
MaterialState = state.MaterialState
PortCondition = state.PortCondition
TransportAndMixingUpdate = connection_layer.TransportAndMixingUpdate
run_macro_coupling_step = connection_layer.run_macro_coupling_step


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

    def hydraulic_solve_step(*, macro_step_time_s: float) -> HydraulicSolveResult:
        assert macro_step_time_s == pytest.approx(12.0)
        call_order.append("hydraulics")
        return _hydraulic_result(edge_flow=0.75)

    def transport_update_step(
        *,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        assert macro_step_time_s == pytest.approx(12.0)
        assert hydraulic_result.unknowns["edge_flow"] == pytest.approx(0.75)
        call_order.append("transport")
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
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        assert macro_step_time_s == pytest.approx(12.0)
        assert hydraulic_result.unknowns["edge_flow"] == pytest.approx(0.75)
        assert tuple(transport_and_mixing_update.transport_results.keys()) == ("edge_b", "edge_a")
        call_order.append("boundary")
        return {
            "reactor_inlet": PortCondition(
                state=_material_state(
                    pressure=120000.0, temperature=315.0, mole_fractions=(0.7, 0.3)
                ),
                through_molar_flow_rate=hydraulic_result.unknowns["edge_flow"],
                macro_step_time_s=macro_step_time_s,
            )
        }

    result_a = run_macro_coupling_step(
        macro_step_time_s=12.0,
        macro_step_index=5,
        hydraulic_solve_step=hydraulic_solve_step,
        transport_update_step=transport_update_step,
        boundary_propagation_step=boundary_propagation_step,
    )
    result_b = run_macro_coupling_step(
        macro_step_time_s=12.0,
        macro_step_index=5,
        hydraulic_solve_step=hydraulic_solve_step,
        transport_update_step=transport_update_step,
        boundary_propagation_step=boundary_propagation_step,
    )

    assert call_order == [
        "hydraulics",
        "transport",
        "boundary",
        "hydraulics",
        "transport",
        "boundary",
    ]
    assert result_a.bookkeeping.executed_sequence == MACRO_COUPLING_SEQUENCE
    assert result_a.bookkeeping == result_b.bookkeeping
    assert result_a.port_conditions == result_b.port_conditions


def test_macro_step_uses_current_hydraulics_result_for_transport_and_boundaries() -> None:
    edge_flows = iter((0.20, 0.85))

    def hydraulic_solve_step(*, macro_step_time_s: float) -> HydraulicSolveResult:
        del macro_step_time_s
        return _hydraulic_result(edge_flow=next(edge_flows))

    def transport_update_step(
        *,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        del macro_step_time_s
        edge_flow = hydraulic_result.unknowns["edge_flow"]
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
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        del macro_step_time_s
        edge_flow = hydraulic_result.unknowns["edge_flow"]
        edge_state = transport_and_mixing_update.transport_results["edge_main"].state
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

    first = run_macro_coupling_step(
        macro_step_time_s=0.1,
        hydraulic_solve_step=hydraulic_solve_step,
        transport_update_step=transport_update_step,
        boundary_propagation_step=boundary_propagation_step,
    )
    second = run_macro_coupling_step(
        macro_step_time_s=0.2,
        hydraulic_solve_step=hydraulic_solve_step,
        transport_update_step=transport_update_step,
        boundary_propagation_step=boundary_propagation_step,
    )

    assert first.transport_and_mixing_update.transport_results[
        "edge_main"
    ].update_fraction == pytest.approx(0.20)
    assert first.port_conditions["feed"].through_molar_flow_rate == pytest.approx(0.20)
    assert first.port_conditions["feed"].state.temperature == pytest.approx(310.0)

    assert second.transport_and_mixing_update.transport_results[
        "edge_main"
    ].update_fraction == pytest.approx(0.85)
    assert second.port_conditions["feed"].through_molar_flow_rate == pytest.approx(0.85)
    assert second.port_conditions["feed"].state.temperature == pytest.approx(342.5)


def test_macro_step_bookkeeping_marks_transport_and_junction_fallbacks() -> None:
    def hydraulic_solve_step(*, macro_step_time_s: float) -> HydraulicSolveResult:
        del macro_step_time_s
        return _hydraulic_result(edge_flow=0.0)

    def transport_update_step(
        *,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
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
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        del hydraulic_result, transport_and_mixing_update, macro_step_time_s
        return {
            "recycle": PortCondition(
                state=_material_state(
                    pressure=105000.0, temperature=305.0, mole_fractions=(0.4, 0.6)
                ),
                through_molar_flow_rate=0.0,
            )
        }

    result = run_macro_coupling_step(
        macro_step_time_s=5.0,
        hydraulic_solve_step=hydraulic_solve_step,
        transport_update_step=transport_update_step,
        boundary_propagation_step=boundary_propagation_step,
    )

    assert result.bookkeeping.executed_sequence == MACRO_COUPLING_SEQUENCE
    assert result.bookkeeping.transport_fallback_keys == ("edge_zero",)
    assert result.bookkeeping.junction_fallback_keys == ("junction_a",)
    assert result.bookkeeping.used_fallback
