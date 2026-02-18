from __future__ import annotations

import importlib

import numpy as np
import pytest

connection_layer = importlib.import_module("modular_simulation.connection.connection_layer")
hydraulic_solver = importlib.import_module("modular_simulation.connection.hydraulic_solver")
state = importlib.import_module("modular_simulation.connection.state")
transport = importlib.import_module("modular_simulation.connection.transport")

HydraulicSolveResult = hydraulic_solver.HydraulicSolveResult
LagTransportState = transport.LagTransportState
LagTransportUpdateResult = transport.LagTransportUpdateResult
MaterialState = state.MaterialState
PicardIterationGateConfig = connection_layer.PicardIterationGateConfig
PortCondition = state.PortCondition
TransportAndMixingUpdate = connection_layer.TransportAndMixingUpdate
run_macro_coupling_step = connection_layer.run_macro_coupling_step


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

    def hydraulic_solve_step(*, macro_step_time_s: float) -> HydraulicSolveResult:
        del macro_step_time_s
        call_counts["hydraulics"] += 1
        return _hydraulic_result(edge_flow=0.25, residual_norm=1.0e-9)

    def transport_update_step(
        *,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        del hydraulic_result, macro_step_time_s
        call_counts["transport"] += 1
        return _transport_update()

    def boundary_propagation_step(
        *,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        del hydraulic_result, transport_and_mixing_update, macro_step_time_s
        call_counts["boundary"] += 1
        return {"inlet": _port_condition(through_molar_flow_rate=0.25)}

    result = run_macro_coupling_step(
        macro_step_time_s=1.0,
        hydraulic_solve_step=hydraulic_solve_step,
        transport_update_step=transport_update_step,
        boundary_propagation_step=boundary_propagation_step,
        picard_gate_config=PicardIterationGateConfig(
            enabled=True,
            residual_threshold=1.0e-6,
            max_iterations=4,
            tolerance=1.0e-12,
        ),
    )

    assert call_counts == {"hydraulics": 1, "transport": 1, "boundary": 1}
    assert not result.bookkeeping.picard_gate_enabled
    assert result.bookkeeping.picard_iterations_used == 0
    assert result.bookkeeping.picard_converged
    assert result.bookkeeping.picard_last_residual is None


def test_gate_on_for_strong_coupling_iterates_until_tolerance() -> None:
    call_count = 0
    boundary_flows = iter((1.0, 0.40, 0.10, 0.1000000001))

    def hydraulic_solve_step(*, macro_step_time_s: float) -> HydraulicSolveResult:
        nonlocal call_count
        del macro_step_time_s
        call_count += 1
        return _hydraulic_result(edge_flow=1.0, residual_norm=5.0)

    def transport_update_step(
        *,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        del hydraulic_result, macro_step_time_s
        return _transport_update()

    def boundary_propagation_step(
        *,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        del hydraulic_result, transport_and_mixing_update, macro_step_time_s
        return {"inlet": _port_condition(through_molar_flow_rate=next(boundary_flows))}

    result = run_macro_coupling_step(
        macro_step_time_s=2.0,
        hydraulic_solve_step=hydraulic_solve_step,
        transport_update_step=transport_update_step,
        boundary_propagation_step=boundary_propagation_step,
        picard_gate_config=PicardIterationGateConfig(
            enabled=True,
            residual_threshold=0.1,
            max_iterations=5,
            tolerance=1.0e-6,
        ),
    )

    assert call_count == 4
    assert result.bookkeeping.picard_gate_enabled
    assert result.bookkeeping.picard_iterations_used == 3
    assert result.bookkeeping.picard_converged
    assert result.bookkeeping.picard_last_residual == pytest.approx(1.0e-10)


def test_gate_stops_at_max_iterations_when_tolerance_not_reached() -> None:
    call_count = 0
    boundary_flows = iter((1.0, 0.80, 0.70, 0.60))

    def hydraulic_solve_step(*, macro_step_time_s: float) -> HydraulicSolveResult:
        nonlocal call_count
        del macro_step_time_s
        call_count += 1
        return _hydraulic_result(edge_flow=1.0, residual_norm=10.0)

    def transport_update_step(
        *,
        hydraulic_result: HydraulicSolveResult,
        macro_step_time_s: float,
    ) -> TransportAndMixingUpdate:
        del hydraulic_result, macro_step_time_s
        return _transport_update()

    def boundary_propagation_step(
        *,
        hydraulic_result: HydraulicSolveResult,
        transport_and_mixing_update: TransportAndMixingUpdate,
        macro_step_time_s: float,
    ) -> dict[str, PortCondition]:
        del hydraulic_result, transport_and_mixing_update, macro_step_time_s
        return {"inlet": _port_condition(through_molar_flow_rate=next(boundary_flows))}

    result = run_macro_coupling_step(
        macro_step_time_s=3.0,
        hydraulic_solve_step=hydraulic_solve_step,
        transport_update_step=transport_update_step,
        boundary_propagation_step=boundary_propagation_step,
        picard_gate_config=PicardIterationGateConfig(
            enabled=True,
            residual_threshold=1.0,
            max_iterations=3,
            tolerance=1.0e-12,
        ),
    )

    assert call_count == 4
    assert result.bookkeeping.picard_gate_enabled
    assert result.bookkeeping.picard_iterations_used == 3
    assert not result.bookkeeping.picard_converged
    assert result.bookkeeping.picard_last_residual == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "enabled": "yes",
                "residual_threshold": 0.0,
                "max_iterations": 1,
                "tolerance": 1.0e-8,
            },
            "enabled must be a bool",
        ),
        (
            {
                "enabled": True,
                "residual_threshold": -1.0,
                "max_iterations": 1,
                "tolerance": 1.0e-8,
            },
            "residual_threshold must be finite and non-negative",
        ),
        (
            {
                "enabled": True,
                "residual_threshold": 0.0,
                "max_iterations": 0,
                "tolerance": 1.0e-8,
            },
            "max_iterations must be positive",
        ),
        (
            {
                "enabled": True,
                "residual_threshold": 0.0,
                "max_iterations": 1,
                "tolerance": 0.0,
            },
            "tolerance must be finite and positive",
        ),
    ],
)
def test_invalid_picard_gate_config_raises_value_error(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PicardIterationGateConfig(**kwargs)
