from __future__ import annotations

import importlib

import pytest

transport = importlib.import_module("modular_simulation.connection.transport")

AdvancedConservativeTransportModelScaffold = transport.AdvancedConservativeTransportModelScaffold
LagTransportState = transport.LagTransportState
MVPLagTransportModel = transport.MVPLagTransportModel
TransportModelSelection = transport.TransportModelSelection
select_transport_model = transport.select_transport_model
update_lag_transport_state = transport.update_lag_transport_state


def test_selector_picks_mvp_and_advanced_models() -> None:
    mvp_model = select_transport_model(
        TransportModelSelection(model="mvp_lag", parameters={"lag_time_constant_s": 1.2})
    )
    advanced_model = select_transport_model(TransportModelSelection(model="advanced_conservative"))

    assert isinstance(mvp_model, MVPLagTransportModel)
    assert isinstance(advanced_model, AdvancedConservativeTransportModelScaffold)
    assert mvp_model.model == "mvp_lag"
    assert advanced_model.model == "advanced_conservative"


def test_invalid_mode_fails_with_actionable_error() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported transport model 'unknown_mode'. Supported models: mvp_lag, advanced_conservative.",
    ):
        select_transport_model(TransportModelSelection(model="unknown_mode"))


def test_mvp_selected_path_matches_existing_lag_update_behavior() -> None:
    current = LagTransportState(composition=(0.7, 0.3), temperature=310.0)
    advected = LagTransportState(composition=(0.2, 0.8), temperature=345.0)

    selected_model = select_transport_model(
        TransportModelSelection(
            model="mvp_lag",
            parameters={
                "lag_time_constant_s": 0.9,
                "near_zero_flow_epsilon": 1.0e-12,
                "flow_smoothing_flow_rate": 1.0e-9,
            },
        )
    )
    selected_result = selected_model.update(
        current_state=current,
        advected_state=advected,
        dt=0.1,
        through_molar_flow_rate=0.8,
        previous_through_molar_flow_rate=0.7,
    )
    direct_result = update_lag_transport_state(
        current_state=current,
        advected_state=advected,
        dt=0.1,
        lag_time_constant_s=0.9,
        through_molar_flow_rate=0.8,
        previous_through_molar_flow_rate=0.7,
        near_zero_flow_epsilon=1.0e-12,
        flow_smoothing_flow_rate=1.0e-9,
    )

    assert selected_result == direct_result


def test_advanced_scaffold_update_has_clear_not_implemented_path() -> None:
    model = select_transport_model(TransportModelSelection(model="advanced_conservative"))

    with pytest.raises(
        NotImplementedError,
        match="Advanced conservative transport is scaffolded only and is not implemented in T3.5.",
    ):
        model.update(
            current_state=LagTransportState(composition=(0.5, 0.5), temperature=300.0),
            advected_state=LagTransportState(composition=(0.6, 0.4), temperature=320.0),
            dt=0.1,
            through_molar_flow_rate=1.0,
            previous_through_molar_flow_rate=-1.0,
        )
