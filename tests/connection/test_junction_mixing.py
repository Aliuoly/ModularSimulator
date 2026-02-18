from __future__ import annotations

import importlib

import pytest

junction = importlib.import_module("modular_simulation.connection.junction")
state = importlib.import_module("modular_simulation.connection.state")

MaterialState = state.MaterialState
PortCondition = state.PortCondition
mix_junction_state = junction.mix_junction_state


def test_weighted_mixing_uses_incoming_flows_only() -> None:
    previous = MaterialState(
        pressure=150000.0,
        temperature=310.0,
        mole_fractions=(0.4, 0.6),
    )
    incoming_a = PortCondition(
        state=MaterialState(
            pressure=100000.0,
            temperature=300.0,
            mole_fractions=(0.8, 0.2),
        ),
        through_molar_flow_rate=2.0,
    )
    incoming_b = PortCondition(
        state=MaterialState(
            pressure=200000.0,
            temperature=350.0,
            mole_fractions=(0.2, 0.8),
        ),
        through_molar_flow_rate=1.0,
    )
    outgoing = PortCondition(
        state=MaterialState(
            pressure=900000.0,
            temperature=1000.0,
            mole_fractions=(1.0, 0.0),
        ),
        through_molar_flow_rate=-5.0,
    )

    result = mix_junction_state(
        incoming_port_conditions={
            "edge_out": outgoing,
            "edge_in_a": incoming_a,
            "edge_in_b": incoming_b,
        },
        previous_state=previous,
    )

    assert not result.used_fallback
    assert result.total_incoming_flow_rate == pytest.approx(3.0)
    assert result.state.pressure == pytest.approx((2.0 * 100000.0 + 1.0 * 200000.0) / 3.0)
    assert result.state.temperature == pytest.approx((2.0 * 300.0 + 1.0 * 350.0) / 3.0)
    assert result.state.mole_fractions[0] == pytest.approx((2.0 * 0.8 + 1.0 * 0.2) / 3.0)
    assert result.state.mole_fractions[1] == pytest.approx((2.0 * 0.2 + 1.0 * 0.8) / 3.0)


def test_no_incoming_flow_falls_back_to_previous_state() -> None:
    previous = MaterialState(
        pressure=101325.0,
        temperature=298.15,
        mole_fractions=(0.3, 0.7),
    )

    result = mix_junction_state(
        incoming_port_conditions={},
        previous_state=previous,
    )

    assert result.used_fallback
    assert result.total_incoming_flow_rate == pytest.approx(0.0)
    assert result.state == previous


def test_near_zero_total_incoming_flow_is_deterministic_fallback() -> None:
    previous = MaterialState(
        pressure=120000.0,
        temperature=315.0,
        mole_fractions=(0.55, 0.45),
    )
    tiny_a = PortCondition(
        state=MaterialState(
            pressure=90000.0,
            temperature=270.0,
            mole_fractions=(0.1, 0.9),
        ),
        through_molar_flow_rate=3.0e-13,
    )
    tiny_b = PortCondition(
        state=MaterialState(
            pressure=300000.0,
            temperature=450.0,
            mole_fractions=(0.9, 0.1),
        ),
        through_molar_flow_rate=7.0e-13,
    )

    result = mix_junction_state(
        incoming_port_conditions={"a": tiny_a, "b": tiny_b},
        previous_state=previous,
        near_zero_incoming_flow_epsilon=1.0e-12,
    )

    assert result.used_fallback
    assert result.total_incoming_flow_rate == pytest.approx(1.0e-12)
    assert result.state == previous
