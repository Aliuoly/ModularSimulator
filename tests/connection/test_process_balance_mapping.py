from __future__ import annotations

import importlib

import pytest

from modular_simulation.connection.state import MaterialState, PortCondition

process_interface = importlib.import_module("modular_simulation.connection.process_interface")
ProcessModelAdapter = process_interface.ProcessModelAdapter


def _material_state(
    *,
    pressure: float,
    temperature: float,
    mole_fractions: tuple[float, ...] = (0.3, 0.7),
) -> MaterialState:
    return MaterialState(
        pressure=pressure,
        temperature=temperature,
        mole_fractions=mole_fractions,
    )


def _port_condition(*, pressure: float, temperature: float, flow: float) -> PortCondition:
    return PortCondition(
        state=_material_state(pressure=pressure, temperature=temperature),
        through_molar_flow_rate=flow,
        macro_step_time_s=2.0,
    )


def _adapter_with_declared_ports(thermal_process_model) -> ProcessModelAdapter:
    thermal_process_model._input_streams = {"feed": object()}  # pyright: ignore[reportPrivateUsage]
    thermal_process_model._output_streams = {"product": object()}  # pyright: ignore[reportPrivateUsage]
    return ProcessModelAdapter(thermal_process_model)


def test_balance_mapping_exposes_inlet_and_outlet_terms_each_macro_step(
    thermal_process_model,
) -> None:
    adapter = _adapter_with_declared_ports(thermal_process_model)
    adapter.write_port_conditions(
        {
            "feed": _port_condition(pressure=130000.0, temperature=315.0, flow=1.2),
            "product": _port_condition(pressure=125000.0, temperature=320.0, flow=0.9),
        }
    )

    mapped = adapter.map_port_conditions_to_balance_terms(
        incoming_material_states={
            "feed": _material_state(
                pressure=131000.0,
                temperature=312.0,
                mole_fractions=(0.1, 0.9),
            ),
            "product": _material_state(
                pressure=124000.0,
                temperature=322.0,
                mole_fractions=(0.8, 0.2),
            ),
        }
    )

    assert tuple(mapped.keys()) == ("feed", "product")

    feed = mapped["feed"]
    assert feed.port_role == "inlet"
    assert feed.inlet_molar_flow_rate == pytest.approx(1.2)
    assert feed.outlet_molar_flow_rate == pytest.approx(0.0)
    assert feed.through_molar_flow_rate == pytest.approx(1.2)

    product = mapped["product"]
    assert product.port_role == "outlet"
    assert product.inlet_molar_flow_rate == pytest.approx(0.0)
    assert product.outlet_molar_flow_rate == pytest.approx(0.9)
    assert product.through_molar_flow_rate == pytest.approx(0.9)


def test_balance_mapping_is_deterministic_and_stable_under_flow_reversal(
    thermal_process_model,
) -> None:
    adapter = _adapter_with_declared_ports(thermal_process_model)
    adapter.write_port_conditions(
        {
            "feed": _port_condition(pressure=128000.0, temperature=316.0, flow=-0.4),
            "product": _port_condition(pressure=123000.0, temperature=319.0, flow=-0.7),
        }
    )

    first = adapter.map_port_conditions_to_balance_terms()
    second = adapter.map_port_conditions_to_balance_terms()

    assert first == second
    assert first["feed"].port_role == "inlet"
    assert first["feed"].inlet_molar_flow_rate == pytest.approx(0.0)
    assert first["feed"].outlet_molar_flow_rate == pytest.approx(0.4)

    assert first["product"].port_role == "outlet"
    assert first["product"].inlet_molar_flow_rate == pytest.approx(0.7)
    assert first["product"].outlet_molar_flow_rate == pytest.approx(0.0)


def test_balance_mapping_includes_required_incoming_material_state_fields(
    thermal_process_model,
) -> None:
    adapter = _adapter_with_declared_ports(thermal_process_model)
    adapter.write_port_conditions(
        {
            "feed": _port_condition(pressure=120000.0, temperature=305.0, flow=0.6),
            "product": _port_condition(pressure=119000.0, temperature=307.0, flow=0.6),
        }
    )

    incoming_feed_state = _material_state(
        pressure=121000.0,
        temperature=301.0,
        mole_fractions=(0.25, 0.75),
    )
    mapped = adapter.map_port_conditions_to_balance_terms(
        incoming_material_states={"feed": incoming_feed_state}
    )

    feed = mapped["feed"]
    assert feed.incoming_pressure == pytest.approx(121000.0)
    assert feed.incoming_temperature == pytest.approx(301.0)
    assert feed.incoming_mole_fractions == pytest.approx((0.25, 0.75))

    product = mapped["product"]
    assert product.incoming_pressure == pytest.approx(product.solved_pressure)
    assert product.incoming_temperature == pytest.approx(product.solved_temperature)
    assert product.incoming_mole_fractions == pytest.approx(product.solved_mole_fractions)
