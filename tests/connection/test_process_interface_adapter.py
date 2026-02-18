from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from modular_simulation.connection.state import MaterialState, PortCondition

process_interface = importlib.import_module("modular_simulation.connection.process_interface")
ProcessInterface = process_interface.ProcessInterface
ProcessModelAdapter = process_interface.ProcessModelAdapter


def _attach_default_solver_options(process_model: Any) -> None:
    dummy_system = SimpleNamespace(
        solver_options={"method": "RK45", "rtol": 1e-8, "atol": 1e-10},
        use_numba=False,
        numba_options={},
    )
    process_model.attach_system(dummy_system)


def _port_condition(*, pressure: float, temperature: float, flow: float) -> PortCondition:
    return PortCondition(
        state=MaterialState(
            pressure=pressure,
            temperature=temperature,
            mole_fractions=(0.5, 0.5),
        ),
        through_molar_flow_rate=flow,
        macro_step_time_s=1.0,
    )


def test_process_model_adapter_wraps_existing_fixture(thermal_process_model) -> None:
    adapter = ProcessModelAdapter(thermal_process_model)

    assert isinstance(adapter, ProcessInterface)
    assert set(adapter.state_names) == {
        "temperature",
        "heat_flux",
        "heater_power",
        "ambient_temperature",
        "cooling_rate",
    }
    assert adapter.input_port_names == ()
    assert adapter.output_port_names == ()

    assert adapter.get_state("temperature") == pytest.approx(300.0)
    adapter.set_state("temperature", 311.0)
    assert thermal_process_model.temperature == pytest.approx(311.0)

    getter = adapter.make_state_getter("temperature", "mK")
    setter = adapter.make_state_setter("temperature", "K")
    setter(315.0)
    assert getter() == pytest.approx(315_000.0)


def test_adapter_exposes_port_condition_read_write_hooks(thermal_process_model) -> None:
    adapter = ProcessModelAdapter(thermal_process_model)

    expected = {
        "inlet": _port_condition(pressure=120000.0, temperature=320.0, flow=1.25),
        "outlet": _port_condition(pressure=118000.0, temperature=318.0, flow=-1.25),
    }
    adapter.write_port_conditions(expected)
    observed = adapter.read_port_conditions()

    assert observed == expected

    observed_mutable = dict(observed)
    observed_mutable["inlet"] = _port_condition(pressure=100000.0, temperature=300.0, flow=0.0)
    assert adapter.read_port_conditions() == expected
    assert adapter.get_port_condition("inlet") == expected["inlet"]


def test_adapter_raises_actionable_errors_for_invalid_state_and_port(thermal_process_model) -> None:
    adapter = ProcessModelAdapter(thermal_process_model)

    with pytest.raises(KeyError, match="Unknown state 'invalid_state'"):
        adapter.get_state("invalid_state")

    with pytest.raises(KeyError, match="Unknown state 'invalid_state'"):
        adapter.set_state("invalid_state", 1.0)

    with pytest.raises(KeyError, match="Unknown port condition 'missing_port'"):
        adapter.get_port_condition("missing_port")


def test_adapter_is_behaviorally_neutral_without_connection_interaction(
    thermal_process_model,
) -> None:
    process_model_cls = thermal_process_model.__class__
    baseline_model = process_model_cls()
    adapted_model = process_model_cls()
    _attach_default_solver_options(baseline_model)
    _attach_default_solver_options(adapted_model)

    baseline_model.temperature = 300.0
    baseline_model.heater_power = 2.0
    adapted_model.temperature = 300.0
    adapted_model.heater_power = 2.0

    adapter = ProcessModelAdapter(adapted_model)
    _ = adapter.state_names

    baseline_model.step(1.0)
    adapted_model.step(1.0)

    assert adapted_model.temperature == pytest.approx(baseline_model.temperature)
    assert adapted_model.heat_flux == pytest.approx(baseline_model.heat_flux)
