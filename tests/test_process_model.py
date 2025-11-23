import numpy as np
import pytest
from astropy import units as u

from modular_simulation.measurables.process_model import StateType


def test_process_model_metadata_categorization(thermal_process_model):
    metadata = thermal_process_model.state_metadata_dict
    assert set(metadata) == {"temperature", "heat_flux", "heater_power", "ambient_temperature", "cooling_rate"}

    differential = thermal_process_model.categorized_state_metadata_dict(StateType.DIFFERENTIAL)
    assert list(differential) == ["temperature"]

    algebraic = thermal_process_model.categorized_state_metadata_dict(StateType.ALGEBRAIC)
    assert list(algebraic) == ["heat_flux"]

    controlled = thermal_process_model.categorized_state_metadata_dict(StateType.CONTROLLED)
    assert list(controlled) == ["heater_power"]

    constants = thermal_process_model.categorized_state_metadata_dict(StateType.CONSTANT)
    assert set(constants) == {"ambient_temperature", "cooling_rate"}


def test_categorized_state_view_round_trip(thermal_process_model):
    diff_view = thermal_process_model.differential_view
    np.testing.assert_allclose(diff_view.to_array(), np.array([300.0]))

    diff_view.update_from_array(np.array([312.5]))
    assert pytest.approx(thermal_process_model.temperature) == 312.5

    control_view = thermal_process_model.controlled_view
    control_view.update_from_array(np.array([1.2]))
    np.testing.assert_allclose(control_view.to_array(), np.array([1.2]))


def test_unit_conversion_helpers(thermal_process_model):
    getter = thermal_process_model.make_converted_getter("temperature", u.mK)
    assert pytest.approx(getter()) == pytest.approx(300_000.0, rel=1e-6)

    setter = thermal_process_model.make_converted_setter("temperature", u.K)
    setter(350.0)
    assert pytest.approx(thermal_process_model.temperature) == pytest.approx(350.0, rel=1e-6)


def test_process_model_step_updates_state(attached_process_model):
    model = attached_process_model
    model.temperature = 300.0
    model.heater_power = 2.5

    model.step(1.0)

    assert model.temperature > 300.0
    assert pytest.approx(model.heat_flux, rel=1e-3) == 0.2 * (model.temperature - model.ambient_temperature)
