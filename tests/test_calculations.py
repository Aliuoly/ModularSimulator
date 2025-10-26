import pytest
from astropy.units import Unit

from modular_simulation.interfaces import ModelInterface, SampledDelayedSensor
from modular_simulation.interfaces.calculations.first_order_filter import FirstOrderFilter


def _find_tag_info(interface, tag):
    for info in interface.tag_infos:
        if info.tag == tag:
            return info
    raise AssertionError(f"Tag {tag} not found")


def test_first_order_filter_initialization_and_update(thermal_model):
    sensor = SampledDelayedSensor(
        measurement_tag="temperature",
        alias_tag="temp_meas",
        sampling_period=0.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
        random_seed=0,
        unit="K",
    )
    filter_calc = FirstOrderFilter(
        filtered_signal_tag="temp_filtered",
        raw_signal_tag="temp_meas",
        time_constant=2.0,
    )
    interface = ModelInterface(
        sensors=[sensor],
        calculations=[filter_calc],
    )
    interface._initialize(thermal_model)

    filtered_info = _find_tag_info(interface, "temp_filtered")
    initial_history = list(filtered_info.history)
    assert len(initial_history) == 1
    assert initial_history[0].value == pytest.approx(thermal_model.temperature)
    assert filtered_info.unit.is_equivalent(Unit("K"))

    thermal_model.temperature = 350.0
    interface.update(1.0)
    interface.update(2.0)
    updated_value = filtered_info.history[-1].value
    assert updated_value > initial_history[0].value
    assert updated_value < 350.0
