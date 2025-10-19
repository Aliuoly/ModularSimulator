import pytest
from astropy.units import Unit

from modular_simulation.usables.calculations.first_order_filter import FirstOrderFilter
from modular_simulation.usables.sensors.sampled_delayed_sensor import SampledDelayedSensor
from modular_simulation.usables.usable_quantities import UsableQuantities


def _find_tag_info(usable, tag):
    for info in usable.tag_infos:
        if info.tag == tag:
            return info
    raise AssertionError(f"Tag {tag} not found")


def test_first_order_filter_initialization_and_update(thermal_measurables):
    sensor = SampledDelayedSensor(
        measurement_tag="temperature",
        alias_tag="temp_meas",
        sampling_period=0.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
        random_seed=0,
    )
    filter_calc = FirstOrderFilter(
        filtered_signal_tag="temp_filtered",
        raw_signal_tag="temp_meas",
        time_constant=2.0,
    )
    usable = UsableQuantities(
        sensors=[sensor],
        calculations=[filter_calc],
        controllers=[],
        measurable_quantities=thermal_measurables,
    )
    usable._initialize()

    filtered_info = _find_tag_info(usable, "temp_filtered")
    initial_history = list(filtered_info.history)
    assert len(initial_history) == 1
    assert initial_history[0].value == pytest.approx(thermal_measurables.states.temperature)
    assert filtered_info.unit.is_equivalent(Unit("K"))

    thermal_measurables.states.temperature = 350.0
    usable.update(1.0)
    usable.update(2.0)
    updated_value = filtered_info.history[-1].value
    assert updated_value > initial_history[0].value
    assert updated_value < 350.0  # filter should lag behind raw measurement
