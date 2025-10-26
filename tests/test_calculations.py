import pytest
from astropy.units import Unit

from modular_simulation.interfaces import SampledDelayedSensor
from modular_simulation.interfaces.calculations.first_order_filter import FirstOrderFilter


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
    sensor._initialize(thermal_model)
    filter_calc._initialize([sensor._tag_info])

    filtered_info = filter_calc._output_tag_info_dict["temp_filtered"]
    initial_history = list(filtered_info.history)
    assert len(initial_history) == 1
    assert initial_history[0].value == pytest.approx(thermal_model.temperature)
    assert filtered_info.unit.is_equivalent(Unit("K"))

    thermal_model.temperature = 350.0
    sensor.measure(1.0)
    filter_calc.calculate(1.0)
    sensor.measure(2.0)
    filter_calc.calculate(2.0)
    updated_value = filtered_info.history[-1].value
    assert updated_value > initial_history[0].value
    assert updated_value < 350.0
