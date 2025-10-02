import pytest
from modular_simulation.validation.exceptions import SensorConfigurationError, CalculationConfigurationError
from modular_simulation.quantities.measurable_quantities import MeasurableQuantities
from modular_simulation.quantities.usable_quantities import UsableQuantities
from dummy_test_definitions import DummyControlElements, DummySensor, DummyStates, AddAllCalculation

def _make_proper_sensors():
    proper_sensors = [
        DummySensor(measurement_tag = "mv1"),
        DummySensor(measurement_tag = "cv1"),
        DummySensor(measurement_tag = "mv2"),
        DummySensor(measurement_tag = "cv2"),
    ]
    return proper_sensors

def _make_improper_sensors():
    improper_sensors = [
        DummySensor(measurement_tag = "mv4"),
        DummySensor(measurement_tag = "cv4"),
    ]
    return improper_sensors

def _make_proper_calculations():
    return [AddAllCalculation(
        output_tag="proper_output",
        mv1_tag="mv1",
        mv2_tag="mv2",
        cv1_tag="cv1",
        cv2_tag="cv2",
    )]

def _make_improper_calculations():
    return [AddAllCalculation(
        output_tag="improper_output",
        mv1_tag="mv5",
        mv2_tag="mv2",
        cv1_tag="cv5",
        cv2_tag="cv2",
    )]



def test_usable_validation():

    def _make_usable(sensors, calculations):
        measurable_quantities = MeasurableQuantities(
            states = DummyStates(),
            control_elements = DummyControlElements(),
        )
        return UsableQuantities(
            sensors = sensors,
            calculations = calculations, 
            measurable_quantities = measurable_quantities
        )

    proper_sensors = _make_proper_sensors()
    improper_sensors = _make_improper_sensors()
    proper_calculations = _make_proper_calculations()
    improper_calculations = _make_improper_calculations()

    with pytest.raises(ExceptionGroup) as ex_info:
        usable = _make_usable(proper_sensors, improper_calculations)
    calc_errors = [ex for ex in ex_info.value.exceptions if isinstance(ex, CalculationConfigurationError)]
    other_errors = [ex for ex in ex_info.value.exceptions if ex not in calc_errors]
    assert len(calc_errors) > 0
    assert len(other_errors) == 0

    with pytest.raises(ExceptionGroup) as ex_info:
        usable = _make_usable(improper_sensors, proper_calculations)
    meas_errors = [ex for ex in ex_info.value.exceptions if isinstance(ex, SensorConfigurationError)]
    assert len(meas_errors) > 0 #here, due to improper measurement, even a proper_calculation might fail, so we don't assert no calc errors.
    
    with pytest.raises(ExceptionGroup) as ex_info:
        usable = _make_usable(improper_sensors, improper_calculations)
    calc_errors = [ex for ex in ex_info.value.exceptions if isinstance(ex, CalculationConfigurationError)]
    meas_errors = [ex for ex in ex_info.value.exceptions if isinstance(ex, SensorConfigurationError)]
    other_errors = [ex for ex in ex_info.value.exceptions if ex not in (calc_errors + meas_errors)]
    assert len(meas_errors) > 0
    assert len(calc_errors) > 0
    assert len(other_errors) == 0

    usable = _make_usable(proper_sensors, proper_calculations)

if __name__ == '__main__':
    test_usable_validation()