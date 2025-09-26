import pytest
from modular_simulation.quantities.utils import ConfigurationError
from modular_simulation.quantities.measurable_quantities import MeasurableQuantities
from modular_simulation.quantities.usable_quantities import UsableQuantities
from dummy_test_definitions import DummyControlElements, DummySensor, DummyStates, AddAllCalculation

def _make_proper_sensors():
    proper_sensors = [
        DummySensor(measurement_tag = "mv1"),
        DummySensor(measurement_tag = "cv1"),
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
        output_tag = "proper_output",
        measured_input_tags = ["mv1","mv2",'cv1','cv2']
    )]

def _make_improper_calculations():
    return [AddAllCalculation(
        output_tag = "improper_output",
        measured_input_tags = ["mv4","cv4"]
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

    with pytest.raises(ConfigurationError):
        usable = _make_usable(proper_sensors, improper_calculations)
    
    with pytest.raises(ConfigurationError):
        usable = _make_usable(improper_sensors, proper_calculations)

    with pytest.raises(ConfigurationError):
        usable = _make_usable(improper_sensors, improper_calculations)

    # this one should raise exception
    usable = _make_usable(proper_sensors, proper_calculations)

if __name__ == '__main__':
    test_usable_validation()