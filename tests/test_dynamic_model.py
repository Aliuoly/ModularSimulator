import numpy as np
import pytest
from astropy.units import Unit
from typing import Annotated

from modular_simulation.core import DynamicModel, MeasurableMetadata, MeasurableType
from modular_simulation.validation.exceptions import MeasurableConfigurationError


class ExampleModel(DynamicModel):
    level: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("m")),
    ] = 5.0
    temperature: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("K")),
    ] = 350.0
    density: Annotated[
        float,
        MeasurableMetadata(MeasurableType.ALGEBRAIC_STATE, Unit("kg / m3")),
    ] = 1000.0
    valve: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONTROL_ELEMENT, Unit(1)),
    ] = 0.5
    area: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("m2")),
    ] = 2.0

    @staticmethod
    def calculate_algebraic_values(*args, **kwargs):
        return np.array([1000.0])

    @staticmethod
    def rhs(*args, **kwargs):
        return np.zeros(2)


def test_category_arrays_round_trip():
    model = ExampleModel()
    array = model.states.to_array()

    assert isinstance(array, np.ndarray)
    assert array.tolist() == [pytest.approx(5.0), pytest.approx(350.0)]

    new_array = np.array([10.0, 360.0])
    model.states.update_from_array(new_array)
    assert model.level == pytest.approx(10.0)
    assert model.temperature == pytest.approx(360.0)


def test_dynamic_model_tag_metadata():
    model = ExampleModel()
    assert set(model.tag_list) == {"level", "temperature", "density", "valve", "area"}
    assert model.tag_unit_info["temperature"].is_equivalent(Unit("K"))
    assert model.tag_unit_info["valve"].is_equivalent(Unit(1))


def test_dynamic_model_requires_fields():
    with pytest.raises(MeasurableConfigurationError):
        class EmptyModel(DynamicModel):
            @staticmethod
            def calculate_algebraic_values(*args, **kwargs):
                return np.zeros(0)

            @staticmethod
            def rhs(*args, **kwargs):
                return np.zeros(0)

        EmptyModel()
