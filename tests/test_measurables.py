import numpy as np
import pytest
from astropy.units import Unit
from typing import Annotated

from modular_simulation.measurables import (
    AlgebraicStates,
    Constants,
    ControlElements,
    MeasurableQuantities,
    States,
)
from modular_simulation.validation.exceptions import MeasurableConfigurationError


class SimpleStates(States):
    level: Annotated[float, Unit("m")] = 5.0
    temperature: Annotated[float, Unit("K")] = 350.0


class SimpleAlgebraic(AlgebraicStates):
    density: Annotated[float, Unit("kg / m3")] = 1000.0


class SimpleControl(ControlElements):
    valve: Annotated[float, Unit(1)] = 0.5


class SimpleConstants(Constants):
    area: Annotated[float, Unit("m2")] = 2.0


def test_base_indexed_model_to_array_and_update():
    states = SimpleStates()
    array = states.to_array()

    assert isinstance(array, np.ndarray)
    assert array.tolist() == [pytest.approx(5.0), pytest.approx(350.0)]

    new_array = np.array([10.0, 360.0])
    states.update_from_array(new_array)
    assert states.level == pytest.approx(10.0)
    assert states.temperature == pytest.approx(360.0)


def test_measurable_quantities_tag_list_and_units():
    mq = MeasurableQuantities(
        states=SimpleStates(),
        algebraic_states=SimpleAlgebraic(),
        control_elements=SimpleControl(),
        constants=SimpleConstants(),
    )

    assert set(mq.tag_list) == {"level", "temperature", "density", "valve", "area"}
    assert mq.tag_unit_info["temperature"].is_equivalent(Unit("K"))
    assert mq.tag_unit_info["valve"].is_equivalent(Unit(1))


def test_measurable_quantities_detects_duplicates():
    class DuplicateStates(States):
        value: Annotated[float, Unit("K")] = 1.0

    class DuplicateControl(ControlElements):
        value: Annotated[float, Unit(1)] = 0.0

    with pytest.raises(MeasurableConfigurationError):
        MeasurableQuantities(
            states=DuplicateStates(),
            control_elements=DuplicateControl(),
        )


def test_measurable_quantities_requires_any_fields():
    with pytest.raises(MeasurableConfigurationError):
        MeasurableQuantities()
