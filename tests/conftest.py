import pytest
from astropy.units import Unit, Quantity
from typing import Annotated

from modular_simulation.measurables import (
    MeasurableQuantities,
    States,
    ControlElements,
    AlgebraicStates,
    Constants,
)


UNIT_TEMPERATURE = Unit("K")
UNIT_POWER = Unit("W")
UNIT_TIME = Unit("s")
UNIT_DIMENSIONLESS = Unit(1)


class ThermalStates(States):
    temperature: Annotated[float, UNIT_TEMPERATURE] = 300.0


class ThermalAlgebraic(AlgebraicStates):
    heat_flux: Annotated[float, UNIT_POWER] = 0.0


class ThermalControlElements(ControlElements):
    heater_power: Annotated[float, UNIT_POWER] = 0.0


class ThermalConstants(Constants):
    ambient_temperature: Annotated[float, UNIT_TEMPERATURE] = 295.0
    time_constant: Annotated[float, UNIT_TIME] = 5.0


@pytest.fixture()
def thermal_measurables() -> MeasurableQuantities:
    return MeasurableQuantities(
        states=ThermalStates(),
        algebraic_states=ThermalAlgebraic(),
        control_elements=ThermalControlElements(),
        constants=ThermalConstants(),
    )


@pytest.fixture()
def heater_mv_range():
    return (
       0, 100.0
    )
