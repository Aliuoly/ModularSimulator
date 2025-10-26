from typing import Annotated

import numpy as np
import pytest
from astropy.units import Unit

from modular_simulation.core import DynamicModel, MeasurableMetadata, MeasurableType


UNIT_TEMPERATURE = Unit("K")
UNIT_POWER = Unit("W")
UNIT_TIME = Unit("s")
UNIT_DIMENSIONLESS = Unit(1)


class ThermalModel(DynamicModel):
    temperature: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, UNIT_TEMPERATURE),
    ] = 300.0
    heat_flux: Annotated[
        float,
        MeasurableMetadata(MeasurableType.ALGEBRAIC_STATE, UNIT_POWER),
    ] = 0.0
    heater_power: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONTROL_ELEMENT, UNIT_POWER),
    ] = 0.0
    ambient_temperature: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, UNIT_TEMPERATURE),
    ] = 295.0
    time_constant: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, UNIT_TIME),
    ] = 5.0

    @staticmethod
    def calculate_algebraic_values(
        y,
        u,
        k,
        y_map,
        u_map,
        k_map,
        algebraic_map,
        algebraic_size,
    ):
        algebraic = np.zeros(algebraic_size)
        temp = float(y[y_map["temperature"]][0])
        heater = float(u[u_map["heater_power"]][0]) if u_map else 0.0
        ambient = float(k[k_map["ambient_temperature"]][0])
        heat_flux = heater - (temp - ambient)
        algebraic[algebraic_map["heat_flux"]] = heat_flux
        return algebraic

    @staticmethod
    def rhs(
        t,
        y,
        u,
        k,
        algebraic,
        u_map,
        y_map,
        k_map,
        algebraic_map,
    ):
        dy = np.zeros_like(y)
        tau = float(k[k_map["time_constant"]][0])
        heat_flux = float(algebraic[algebraic_map["heat_flux"]][0])
        dy[y_map["temperature"]] = heat_flux / tau
        return dy


@pytest.fixture()
def thermal_model() -> ThermalModel:
    return ThermalModel()


@pytest.fixture()
def heater_mv_range():
    return (0, 100.0)
