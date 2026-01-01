import numpy as np
import pytest
from types import SimpleNamespace
from typing import Annotated

from modular_simulation.measurables.process_model import (
    ProcessModel,
    StateMetadata as M,
    StateType as T,
)
from modular_simulation.utils.typing import ArrayIndex


class ThermalProcessModel(ProcessModel):
    """Simple first-order process used for testing."""

    temperature: Annotated[
        float, M(type=T.DIFFERENTIAL, unit="K", description="process temperature")
    ] = 300.0
    heat_flux: Annotated[float, M(type=T.ALGEBRAIC, unit="K/s", description="cooling rate")] = 0.0
    heater_power: Annotated[
        float, M(type=T.CONTROLLED, unit="K/s", description="heating input")
    ] = 0.0
    ambient_temperature: Annotated[
        float, M(type=T.CONSTANT, unit="K", description="ambient temperature")
    ] = 295.0
    cooling_rate: Annotated[
        float, M(type=T.CONSTANT, unit="1/s", description="linear cooling coefficient")
    ] = 0.2

    @staticmethod
    def calculate_algebraic_values(
        y: np.ndarray,
        u: np.ndarray,
        k: np.ndarray,
        y_map: dict[str, ArrayIndex],
        u_map: dict[str, ArrayIndex],
        k_map: dict[str, ArrayIndex],
        algebraic_map: dict[str, ArrayIndex],
        algebraic_size: int,
    ) -> np.ndarray:
        result = np.zeros(algebraic_size)
        temperature = y[y_map["temperature"]]
        ambient = k[k_map["ambient_temperature"]]
        rate = k[k_map["cooling_rate"]]
        result[algebraic_map["heat_flux"]] = rate * (temperature - ambient)
        return result

    @staticmethod
    def differential_rhs(
        t: float,
        y: np.ndarray,
        u: np.ndarray,
        k: np.ndarray,
        algebraic: np.ndarray,
        y_map: dict[str, ArrayIndex],
        u_map: dict[str, ArrayIndex],
        k_map: dict[str, ArrayIndex],
        algebraic_map: dict[str, ArrayIndex],
    ) -> np.ndarray:
        dy = np.zeros_like(y)
        heater = u[u_map["heater_power"]]
        cooling = algebraic[algebraic_map["heat_flux"]]
        dy[y_map["temperature"]] = heater - cooling
        return dy

    def model_dump(self, *args, **kwargs):  # type: ignore[override]
        exclude = kwargs.pop("exclude", set())
        if isinstance(exclude, set):
            exclude = set(exclude)
            exclude.add("t")
        elif isinstance(exclude, dict):
            exclude = {**exclude, "t": True}
        else:
            exclude = {"t"}
        return super().model_dump(*args, exclude=exclude, **kwargs)


@pytest.fixture()
def thermal_process_model() -> ThermalProcessModel:
    return ThermalProcessModel()


@pytest.fixture()
def attached_process_model(thermal_process_model: ThermalProcessModel) -> ThermalProcessModel:
    dummy_system = SimpleNamespace(
        solver_options={"method": "RK45", "rtol": 1e-8, "atol": 1e-10},
        use_numba=False,
        numba_options={},
    )
    thermal_process_model.attach_system(dummy_system)
    return thermal_process_model
