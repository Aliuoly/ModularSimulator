import numpy as np
import pytest
from astropy.units import Quantity, Unit

from modular_simulation.framework.utils import create_system
from modular_simulation.framework.system_old import System
from modular_simulation.usables.controllers.controller_base import ControllerBase, ControllerMode
from modular_simulation.usables.controllers.trajectory import Trajectory
from modular_simulation.usables.sensors.sampled_delayed_sensor import SampledDelayedSensor

from conftest import (
    ThermalAlgebraic,
    ThermalConstants,
    ThermalControlElements,
    ThermalStates,
    UNIT_TEMPERATURE,
)


class HeaterSystem(System):
    @staticmethod
    def calculate_algebraic_values(y, u, k, y_map, u_map, k_map, algebraic_map, algebraic_size):
        algebraic = np.zeros(algebraic_size)
        temp = float(y[y_map["temperature"]][0])
        heater = float(u[u_map["heater_power"]][0]) if u_map else 0.0
        ambient = float(k[k_map["ambient_temperature"]][0])
        heat_flux = heater - (temp - ambient)
        algebraic[algebraic_map["heat_flux"]] = heat_flux
        return algebraic

    @staticmethod
    def rhs(t, y, u, k, algebraic, u_map, y_map, k_map, algebraic_map):
        dy = np.zeros_like(y)
        tau = float(k[k_map["time_constant"]][0])
        heat_flux = float(algebraic[algebraic_map["heat_flux"]][0])
        dy[y_map["temperature"]] = heat_flux / tau
        return dy


class SimpleController(ControllerBase):
    def _control_algorithm(self, t, cv, sp):  # type: ignore[override]
        return self._u0 + (sp - cv)


@pytest.fixture()
def heater_system():
    dt = Quantity(1.0, Unit("s"))
    states = ThermalStates()
    controls = ThermalControlElements(heater_power=10.0)
    algebraic = ThermalAlgebraic()
    constants = ThermalConstants(ambient_temperature=295.0, time_constant=5.0)

    temp_sensor = SampledDelayedSensor(
        measurement_tag="temperature",
        alias_tag="temp_meas",
        sampling_period=0.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
    )
    mv_sensor = SampledDelayedSensor(
        measurement_tag="heater_power",
        alias_tag="heater_power",
        sampling_period=0.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
    )

    system = create_system(
        system_class=HeaterSystem,
        dt=dt,
        initial_states=states,
        initial_controls=controls,
        initial_algebraic=algebraic,
        system_constants=constants,
        sensors=[temp_sensor, mv_sensor],
        calculations=[],
        controllers=[],
        use_numba=False,
        record_history=True,
        solver_options={"method": "RK45", "rtol": 1e-9, "atol": 1e-9},
    )
    return system


def test_system_step_advances_state(heater_system):
    system = heater_system
    system.step(duration=Quantity(3.0, Unit("s")))

    expected = (300.0 - (295.0 + 10.0)) * np.exp(-3.0 / 5.0) + (295.0 + 10.0)
    assert system.measurable_quantities.states.temperature == pytest.approx(expected, rel=1e-4)
    assert system.time == pytest.approx(3.0)

    history = system.history
    assert "temperature" in history
    assert len(history["temperature"]) == 3

    measured = system.measured_history["sensors"]["temp_meas"]
    assert len(measured) >= 3


def test_controller_management_in_system(heater_system, heater_mv_range):
    dt = Quantity(1.0, Unit("s"))
    states = ThermalStates()
    controls = ThermalControlElements(heater_power=10.0)
    algebraic = ThermalAlgebraic()
    constants = ThermalConstants(ambient_temperature=295.0, time_constant=5.0)

    sensor = SampledDelayedSensor(
        measurement_tag="temperature",
        alias_tag="temp_meas",
        sampling_period=0.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
    )
    mv_sensor = SampledDelayedSensor(
        measurement_tag="heater_power",
        alias_tag="heater_power",
        sampling_period=0.0,
        deadtime=0.0,
        coefficient_of_variance=0.0,
    )

    controller = SimpleController(
        mv_tag="heater_power",
        cv_tag="temp_meas",
        sp_trajectory=Trajectory(y0=300.0, unit=UNIT_TEMPERATURE),
        mv_range=heater_mv_range,
    )

    system = create_system(
        system_class=HeaterSystem,
        dt=dt,
        initial_states=states,
        initial_controls=controls,
        initial_algebraic=algebraic,
        system_constants=constants,
        sensors=[sensor, mv_sensor],
        calculations=[],
        controllers=[controller],
        use_numba=False,
        record_history=False,
        solver_options={"method": "RK45", "rtol": 1e-9, "atol": 1e-9},
    )

    assert controller.cv_tag in system.controller_dictionary
    new_mode = system.set_controller_mode("temp_meas", "tracking")
    assert new_mode == ControllerMode.TRACKING

    system.set_controller_mode("temp_meas", ControllerMode.AUTO)
    trajectory = system.extend_controller_trajectory("temp_meas", value=310.0)
    assert trajectory(0.0) == pytest.approx(310.0)
