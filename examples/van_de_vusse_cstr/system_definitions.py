import numpy as np
import numba  # type: ignore
from enum import Enum
from typing import Any, ClassVar, Dict, List, Type

from pydantic import ConfigDict, Field, PrivateAttr

from modular_simulation.control_system import Trajectory, Controller
from modular_simulation.measurables import ControlElements, States, AlgebraicStates
from modular_simulation.system import FastSystem, System
from modular_simulation.usables import Calculation, Measurement
from numpy.typing import NDArray


class VanDeVusseStateMap(Enum):
    """Mapping from state names to indices in the state array."""

    Ca = 0
    Cb = 1
    T = 2
    Tk = 3


class VanDeVusseStates(States):
    """Differential state variables for the Van de Vusse CSTR."""

    model_config = ConfigDict(extra="forbid")
    StateMap: ClassVar = VanDeVusseStateMap

    Ca: float = Field(description="Concentration of A in the reactor [mol/L]")
    Cb: float = Field(description="Concentration of B in the reactor [mol/L]")
    T: float = Field(description="Reactor temperature [°C]")
    Tk: float = Field(description="Jacket temperature [°C]")


class VanDeVusseControlElements(ControlElements):
    """Externally actuated variables for the Van de Vusse reactor."""
    Tj_in: float = Field(description="Jacket inlet temperature [°C]")


class HeatDutyCalculation(Calculation):
    """Calculate the instantaneous heat duty transferred between jacket and reactor."""

    kw: float
    area: float

    def calculate(self, usable_results: Dict[str, Measurement]) -> Measurement:
        jacket_measurement = usable_results["Tk"]
        reactor_measurement = usable_results["T"]

        heat_duty = self.kw * self.area * (jacket_measurement.value - reactor_measurement.value)
        return Measurement(t=reactor_measurement.t, value=heat_duty)


class VanDeVusseAlgebraicStates(AlgebraicStates):
    """No algebraic states are required for this model."""

    model_config = ConfigDict(extra="forbid")


class VanDeVusseSystem(System):
    """Readable implementation of the Van de Vusse reactor model."""

    @staticmethod
    def _calculate_algebraic_values(
        y: NDArray,
        StateMap: Type[Enum],
        control_elements: ControlElements,
        system_constants: Dict,
    ) -> Dict[str, Any]:
        """No algebraic states are computed for this system."""

        return {}

    @staticmethod
    def rhs(
        t: float,
        y: NDArray,
        StateMap: Type[Enum],
        algebraic_values_dict: Dict[str, Any],
        control_elements: ControlElements,
        system_constants: Dict,
    ) -> NDArray:
        del algebraic_values_dict  # Unused, but part of the interface.

        Ca = y[StateMap.Ca.value]  # type: ignore[index]
        Cb = y[StateMap.Cb.value]  # type: ignore[index]
        T = y[StateMap.T.value]  # type: ignore[index]
        Tk = y[StateMap.Tk.value]  # type: ignore[index]

        F = system_constants["F"]
        Ca0 = system_constants["Ca0"]
        T0 = system_constants["T0"]
        k10 = system_constants["k10"]
        E1 = system_constants["E1"]
        dHr1 = system_constants["dHr1"]
        rho = system_constants["rho"]
        Cp = system_constants["Cp"]
        kw = system_constants["kw"]
        AR = system_constants["AR"]
        VR = system_constants["VR"]
        mK = system_constants["mK"]
        CpK = system_constants["CpK"]
        Fj = system_constants["Fj"]

        Tj_in = control_elements.Tj_in  # type: ignore[attr-defined]

        temperature_kelvin = max(1e-6, T + 273.15)

        k1 = k10 * np.exp(-E1 / temperature_kelvin)
        r1 = k1 * VR * Ca

        dCa = (-r1 + F * (Ca0 - Ca)) / VR
        dCb = (r1 - F * Cb) / VR

        heat_capacity_term = max(1e-9, rho * Cp * VR)
        dT = (
            F * rho * Cp * (T0 - T)
            - r1 * dHr1
            + kw * AR * (Tk - T)
        ) / heat_capacity_term

        jacket_capacity_term = max(1e-9, mK * CpK)
        dTk = (
            Fj * CpK * (Tj_in - Tk)
            + kw * AR * (T - Tk)
        ) / jacket_capacity_term

        dy = np.zeros_like(y)
        dy[StateMap.Ca.value] = dCa  # type: ignore[index]
        dy[StateMap.Cb.value] = dCb  # type: ignore[index]
        dy[StateMap.T.value] = dT  # type: ignore[index]
        dy[StateMap.Tk.value] = dTk  # type: ignore[index]
        return dy


class VanDeVusseFastSystem(FastSystem):
    """Numba-friendly version of the Van de Vusse reactor."""

    @staticmethod
    def _get_constants_map() -> List[str]:
        return [
            "F",
            "Ca0",
            "T0",
            "k10",
            "E1",
            "dHr1",
            "rho",
            "Cp",
            "kw",
            "AR",
            "VR",
            "mK",
            "CpK",
            "Fj",
        ]

    @staticmethod
    def _get_controls_map() -> List[str]:
        return ["Tj_in"]

    @staticmethod
    def _calculate_algebraic_values_fast(
        y: NDArray,
        control_elements_arr: NDArray,
        constants_arr: NDArray,
    ) -> NDArray:
        del y, control_elements_arr, constants_arr
        return np.zeros(0, dtype=np.float64)

    @staticmethod
    @numba.jit(nopython=True)
    def rhs_fast(
        t: float,
        y: NDArray,
        algebraic_states_arr: NDArray,
        control_elements_arr: NDArray,
        constants_arr: NDArray,
    ) -> NDArray:
        del t, algebraic_states_arr

        Ca, Cb, T, Tk = y

        F = constants_arr[0]
        Ca0 = constants_arr[1]
        T0 = constants_arr[2]
        k10 = constants_arr[3]
        E1 = constants_arr[4]
        dHr1 = constants_arr[5]
        rho = constants_arr[6]
        Cp = constants_arr[7]
        kw = constants_arr[8]
        AR = constants_arr[9]
        VR = constants_arr[10]
        mK = constants_arr[11]
        CpK = constants_arr[12]
        Fj = constants_arr[13]

        Tj_in = control_elements_arr[0]

        temperature_kelvin = max(1e-6, T + 273.15)
        k1 = k10 * np.exp(-E1 / temperature_kelvin)

        r1 = k1 * VR * Ca

        dCa = (-r1 + F * (Ca0 - Ca)) / VR
        dCb = (r1 - F * Cb) / VR


        heat_capacity_term = max(1e-9, rho * Cp * VR)
        dT = (
            F * rho * Cp * (T0 - T)
            - r1 * dHr1
            + kw * AR * (Tk - T)
        ) / heat_capacity_term

        jacket_capacity_term = max(1e-9, mK * CpK)
        dTk = (
            Fj * CpK * (Tj_in - Tk)
            + kw * AR * (T - Tk)
        ) / jacket_capacity_term

        return np.array([dCa, dCb, dT, dTk], dtype=np.float64)


class ConstantTrajectory(Trajectory):
    """Maintains a constant setpoint value over time."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, t: float) -> float:
        del t
        return self.value

    def change(self, new_value: float) -> None:
        self.value = new_value


class PIController(Controller):
    """Simple PI controller with configurable output limits."""

    Kp: float = Field(..., description="Proportional gain")
    Ti: float = Field(..., description="Integral time constant [time units]")
    u_min: float = Field(-np.inf, description="Lower saturation limit")
    u_max: float = Field(np.inf, description="Upper saturation limit")

    _last_t: float = PrivateAttr(default=0.0)
    _integral: float = PrivateAttr(default=0.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _control_algorithm(
        self,
        pv_value: float,
        sp_value: float,
        usable_results: Dict[str, Any],
        t: float,
    ) -> float:
        del usable_results

        dt = t - self._last_t
        if dt <= 0.0:
            dt = 1e-9
        self._last_t = t

        error = sp_value - pv_value
        self._integral += error * dt

        integral_term = 0.0
        if self.Ti > 0.0:
            integral_term = (self.Kp / self.Ti) * self._integral

        output = self.Kp * error + integral_term
        output = min(self.u_max, max(self.u_min, output))
        return output
