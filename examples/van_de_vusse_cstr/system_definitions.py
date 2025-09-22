from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, List, Type

import numba  # type: ignore
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, PrivateAttr

from modular_simulation.control_system import Controller, Trajectory
from modular_simulation.measurables import AlgebraicStates, Constants, ControlElements, States
from modular_simulation.system import FastSystem, System
from modular_simulation.usables import Calculation, TimeValueQualityTriplet


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


class VanDeVusseAlgebraicStates(AlgebraicStates):
    """No algebraic states are required for this model."""

    model_config = ConfigDict(extra="forbid")


class VanDeVusseConstants(Constants):
    """Physical constants for the Van de Vusse model."""

    F: float
    Ca0: float
    T0: float
    k10: float
    E1: float
    dHr1: float
    rho: float
    Cp: float
    kw: float
    AR: float
    VR: float
    mK: float
    CpK: float
    Fj: float


class HeatDutyCalculation(Calculation):
    """Calculate the instantaneous heat duty transferred between jacket and reactor."""

    kw: float = Field(...)
    area: float = Field(...)

    def _calculation_algorithm(
        self,
        t: float,
        inputs_dict: Dict[str, TimeValueQualityTriplet | float | NDArray],
    ) -> TimeValueQualityTriplet:
        jacket_measurement = inputs_dict["Tk"]
        reactor_measurement = inputs_dict["T"]
        assert isinstance(jacket_measurement, TimeValueQualityTriplet)
        assert isinstance(reactor_measurement, TimeValueQualityTriplet)

        heat_duty = self.kw * self.area * (jacket_measurement.value - reactor_measurement.value)
        ok = jacket_measurement.ok and reactor_measurement.ok
        return TimeValueQualityTriplet(t=t, value=heat_duty, ok=ok)


class VanDeVusseSystem(System):
    """Readable implementation of the Van de Vusse reactor model."""

    @staticmethod
    def _calculate_algebraic_values(
        y: NDArray,
        y_map: Type[Enum],
        u: NDArray,
        u_map: Type[Enum],
        k: NDArray,
        k_map: Type[Enum],
    ) -> NDArray:
        del y, y_map, u, u_map, k, k_map
        return np.zeros(0, dtype=np.float64)

    @staticmethod
    def rhs(
        t: float,
        y: NDArray,
        y_map: Type[Enum],
        u: NDArray,
        u_map: Type[Enum],
        k: NDArray,
        k_map: Type[Enum],
        algebraic: NDArray,
        algebraic_map: Type[Enum],
    ) -> NDArray:
        del t, algebraic, algebraic_map

        Ca = float(y[y_map.Ca.value])  # type: ignore[arg-type]
        Cb = float(y[y_map.Cb.value])  # type: ignore[arg-type]
        T = float(y[y_map.T.value])  # type: ignore[arg-type]
        Tk = float(y[y_map.Tk.value])  # type: ignore[arg-type]

        F = float(k[k_map.F.value])  # type: ignore[arg-type]
        Ca0 = float(k[k_map.Ca0.value])  # type: ignore[arg-type]
        T0 = float(k[k_map.T0.value])  # type: ignore[arg-type]
        k10 = float(k[k_map.k10.value])  # type: ignore[arg-type]
        E1 = float(k[k_map.E1.value])  # type: ignore[arg-type]
        dHr1 = float(k[k_map.dHr1.value])  # type: ignore[arg-type]
        rho = float(k[k_map.rho.value])  # type: ignore[arg-type]
        Cp = float(k[k_map.Cp.value])  # type: ignore[arg-type]
        kw = float(k[k_map.kw.value])  # type: ignore[arg-type]
        AR = float(k[k_map.AR.value])  # type: ignore[arg-type]
        VR = float(k[k_map.VR.value])  # type: ignore[arg-type]
        mK = float(k[k_map.mK.value])  # type: ignore[arg-type]
        CpK = float(k[k_map.CpK.value])  # type: ignore[arg-type]
        Fj = float(k[k_map.Fj.value])  # type: ignore[arg-type]

        Tj_in = float(u[u_map.Tj_in.value])  # type: ignore[arg-type]

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
        dy[y_map.Ca.value] = dCa  # type: ignore[arg-type]
        dy[y_map.Cb.value] = dCb  # type: ignore[arg-type]
        dy[y_map.T.value] = dT  # type: ignore[arg-type]
        dy[y_map.Tk.value] = dTk  # type: ignore[arg-type]
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
        cv_value: TimeValueQualityTriplet,
        sp_value: TimeValueQualityTriplet | float | NDArray,
        t: float,
    ) -> TimeValueQualityTriplet:
        if isinstance(sp_value, TimeValueQualityTriplet):
            sp_ok = sp_value.ok
            sp_val = float(sp_value.value)
        else:
            sp_ok = True
            sp_val = float(sp_value)

        if not cv_value.ok or not sp_ok:
            return TimeValueQualityTriplet(t=t, value=float(self._last_value.value), ok=False)

        dt = t - self._last_t
        if dt <= 0.0:
            dt = 1e-9
        self._last_t = t

        error = sp_val - float(cv_value.value)
        self._integral += error * dt

        integral_term = 0.0
        if self.Ti > 0.0:
            integral_term = (self.Kp / self.Ti) * self._integral

        output = self.Kp * error + integral_term
        output = float(np.clip(output, self.u_min, self.u_max))
        return TimeValueQualityTriplet(t=t, value=output, ok=True)


class ConstantTrajectory(Trajectory):
    """Maintains a constant setpoint value over time."""

    def __init__(self, value: float) -> None:
        super().__init__(y0=value)

    def __call__(self, t: float) -> float:
        return float(self.eval(t))

    def change(self, new_value: float) -> None:
        self.set_now(0.0, new_value)
