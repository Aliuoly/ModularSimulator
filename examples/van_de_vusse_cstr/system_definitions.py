from enum import Enum
from typing import ClassVar, Dict, Mapping

from numba import njit
from numba.typed.typeddict import Dict as NDict
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, PrivateAttr

from modular_simulation.control_system import Controller, Trajectory
from modular_simulation.measurables import AlgebraicStates, Constants, ControlElements, States
from modular_simulation.framework.system import FastSystem, System
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
        y_map: Mapping[str, slice],
        u: NDArray,
        u_map: Mapping[str, slice],
        k: NDArray,
        k_map: Mapping[str, slice],
    ) -> NDArray:
        del y, y_map, u, u_map, k, k_map
        return np.zeros(0, dtype=np.float64)

    @staticmethod
    def rhs(
        t: float,
        y: NDArray,
        y_map: Mapping[str, slice],
        u: NDArray,
        u_map: Mapping[str, slice],
        k: NDArray,
        k_map: Mapping[str, slice],
        algebraic: NDArray,
        algebraic_map: Mapping[str, slice],
    ) -> NDArray:
        del t, algebraic, algebraic_map

        Ca = float(y[y_map["Ca"]])  # type: ignore[arg-type]
        Cb = float(y[y_map["Cb"]])  # type: ignore[arg-type]
        T = float(y[y_map["T"]])  # type: ignore[arg-type]
        Tk = float(y[y_map["Tk"]])  # type: ignore[arg-type]

        F = float(k[k_map["F"]])  # type: ignore[arg-type]
        Ca0 = float(k[k_map["Ca0"]])  # type: ignore[arg-type]
        T0 = float(k[k_map["T0"]])  # type: ignore[arg-type]
        k10 = float(k[k_map["k10"]])  # type: ignore[arg-type]
        E1 = float(k[k_map["E1"]])  # type: ignore[arg-type]
        dHr1 = float(k[k_map["dHr1"]])  # type: ignore[arg-type]
        rho = float(k[k_map["rho"]])  # type: ignore[arg-type]
        Cp = float(k[k_map["Cp"]])  # type: ignore[arg-type]
        kw = float(k[k_map["kw"]])  # type: ignore[arg-type]
        AR = float(k[k_map["AR"]])  # type: ignore[arg-type]
        VR = float(k[k_map["VR"]])  # type: ignore[arg-type]
        mK = float(k[k_map["mK"]])  # type: ignore[arg-type]
        CpK = float(k[k_map["CpK"]])  # type: ignore[arg-type]
        Fj = float(k[k_map["Fj"]])  # type: ignore[arg-type]

        Tj_in = float(u[u_map["Tj_in"]])  # type: ignore[arg-type]

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
        dy[y_map["Ca"]] = dCa  # type: ignore[arg-type]
        dy[y_map["Cb"]] = dCb  # type: ignore[arg-type]
        dy[y_map["T"]] = dT  # type: ignore[arg-type]
        dy[y_map["Tk"]] = dTk  # type: ignore[arg-type]
        return dy


class VanDeVusseFastSystem(FastSystem):
    """Numba-friendly version of the Van de Vusse reactor."""

    @staticmethod
    @njit
    def calculate_algebraic_values_fast(
        y: NDArray,
        u: NDArray,
        k: NDArray,
        y_map: NDict,
        u_map: NDict,
        k_map: NDict,
        algebraic_map: NDict,
        algebraic_size: int,
    ) -> NDArray:
        del y, u, k, y_map, u_map, k_map, algebraic_map
        return np.zeros(algebraic_size)

    @staticmethod
    @njit
    def rhs_fast(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        algebraic: NDArray,
        y_map: NDict,
        u_map: NDict,
        k_map: NDict,
        algebraic_map: NDict,
    ) -> NDArray:
        del t, algebraic, algebraic_map

        Ca = y[y_map["Ca"]][0]
        Cb = y[y_map["Cb"]][0]
        T = y[y_map["T"]][0]
        Tk = y[y_map["Tk"]][0]

        F = k[k_map["F"]][0]
        Ca0 = k[k_map["Ca0"]][0]
        T0 = k[k_map["T0"]][0]
        k10 = k[k_map["k10"]][0]
        E1 = k[k_map["E1"]][0]
        dHr1 = k[k_map["dHr1"]][0]
        rho = k[k_map["rho"]][0]
        Cp = k[k_map["Cp"]][0]
        kw = k[k_map["kw"]][0]
        AR = k[k_map["AR"]][0]
        VR = k[k_map["VR"]][0]
        mK = k[k_map["mK"]][0]
        CpK = k[k_map["CpK"]][0]
        Fj = k[k_map["Fj"]][0]

        Tj_in = u[u_map["Tj_in"]][0]

        temperature_kelvin = T + 273.15
        if temperature_kelvin < 1e-6:
            temperature_kelvin = 1e-6

        k1 = k10 * np.exp(-E1 / temperature_kelvin)
        r1 = k1 * VR * Ca

        dCa = (-r1 + F * (Ca0 - Ca)) / VR
        dCb = (r1 - F * Cb) / VR

        heat_capacity_term = rho * Cp * VR
        if heat_capacity_term < 1e-9:
            heat_capacity_term = 1e-9
        dT = (
            F * rho * Cp * (T0 - T)
            - r1 * dHr1
            + kw * AR * (Tk - T)
        ) / heat_capacity_term

        jacket_capacity_term = mK * CpK
        if jacket_capacity_term < 1e-9:
            jacket_capacity_term = 1e-9
        dTk = (
            Fj * CpK * (Tj_in - Tk)
            + kw * AR * (T - Tk)
        ) / jacket_capacity_term

        dy = np.zeros_like(y)
        dy[y_map["Ca"]] = dCa
        dy[y_map["Cb"]] = dCb
        dy[y_map["T"]] = dT
        dy[y_map["Tk"]] = dTk

        return dy


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
