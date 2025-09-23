from enum import Enum
from typing import ClassVar, Dict, Mapping

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, PrivateAttr

from modular_simulation.control_system import Controller, Trajectory
from modular_simulation.measurables import AlgebraicStates, Constants, ControlElements, States
from modular_simulation.framework import FastSystem, System
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


class VanDeVusseFastSystem(FastSystem, VanDeVusseSystem):
    """Fast variant that reuses the readable implementation via numba."""

