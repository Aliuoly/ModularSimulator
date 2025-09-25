from enum import Enum
from typing import ClassVar, Dict, Mapping
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from modular_simulation.measurables import AlgebraicStates, Constants, ControlElements, States
from modular_simulation.framework.system import System
from modular_simulation.usables import Calculation


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

    def _calculation_algorithm(
        self,
        t: float,
        inputs_dict: Dict[str, float],
        ) -> float:
        Tk = inputs_dict["Tk"]
        T = inputs_dict["T"]

        heat_duty = inputs_dict["kw"] * inputs_dict["area"] * (Tk - T)
        return heat_duty


class VanDeVusseSystem(System):
    """Readable implementation of the Van de Vusse reactor model."""

    @staticmethod
    def calculate_algebraic_values(
        y: NDArray, 
        u: NDArray,
        k: NDArray,
        y_map: Mapping[str, slice],
        u_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
        algebraic_size: int,
    ) -> NDArray:
        return np.zeros(algebraic_size)

    @staticmethod
    def rhs(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        algebraic: NDArray,
        y_map: Mapping[str, slice],
        u_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
        ) -> NDArray:

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
        dy[y_map["Ca"]] = dCa  
        dy[y_map["Cb"]] = dCb  
        dy[y_map["T"]] = dT  
        dy[y_map["Tk"]] = dTk  
        return dy
