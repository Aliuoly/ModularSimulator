
from typing import Annotated, Mapping
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from modular_simulation.measurables import AlgebraicStates, Constants, ControlElements, States
from modular_simulation.framework.system_old import System
from modular_simulation.usables import CalculationBase, Constant, MeasuredTag, OutputTag
from astropy.units import Unit


class VanDeVusseStates(States):
    """Differential state variables for the Van de Vusse CSTR."""

    Ca: Annotated[float, Unit("mol/L")] = Field(description="Concentration of A in the reactor [mol/L]")
    Cb: Annotated[float, Unit("mol/L")] = Field(description="Concentration of B in the reactor [mol/L]")
    T: Annotated[float, Unit("K")] = Field(description="Reactor temperature [K]")
    Tk: Annotated[float, Unit("K")] = Field(description="Jacket temperature [K]")


class VanDeVusseControlElements(ControlElements):
    """Externally actuated variables for the Van de Vusse reactor."""

    Tj_in: Annotated[float, Unit("K")] = Field(description="Jacket inlet temperature [K]")


class VanDeVusseAlgebraicStates(AlgebraicStates):
    """No algebraic states are required for this model."""

    model_config = ConfigDict(extra="forbid")


class VanDeVusseConstants(Constants):
    """Physical constants for the Van de Vusse model."""

    F: Annotated[float, Unit("L/s")]
    Ca0: Annotated[float, Unit("mol/L")]
    T0: Annotated[float, Unit("K")]
    k10: Annotated[float, Unit("1/s")]
    E1: Annotated[float, Unit("J/mol")]
    dHr1: Annotated[float, Unit("J/mol")]
    rho: Annotated[float, Unit("kg/L")]
    Cp: Annotated[float, Unit("J/(kg*K)")]
    kw: Annotated[float, Unit("J/(s*K*L**2)")]
    AR: Annotated[float, Unit("L**2")]
    VR: Annotated[float, Unit("L")]
    mK: Annotated[float, Unit("kg")]
    CpK: Annotated[float, Unit("J/(kg*K)")]
    Fj: Annotated[float, Unit("L/s")]


class HeatDutyCalculation(CalculationBase):
    """Calculate the instantaneous heat duty transferred between jacket and reactor."""

    heat_duty_tag: OutputTag

    Tk_tag: MeasuredTag
    T_tag: MeasuredTag

    kw: Constant
    area: Constant

    def _calculation_algorithm(
        self,
        t: float,
        inputs_dict: dict[str, float],
    ) -> dict[str, float]:
        Tk = inputs_dict[self.Tk_tag]
        T = inputs_dict[self.T_tag]

        heat_duty = self.kw * self.area * (Tk - T)
        return {self.heat_duty_tag: heat_duty}


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
