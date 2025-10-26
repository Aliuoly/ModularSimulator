"""Dynamic model and utilities for the Van de Vusse CSTR example."""
from __future__ import annotations

from typing import Annotated, Mapping

import numpy as np
from astropy.units import Unit
from numpy.typing import NDArray
from pydantic import Field

from modular_simulation.core import DynamicModel, MeasurableMetadata, MeasurableType
from modular_simulation.interfaces import CalculationBase, Constant, MeasuredTag, OutputTag


class VanDeVusseModel(DynamicModel):
    """Dynamic model for the Van de Vusse reactor."""

    Ca: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("mol/L")),
    ] = Field(2.2291, description="Concentration of A in the reactor [mol/L]")
    Cb: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("mol/L")),
    ] = Field(1.0417, description="Concentration of B in the reactor [mol/L]")
    T: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("K")),
    ] = Field(79.591, description="Reactor temperature [K]")
    Tk: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("K")),
    ] = Field(77.69, description="Jacket temperature [K]")

    Tj_in: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONTROL_ELEMENT, Unit("K")),
    ] = Field(77.69, description="Jacket inlet temperature [K]")

    F: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("L/s")),
    ] = 14.19
    Ca0: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("mol/L")),
    ] = 5.1
    T0: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("K")),
    ] = 104.9
    k10: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("1/s")),
    ] = 1.287e10
    E1: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("K")),
    ] = 9758.3
    dHr1: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("J/mol")),
    ] = 4.2
    rho: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("kg/L")),
    ] = 0.9342
    Cp: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("J/(kg*K)")),
    ] = 3.01
    kw: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("J/(s*K*L**2)")),
    ] = 4032.0
    AR: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("L**2")),
    ] = 0.215
    VR: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("L")),
    ] = 10.0
    mK: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("kg")),
    ] = 5.0
    CpK: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("J/(kg*K)")),
    ] = 2.0
    Fj: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("L/s")),
    ] = 10.0

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
        u_map: Mapping[str, slice],
        y_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
    ) -> NDArray:
        dy = np.zeros_like(y)

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

        dy[y_map["Ca"]] = dCa
        dy[y_map["Cb"]] = dCb
        dy[y_map["T"]] = dT
        dy[y_map["Tk"]] = dTk
        return dy


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


__all__ = ["VanDeVusseModel", "HeatDutyCalculation"]
