"""Van de Vusse CSTR process description for the ProcessModel API."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Annotated

import numpy as np
from numpy.typing import NDArray

from modular_simulation.measurables import (
    ProcessModel,
    StateMetadata as M,
    StateType as T,
)
from modular_simulation.utils.typing import ArrayIndex

DIFFERENTIAL = partial(M, type=T.DIFFERENTIAL)
CONTROLLED = partial(M, type=T.CONTROLLED)
CONSTANT = partial(M, type=T.CONSTANT)


class VanDeVusseProcessModel(ProcessModel):
    """Process model capturing the Van de Vusse CSTR dynamics."""

    # Differential states (C in mol/L, T in degC)
    Ca: Annotated[float, DIFFERENTIAL(unit="mol/L", description="Reactor concentration of A")]= 2.2291
    Cb: Annotated[float, DIFFERENTIAL(unit="mol/L", description="Reactor concentration of B")]= 1.0417
    T: Annotated[float, DIFFERENTIAL(unit="deg_C", description="Reactor temperature")]= 79.591
    Tk: Annotated[float, DIFFERENTIAL(unit="deg_C", description="Jacket temperature")]= 77.69

    # Manipulated variable
    Tj_in: Annotated[float, CONTROLLED(unit="deg_C", description="Jacket inlet temperature")]= 77.69

    # Constants (converted to per-second units where appropriate)
    F: Annotated[float, CONSTANT(unit="L/s", description="Feed volumetric flow rate")]= 14.19 / 3600.0
    Ca0: Annotated[float, CONSTANT(unit="mol/L", description="Feed concentration of A")]= 5.1
    T0: Annotated[float, CONSTANT(unit="deg_C", description="Feed temperature")]= 104.9
    k10: Annotated[float, CONSTANT(unit="1/s", description="Pre-exponential factor for A -> B")]= 1.287e10 / 3600.0
    E1: Annotated[float, CONSTANT(unit="K", description="Activation energy over gas constant")]= 9758.3
    dHr1: Annotated[float, CONSTANT(unit="kJ/mol", description="Heat of reaction for A -> B")]= 4.2
    rho: Annotated[float, CONSTANT(unit="kg/L", description="Liquid density")]= 0.9342
    Cp: Annotated[float, CONSTANT(unit="kJ/(kg*K)", description="Heat capacity of liquid phase")]= 3.01
    kw: Annotated[float, CONSTANT(unit="kJ/(s*K*m2)", description="Heat transfer coefficient")]= 4032.0 / 3600.0
    AR: Annotated[float, CONSTANT(unit="m2", description="Heat transfer area")]= 0.215
    VR: Annotated[float, CONSTANT(unit="L", description="Reactor volume")]= 10.0
    mK: Annotated[float, CONSTANT(unit="kg", description="Jacket mass")]= 5.0
    CpK: Annotated[float, CONSTANT(unit="kJ/(kg*K)", description="Jacket heat capacity")]= 2.0
    Fj: Annotated[float, CONSTANT(unit="kg/s", description="Jacket fluid mass flow rate")]= 10.0 / 3600.0

    @staticmethod
    def calculate_algebraic_values(
        y: NDArray[np.float64],
        u: NDArray[np.float64],
        k: NDArray[np.float64],
        y_map: Mapping[str, ArrayIndex],
        u_map: Mapping[str, ArrayIndex],
        k_map: Mapping[str, ArrayIndex],
        algebraic_map: Mapping[str, ArrayIndex],
        algebraic_size: int,
    ) -> NDArray[np.float64]:
        """The Van de Vusse model has no algebraic states."""
        return np.zeros(algebraic_size, dtype=np.float64)

    @staticmethod
    def differential_rhs(
        t: float,
        y: NDArray[np.float64],
        u: NDArray[np.float64],
        k: NDArray[np.float64],
        algebraic: NDArray[np.float64],
        y_map: Mapping[str, ArrayIndex],
        u_map: Mapping[str, ArrayIndex],
        k_map: Mapping[str, ArrayIndex],
        algebraic_map: Mapping[str, ArrayIndex],
    ) -> NDArray[np.float64]:
        """Differential equations describing the Van de Vusse reactor."""
        Ca = y[y_map["Ca"]]
        Cb = y[y_map["Cb"]]
        T = y[y_map["T"]]
        Tk = y[y_map["Tk"]]

        F = k[k_map["F"]]
        Ca0 = k[k_map["Ca0"]]
        T0 = k[k_map["T0"]]
        k10 = k[k_map["k10"]]
        E1 = k[k_map["E1"]]
        dHr1 = k[k_map["dHr1"]]
        rho = k[k_map["rho"]]
        Cp = k[k_map["Cp"]]
        kw = k[k_map["kw"]]
        AR = k[k_map["AR"]]
        VR = k[k_map["VR"]]
        mK = k[k_map["mK"]]
        CpK = k[k_map["CpK"]]
        Fj = k[k_map["Fj"]]

        Tj_in = u[u_map["Tj_in"]]

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
