from enum import Enum
from typing import ClassVar, Mapping

from numba import njit
from numba.typed.typeddict import Dict as NDict
import numpy as np
from numpy.typing import NDArray

from modular_simulation.measurables import AlgebraicStates, Constants, ControlElements, States
from modular_simulation.framework import FastSystem, System


# 1. Define the Data Structures for the System
# ============================================



class EnergyBalanceStates(States):
    """Pydantic model for the differential states of the system."""

    V: float
    A: float
    B: float
    T: float
    T_J: float


class EnergyBalanceControlElements(ControlElements):
    """dataclass for the externally controlled variables."""

    F_in: float  # Inlet flow rate
    T_J_in: float  # Jacket inlet flow rate


class EnergyBalanceAlgebraicStates(AlgebraicStates):
    """Pydantic model for algebraic states."""

    F_out: float  # Outlet flow rate, an algebraic function of volume


class EnergyBalanceConstants(Constants):
    """Container for the physical constants used in the model."""

    k0: float
    activation_energy: float
    gas_constant: float
    Cv: float
    CA_in: float
    T_in: float
    reaction_enthalpy: float
    rho_cp: float
    overall_heat_transfer_coefficient: float
    heat_transfer_area: float
    jacket_volume: float
    jacket_rho_cp: float
    jacket_flow: float


# 2. Define the System Dynamics
# =============================


class EnergyBalanceSystem(System):
    """Dynamic model of an irreversible reaction with an energy balance."""

    
    @staticmethod
    def calculate_algebraic_values(
        y: NDArray,
        u: NDArray,
        k: NDArray,
        y_map: Mapping[str, slice],
        u_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
    ) -> NDArray:
        """Calculate the outlet flow (F_out) from the current reactor volume."""
        return_array = np.zeros(len(algebraic_map))

        volume = max(1e-6, float(y[y_map["V"]]))  # type: ignore[arg-type]
        Cv = float(k[k_map["Cv"]])  # type: ignore[arg-type]
        return_array[algebraic_map["F_out"]] = Cv * np.sqrt(volume)
        return return_array

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
        """Calculate the differential state derivatives."""

        F_out = float(algebraic[algebraic_map["F_out"]])  # type: ignore[arg-type]
        F_in = float(u[u_map["F_in"]])  # type: ignore[arg-type]
        F_J_in = float(k[k_map["jacket_flow"]])  # type: ignore[arg-type]

        k0 = float(k[k_map["k0"]])  # type: ignore[arg-type]
        activation_energy = float(k[k_map["activation_energy"]])  # type: ignore[arg-type]
        gas_constant = float(k[k_map["gas_constant"]])  # type: ignore[arg-type]
        CA_in = float(k[k_map["CA_in"]])  # type: ignore[arg-type]
        feed_temperature = float(k[k_map["T_in"]])  # type: ignore[arg-type]
        reaction_enthalpy = float(k[k_map["reaction_enthalpy"]])  # type: ignore[arg-type]
        rho_cp = max(1e-9, float(k[k_map["rho_cp"]]))  # type: ignore[arg-type]
        overall_heat_transfer_coefficient = float(k[k_map["overall_heat_transfer_coefficient"]])  # type: ignore[arg-type]
        heat_transfer_area = float(k[k_map["heat_transfer_area"]])  # type: ignore[arg-type]
        jacket_volume = max(1e-9, float(k[k_map["jacket_volume"]]))  # type: ignore[arg-type]
        jacket_rho_cp = max(1e-9, float(k[k_map["jacket_rho_cp"]]))  # type: ignore[arg-type]
        jacket_inlet_temperature = float(u[u_map["T_J_in"]])  # type: ignore[arg-type]

        UA = overall_heat_transfer_coefficient * heat_transfer_area

        volume = max(1e-6, float(y[y_map["V"]]))  # type: ignore[arg-type]
        molarity_A = float(y[y_map["A"]])  # type: ignore[arg-type]
        molarity_B = float(y[y_map["B"]])  # type: ignore[arg-type]
        reactor_temperature = float(y[y_map["T"]])  # type: ignore[arg-type]
        jacket_temperature = float(y[y_map["T_J"]])  # type: ignore[arg-type]

        arrhenius_temperature = max(1e-6, reactor_temperature)
        rate_constant = k0 * np.exp(-activation_energy / (gas_constant * arrhenius_temperature))
        reaction_rate = molarity_A * volume * rate_constant

        dy = np.zeros_like(y)
        dV_dt = F_in - F_out
        dy[y_map["V"]] = dV_dt  # type: ignore[arg-type]

        dy[y_map["A"]] = (1.0 / volume) * (
            -reaction_rate
            + F_in * CA_in
            - F_out * molarity_A
            - molarity_A * dV_dt
        )

        dy[y_map["B"]] = (1.0 / volume) * (
            2.0 * reaction_rate
            - F_out * molarity_B
            - molarity_B * dV_dt
        )

        heat_generation = reaction_enthalpy * reaction_rate
        dy[y_map["T"]] = (
            (F_in / volume) * (feed_temperature - reactor_temperature)
            + heat_generation / (rho_cp * volume)
            - UA * (reactor_temperature - jacket_temperature) / (rho_cp * volume)
        )

        dy[y_map["T_J"]] = (
            (F_J_in / jacket_volume)
            * (jacket_inlet_temperature - jacket_temperature)
            + UA * (reactor_temperature - jacket_temperature)
            / (jacket_rho_cp * jacket_volume)
        )

        return dy


# --- Fast System Implementation ---


class EnergyBalanceFastSystem(FastSystem):
    """A performance-optimized implementation of the energy balance system."""

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
        result = np.zeros(algebraic_size)
        volume_slice = y_map["V"]
        volume = y[volume_slice][0]
        if volume < 1e-6:
            volume = 1e-6

        Cv = k[k_map["Cv"]][0]
        result_slice = algebraic_map["F_out"]
        result[result_slice] = Cv * (volume ** 0.5)
        return result

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

        volume_slice = y_map["V"]
        volume = y[volume_slice][0]
        if volume < 1e-6:
            volume = 1e-6
        molarity_A = y[y_map["A"]][0]
        molarity_B = y[y_map["B"]][0]
        reactor_temperature = y[y_map["T"]][0]
        jacket_temperature = y[y_map["T_J"]][0]

        F_out = algebraic[algebraic_map["F_out"]][0]
        F_in = u[u_map["F_in"]][0]
        F_J_in = k[k_map["jacket_flow"]][0]

        k0 = k[k_map["k0"]][0]
        activation_energy = k[k_map["activation_energy"]][0]
        gas_constant = k[k_map["gas_constant"]][0]
        CA_in = k[k_map["CA_in"]][0]
        feed_temperature = k[k_map["T_in"]][0]
        reaction_enthalpy = k[k_map["reaction_enthalpy"]][0]
        rho_cp = k[k_map["rho_cp"]][0]
        if rho_cp < 1e-9:
            rho_cp = 1e-9
        overall_heat_transfer_coefficient = k[k_map["overall_heat_transfer_coefficient"]][0]
        heat_transfer_area = k[k_map["heat_transfer_area"]][0]
        jacket_volume = k[k_map["jacket_volume"]][0]
        if jacket_volume < 1e-9:
            jacket_volume = 1e-9
        jacket_rho_cp = k[k_map["jacket_rho_cp"]][0]
        if jacket_rho_cp < 1e-9:
            jacket_rho_cp = 1e-9
        jacket_inlet_temperature = u[u_map["T_J_in"]][0]

        UA = overall_heat_transfer_coefficient * heat_transfer_area

        arrhenius_temperature = reactor_temperature
        if arrhenius_temperature < 1e-6:
            arrhenius_temperature = 1e-6
        rate_constant = k0 * np.exp(-activation_energy / (gas_constant * arrhenius_temperature))
        reaction_rate = molarity_A * volume * rate_constant

        dy = np.zeros_like(y)
        dV_dt = F_in - F_out
        dy[y_map["V"]] = dV_dt

        dy[y_map["A"]] = (1.0 / volume) * (
            -reaction_rate
            + F_in * CA_in
            - F_out * molarity_A
            - molarity_A * dV_dt
        )

        dy[y_map["B"]] = (1.0 / volume) * (
            2.0 * reaction_rate
            - F_out * molarity_B
            - molarity_B * dV_dt
        )

        heat_generation = reaction_enthalpy * reaction_rate
        dy[y_map["T"]] = (
            (F_in / volume) * (feed_temperature - reactor_temperature)
            + heat_generation / (rho_cp * volume)
            - UA * (reactor_temperature - jacket_temperature) / (rho_cp * volume)
        )

        dy[y_map["T_J"]] = (
            (F_J_in / jacket_volume)
            * (jacket_inlet_temperature - jacket_temperature)
            + UA * (reactor_temperature - jacket_temperature)
            / (jacket_rho_cp * jacket_volume)
        )

        return dy
