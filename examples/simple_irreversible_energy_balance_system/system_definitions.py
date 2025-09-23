from enum import Enum
from typing import ClassVar, List, Type

import numba  # type: ignore
import numpy as np
from numpy.typing import NDArray

from modular_simulation.measurables import AlgebraicStates, Constants, ControlElements, States
from modular_simulation.system import FastSystem, System


# 1. Define the Data Structures for the System
# ============================================


class EnergyBalanceStateMap(Enum):
    """Maps the differential state names to their index in the NumPy array."""

    V = 0  # Reactor Volume
    A = 1  # Concentration of A
    B = 2  # Concentration of B
    T = 3  # Reactor temperature
    T_J = 4  # Cooling jacket temperature


class EnergyBalanceStates(States):
    """Pydantic model for the differential states of the system."""

    StateMap: ClassVar = EnergyBalanceStateMap
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
        y_map: Type[Enum],
        u_map: Type[Enum],
        k_map: Type[Enum],
        algebraic_map: Type[Enum],
    ) -> NDArray:
        """Calculate the outlet flow (F_out) from the current reactor volume."""
        return_array = np.zeros(len(algebraic_map))

        volume = max(1e-6, float(y[y_map.V.value]))  # type: ignore[arg-type]
        Cv = float(k[k_map.Cv.value])  # type: ignore[arg-type]
        return_array[algebraic_map.F_out.value] = Cv * np.sqrt(volume)
        return return_array

    @staticmethod
    def rhs(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        algebraic: NDArray,
        y_map: Type[Enum],
        u_map: Type[Enum],
        k_map: Type[Enum],
        algebraic_map: Type[Enum],
        ) -> NDArray:
        """Calculate the differential state derivatives."""

        del t
        F_out = float(algebraic[algebraic_map.F_out.value])  # type: ignore
        F_in = float(u[u_map.F_in.value])  # type: ignore[arg-type]as
        F_J_in = float(k[k_map.jacket_flow.value])  # type: ignore[arg-type]

        k0 = float(k[k_map.k0.value])  # type: ignore[arg-type]
        activation_energy = float(k[k_map.activation_energy.value])  # type: ignore[arg-type]
        gas_constant = float(k[k_map.gas_constant.value])  # type: ignore[arg-type]
        CA_in = float(k[k_map.CA_in.value])  # type: ignore[arg-type]
        feed_temperature = float(k[k_map.T_in.value])  # type: ignore[arg-type]
        reaction_enthalpy = float(k[k_map.reaction_enthalpy.value])  # type: ignore[arg-type]
        rho_cp = max(1e-9, float(k[k_map.rho_cp.value]))  # type: ignore[arg-type]
        overall_heat_transfer_coefficient = float(k[k_map.overall_heat_transfer_coefficient.value])  # type: ignore[arg-type]
        heat_transfer_area = float(k[k_map.heat_transfer_area.value])  # type: ignore[arg-type]
        jacket_volume = max(1e-9, float(k[k_map.jacket_volume.value]))  # type: ignore[arg-type]
        jacket_rho_cp = max(1e-9, float(k[k_map.jacket_rho_cp.value]))  # type: ignore[arg-type]
        jacket_inlet_temperature = float(u[u_map.T_J_in.value])  # type: ignore[arg-type]

        UA = overall_heat_transfer_coefficient * heat_transfer_area

        volume = max(1e-6, float(y[y_map.V.value]))  # type: ignore[arg-type]
        molarity_A = float(y[y_map.A.value])  # type: ignore[arg-type]
        molarity_B = float(y[y_map.B.value])  # type: ignore[arg-type]
        reactor_temperature = float(y[y_map.T.value])  # type: ignore[arg-type]
        jacket_temperature = float(y[y_map.T_J.value])  # type: ignore[arg-type]

        arrhenius_temperature = max(1e-6, reactor_temperature)
        rate_constant = k0 * np.exp(-activation_energy / (gas_constant * arrhenius_temperature))
        reaction_rate = molarity_A * volume * rate_constant

        dy = np.zeros_like(y)
        dV_dt = F_in - F_out
        dy[y_map.V.value] = dV_dt  # type: ignore[arg-type]

        dy[y_map.A.value] = (1.0 / volume) * (
            -reaction_rate
            + F_in * CA_in
            - F_out * molarity_A
            - molarity_A * dV_dt
        )

        dy[y_map.B.value] = (1.0 / volume) * (
            2.0 * reaction_rate
            - F_out * molarity_B
            - molarity_B * dV_dt
        )

        heat_generation = reaction_enthalpy * reaction_rate
        dy[y_map.T.value] = (
            (F_in / volume) * (feed_temperature - reactor_temperature)
            + heat_generation / (rho_cp * volume)
            - UA * (reactor_temperature - jacket_temperature) / (rho_cp * volume)
        )

        dy[y_map.T_J.value] = (
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
    def _get_constants_map() -> List[str]:
        return [
            "k0",
            "activation_energy",
            "gas_constant",
            "Cv",
            "CA_in",
            "T_in",
            "reaction_enthalpy",
            "rho_cp",
            "overall_heat_transfer_coefficient",
            "heat_transfer_area",
            "jacket_volume",
            "jacket_rho_cp",
            "jacket_inlet_temperature",
        ]

    @staticmethod
    def _get_controls_map() -> List[str]:
        return ["F_in", "F_J_in"]

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_algebraic_values_fast(
        y: NDArray,
        control_elements_arr: NDArray,
        constants_arr: NDArray,
    ) -> NDArray:
        del control_elements_arr
        Cv = constants_arr[3]
        volume = max(1e-6, y[0])
        F_out = Cv * np.sqrt(volume)
        return np.asarray([F_out], dtype=np.float64)

    @staticmethod
    @numba.jit(nopython=True)
    def rhs_fast(
        t: float,
        y: NDArray,
        algebraic_states_arr: NDArray,
        control_elements_arr: NDArray,
        constants_arr: NDArray,
    ) -> NDArray:
        del t
        V_idx, A_idx, B_idx, T_idx, Tj_idx = 0, 1, 2, 3, 4

        volume = max(1e-6, y[V_idx])
        molarity_A = y[A_idx]
        molarity_B = y[B_idx]
        reactor_temperature = y[T_idx]
        jacket_temperature = y[Tj_idx]

        k0 = constants_arr[0]
        activation_energy = constants_arr[1]
        gas_constant = constants_arr[2]

        CA_in = constants_arr[4]
        feed_temperature = constants_arr[5]
        reaction_enthalpy = constants_arr[6]
        rho_cp = max(1e-9, constants_arr[7])
        overall_heat_transfer_coefficient = constants_arr[8]
        heat_transfer_area = constants_arr[9]
        jacket_volume = max(1e-9, constants_arr[10])
        jacket_rho_cp = max(1e-9, constants_arr[11])
        jacket_inlet_temperature = constants_arr[12]

        UA = overall_heat_transfer_coefficient * heat_transfer_area

        F_in = control_elements_arr[0]
        F_J_in = control_elements_arr[1]
        F_out = algebraic_states_arr[0]

        arrhenius_temperature = max(1e-6, reactor_temperature)
        rate_constant = k0 * np.exp(-activation_energy / (gas_constant * arrhenius_temperature))
        reaction_rate = molarity_A * volume * rate_constant

        dy = np.zeros_like(y)
        dV_dt = F_in - F_out
        dy[V_idx] = dV_dt

        dy[A_idx] = (1.0 / volume) * (
            -reaction_rate
            + F_in * CA_in
            - F_out * molarity_A
            - molarity_A * dV_dt
        )

        dy[B_idx] = (1.0 / volume) * (
            2.0 * reaction_rate
            - F_out * molarity_B
            - molarity_B * dV_dt
        )

        heat_generation = reaction_enthalpy * reaction_rate
        dy[T_idx] = (
            (F_in / volume) * (feed_temperature - reactor_temperature)
            + heat_generation / (rho_cp * volume)
            - UA * (reactor_temperature - jacket_temperature) / (rho_cp * volume)
        )

        dy[Tj_idx] = (
            (F_J_in / jacket_volume)
            * (jacket_inlet_temperature - jacket_temperature)
            + UA * (reactor_temperature - jacket_temperature)
            / (jacket_rho_cp * jacket_volume)
        )

        return dy
