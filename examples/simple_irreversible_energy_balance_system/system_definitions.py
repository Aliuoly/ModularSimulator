import numpy as np
from enum import Enum
from typing import Dict, Any, Type, ClassVar, List
import numba  # type: ignore

from modular_simulation.measurables import States, ControlElements, AlgebraicStates
from modular_simulation.system import System, FastSystem
from numpy.typing import NDArray


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
    F_J_in: float  # Jacket inlet flow rate


class EnergyBalanceAlgebraicStates(AlgebraicStates):
    """Pydantic model for algebraic states."""

    F_out: float  # Outlet flow rate, an algebraic function of volume


# 2. Define the System Dynamics
# =============================

class EnergyBalanceSystem(System):
    """Dynamic model of an irreversible reaction with an energy balance."""

    @staticmethod
    def _calculate_algebraic_values(
        y: NDArray,
        StateMap: Type[Enum],
        control_elements: ControlElements,
        system_constants: Dict,
    ) -> Dict[str, Any]:
        """Calculate the outlet flow (F_out) from the current reactor volume."""

        volume = max(1e-6, y[StateMap.V.value])  # type: ignore
        F_out = system_constants["Cv"] * (volume**0.5)
        return {"F_out": F_out}

    @staticmethod
    def rhs(
        t: float,
        y: NDArray,
        StateMap: Type[Enum],
        algebraic_values_dict: Dict[str, Any],
        control_elements: ControlElements,
        system_constants: Dict,
    ) -> NDArray:
        """Calculate the differential state derivatives."""

        F_out = algebraic_values_dict["F_out"]
        F_in = control_elements.F_in
        F_J_in = control_elements.F_J_in

        k0 = system_constants["k0"]
        activation_energy = system_constants["activation_energy"]
        gas_constant = system_constants.get("gas_constant", 8.314)
        CA_in = system_constants["CA_in"]
        feed_temperature = system_constants["T_in"]
        reaction_enthalpy = system_constants["reaction_enthalpy"]
        rho_cp = max(1e-9, system_constants["rho_cp"])
        overall_heat_transfer_coefficient = system_constants["overall_heat_transfer_coefficient"]
        heat_transfer_area = system_constants["heat_transfer_area"]
        UA = overall_heat_transfer_coefficient * heat_transfer_area

        jacket_volume = max(1e-9, system_constants["jacket_volume"])
        jacket_rho_cp = max(1e-9, system_constants["jacket_rho_cp"])
        jacket_inlet_temperature = system_constants["jacket_inlet_temperature"]

        volume = max(1e-6, y[StateMap.V.value])  # type: ignore
        molarity_A = y[StateMap.A.value]  # type: ignore
        molarity_B = y[StateMap.B.value]  # type: ignore
        reactor_temperature = y[StateMap.T.value]  # type: ignore
        jacket_temperature = y[StateMap.T_J.value]  # type: ignore

        arrhenius_temperature = max(1e-6, reactor_temperature) #type: ignore
        rate_constant = k0 * np.exp(-activation_energy / (gas_constant * arrhenius_temperature))
        reaction_rate = molarity_A * volume * rate_constant

        dy = np.zeros_like(y)
        dV_dt = F_in - F_out
        dy[StateMap.V.value] = dV_dt  # type: ignore

        dy[StateMap.A.value] = (1 / volume) * ( # type: ignore
            -reaction_rate
            + F_in * CA_in
            - F_out * molarity_A
            - molarity_A * dV_dt
        )  

        dy[StateMap.B.value] = (1 / volume) * (# type: ignore
            2.0 * reaction_rate
            - F_out * molarity_B
            - molarity_B * dV_dt
        )  

        heat_generation = (reaction_enthalpy) * reaction_rate
        dy[StateMap.T.value] = ( # type: ignore
            (F_in / volume) * (feed_temperature - reactor_temperature)
            + heat_generation / (rho_cp * volume)
            - UA * (reactor_temperature - jacket_temperature) / (rho_cp * volume)
        )  

        dy[StateMap.T_J.value] = ( # type: ignore
            (F_J_in / jacket_volume)
            * (jacket_inlet_temperature - jacket_temperature)
            + UA * (reactor_temperature - jacket_temperature)
            / (jacket_rho_cp * jacket_volume)
        )  # type: ignore

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
    @numba.jit(nopython = True)
    def _calculate_algebraic_values_fast(
        y: NDArray,
        control_elements_arr: NDArray,
        constants_arr: NDArray,
    ) -> NDArray:
        Cv = constants_arr[3]
        volume = max(1e-6, y[0])
        F_out = Cv * (volume**0.5)
        return np.array([F_out])

    @staticmethod
    @numba.jit(nopython = True)
    def rhs_fast(
        t: float,
        y: NDArray,
        constants_arr: NDArray,
        control_elements_arr: NDArray,
        algebraic_states_arr: NDArray,
    ) -> NDArray:
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

        heat_generation = (reaction_enthalpy) * reaction_rate
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