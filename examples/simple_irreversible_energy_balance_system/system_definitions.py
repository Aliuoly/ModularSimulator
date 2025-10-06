from typing import Annotated, Mapping
import numpy as np
from numpy.typing import NDArray
from modular_simulation.measurables import AlgebraicStates, Constants, ControlElements, States
from modular_simulation.framework import System
from astropy.units import Unit, UnitBase


# 1. Define the Data Structures for the System
# ============================================



class EnergyBalanceStates(States):
    """Pydantic model for the differential states of the system."""

    V: Annotated[float, Unit("L")]
    A: Annotated[float, Unit("mol")/Unit("L")]
    B: Annotated[float, Unit("mol/L")]
    T: Annotated[float, Unit("K")]
    T_J: Annotated[float, Unit("K")]


class EnergyBalanceControlElements(ControlElements):
    """dataclass for the externally controlled variables."""

    F_in: Annotated[float, Unit("L/s")]  # Inlet flow rate
    T_J_in: Annotated[float, Unit("K")]  # Jacket inlet temperature


class EnergyBalanceAlgebraicStates(AlgebraicStates):
    """Pydantic model for algebraic states."""

    F_out: Annotated[float, Unit("L/s")]  # Outlet flow rate, an algebraic function of volume


class EnergyBalanceConstants(Constants):
    """Container for the physical constants used in the model."""

    k0: Annotated[float, Unit("1/s")]
    activation_energy: Annotated[float, Unit("J/mol")]
    gas_constant: Annotated[float, Unit("J/(mol*K)")]
    Cv: Annotated[float, Unit("L")**0.5/Unit("s")]
    CA_in: Annotated[float, Unit("mol/L")]
    T_in: Annotated[float, Unit("K")]
    reaction_enthalpy: Annotated[float, Unit("J/mol")]
    rho_cp: Annotated[float, Unit("J/(K*L)")]
    overall_heat_transfer_coefficient: Annotated[float, Unit("J/(s*K*L**2)")]
    heat_transfer_area: Annotated[float, Unit("L**2")]
    jacket_volume: Annotated[float, Unit("L")]
    jacket_rho_cp: Annotated[float, Unit("J/(K*L)")]
    jacket_flow: Annotated[float, Unit("L/s")]


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
        algebraic_size: int,
    ) -> NDArray:
        """Calculate the outlet flow (F_out) from the current reactor volume."""
        return_array = np.zeros(algebraic_size)

        volume = max(1e-6, y[y_map["V"]][0]) 
        Cv = k[k_map["Cv"]][0]
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

        F_out = algebraic[algebraic_map["F_out"]][0]
        F_in = u[u_map["F_in"]][0]
        F_J_in = k[k_map["jacket_flow"]][0]

        k0 = k[k_map["k0"]][0]
        activation_energy = k[k_map["activation_energy"]][0]
        gas_constant = k[k_map["gas_constant"]][0]
        CA_in = k[k_map["CA_in"]][0]
        feed_temperature = k[k_map["T_in"]][0]
        reaction_enthalpy = k[k_map["reaction_enthalpy"]][0]
        rho_cp = max(1e-9, k[k_map["rho_cp"]][0])
        overall_heat_transfer_coefficient = k[k_map["overall_heat_transfer_coefficient"]][0]
        heat_transfer_area = k[k_map["heat_transfer_area"]][0]
        jacket_volume = max(1e-9, k[k_map["jacket_volume"]][0])
        jacket_rho_cp = max(1e-9, k[k_map["jacket_rho_cp"]][0])
        jacket_inlet_temperature = u[u_map["T_J_in"]][0]

        UA = overall_heat_transfer_coefficient * heat_transfer_area

        volume = max(1e-6, y[y_map["V"]][0])
        molarity_A = y[y_map["A"]][0]
        molarity_B = y[y_map["B"]][0]
        reactor_temperature = y[y_map["T"]][0]
        jacket_temperature = y[y_map["T_J"]][0]

        arrhenius_temperature = max(1e-6, reactor_temperature)
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