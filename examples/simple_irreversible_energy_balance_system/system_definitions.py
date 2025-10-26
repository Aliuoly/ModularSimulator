"""Dynamic model for the irreversible reaction with an energy balance."""
from __future__ import annotations

from typing import Annotated, Mapping

import numpy as np
from astropy.units import Unit
from numpy.typing import NDArray

from modular_simulation.core import DynamicModel, MeasurableMetadata, MeasurableType

SQRT_L_PER_S = Unit("L") ** 0.5 / Unit("s")


class EnergyBalanceModel(DynamicModel):
    """Dynamic model of an irreversible reaction coupled with heat balances."""

    V: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("L")),
    ] = 100.0
    A: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("mol/L")),
    ] = 1.0
    B: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("mol/L")),
    ] = 0.0
    T: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("K")),
    ] = 300.0
    T_J: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("K")),
    ] = 300.0

    F_out: Annotated[
        float,
        MeasurableMetadata(MeasurableType.ALGEBRAIC_STATE, Unit("L/s")),
    ] = 1.0

    F_in: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONTROL_ELEMENT, Unit("L/s")),
    ] = 1.0
    T_J_in: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONTROL_ELEMENT, Unit("K")),
    ] = 300.0

    k0: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("1/s")),
    ] = 1.5e9
    activation_energy: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("J/mol")),
    ] = 72_500.0
    gas_constant: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("J/(mol*K)")),
    ] = 8.314
    Cv: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, SQRT_L_PER_S),
    ] = 2.0
    CA_in: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("mol/L")),
    ] = 2.0
    T_in: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("K")),
    ] = 300.0
    reaction_enthalpy: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("J/mol")),
    ] = 825_000.0
    rho_cp: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("J/(K*L)")),
    ] = 4_000.0
    overall_heat_transfer_coefficient: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("J/(s*K*L**2)")),
    ] = 500_000.0
    heat_transfer_area: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("L**2")),
    ] = 10.0
    jacket_volume: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("L")),
    ] = 500_000.0
    jacket_rho_cp: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("J/(K*L)")),
    ] = 3_200.0
    jacket_flow: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("L/s")),
    ] = 500.0

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
        """Calculate the outlet flow based on volume."""

        result = np.zeros(algebraic_size)
        volume = max(1.0e-6, y[y_map["V"]][0])
        Cv = k[k_map["Cv"]][0]
        result[algebraic_map["F_out"]] = Cv * np.sqrt(volume)
        return result

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
        """Evaluate the coupled mass and energy balance derivatives."""

        dy = np.zeros_like(y)

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

        dV_dt = F_in - F_out
        dy[y_map["V"]] = dV_dt

        dy[y_map["A"]] = (1.0 / volume) * (
            -reaction_rate
            + F_in * CA_in
            - F_out * molarity_A
            - molarity_A * dV_dt
        )

        dy[y_map["B"]] = (1.0 / volume) * (
            2.0 * reaction_rate - F_out * molarity_B - molarity_B * dV_dt
        )

        heat_generation = reaction_enthalpy * reaction_rate
        dy[y_map["T"]] = (
            (F_in / volume) * (feed_temperature - reactor_temperature)
            + heat_generation / (rho_cp * volume)
            - UA * (reactor_temperature - jacket_temperature) / (rho_cp * volume)
        )

        dy[y_map["T_J"]] = (
            (F_J_in / jacket_volume) * (jacket_inlet_temperature - jacket_temperature)
            + UA * (reactor_temperature - jacket_temperature)
            / (jacket_rho_cp * jacket_volume)
        )

        return dy


__all__ = ["EnergyBalanceModel"]
