"""Energy balance process definition using the new :class:`ProcessModel` API."""

from __future__ import annotations

import numpy as np
from typing import Annotated
from collections.abc import Mapping
from numpy.typing import NDArray
from astropy.units import Unit

from modular_simulation.measurables import (
    ProcessModel,
    StateType as T,
    StateMetadata as M,
)
from modular_simulation.utils.typing import ArrayIndex

SQRT_L_PER_S = Unit("L") ** 0.5 / Unit("s")
JOULE_PER_MOL_K = Unit("J") / (Unit("mol") * Unit("K"))
JOULE_PER_LITER_K = Unit("J") / (Unit("L") * Unit("K"))
JOULE_PER_S_K_L2 = Unit("J") / (Unit("s") * Unit("K") * Unit("L") ** 2)


class EnergyBalanceProcessModel(ProcessModel):
    """Process model for an irreversible reaction with an energy balance."""

    # ---- Differential states ----
    V: Annotated[float, M(type=T.DIFFERENTIAL, unit="L", description="system volume")]=100.0
    A: Annotated[float, M(type=T.DIFFERENTIAL, unit="mol/L", description="concentration of A")]=1.0
    B: Annotated[float, M(type=T.DIFFERENTIAL, unit="mol/L", description="concentration of B")]=0.0
    T: Annotated[float, M(type=T.DIFFERENTIAL, unit="K", description="reactor temperature")]=300.0
    T_J: Annotated[float, M(type=T.DIFFERENTIAL, unit="K", description="jacket temperature")]=300.0

    # ---- Algebraic states ----
    F_out: Annotated[float, M(type=T.ALGEBRAIC, unit="L/s", description="outlet volumetric flow")]=1.0

    # ---- Controlled states ----
    F_in: Annotated[float, M(type=T.CONTROLLED, unit="L/s", description="inlet volumetric flow")]=1.0
    T_J_in: Annotated[float, M(type=T.CONTROLLED, unit="K", description="jacket inlet temperature")]=300.0

    # ---- Constants ----
    k0: Annotated[float, M(type=T.CONSTANT, unit="1/s", description="pre-exponential factor")]=1.5e9
    activation_energy: Annotated[float, M(type=T.CONSTANT, unit="J/mol", description="activation energy")]=72500.0
    gas_constant: Annotated[float, M(type=T.CONSTANT, unit=JOULE_PER_MOL_K, description="ideal gas constant")]=8.314
    Cv: Annotated[float, M(type=T.CONSTANT, unit=SQRT_L_PER_S, description="outlet valve coefficient")]=2.0
    CA_in: Annotated[float, M(type=T.CONSTANT, unit="mol/L", description="inlet concentration of A")]=2.0
    T_in: Annotated[float, M(type=T.CONSTANT, unit="K", description="feed temperature")]=300.0
    reaction_enthalpy: Annotated[float, M(type=T.CONSTANT, unit="J/mol", description="reaction enthalpy")]=825000.0
    rho_cp: Annotated[float, M(type=T.CONSTANT, unit=JOULE_PER_LITER_K, description="heat capacity of reactor contents")]=4000.0
    overall_heat_transfer_coefficient: Annotated[
        float,
        M(type=T.CONSTANT, unit=JOULE_PER_S_K_L2, description="overall heat transfer coefficient"),
    ]=500000.0
    heat_transfer_area: Annotated[float, M(type=T.CONSTANT, unit="L**2", description="heat transfer area")]=10.0
    jacket_volume: Annotated[float, M(type=T.CONSTANT, unit="L", description="jacket volume")]=500000.0
    jacket_rho_cp: Annotated[float, M(type=T.CONSTANT, unit=JOULE_PER_LITER_K, description="jacket heat capacity")]=3200.0
    jacket_flow: Annotated[float, M(type=T.CONSTANT, unit="L/s", description="jacket volumetric flow")]=500.0

    @staticmethod
    def calculate_algebraic_values(
        y: NDArray,
        u: NDArray,
        k: NDArray,
        y_map: Mapping[str, ArrayIndex],
        u_map: Mapping[str, ArrayIndex],
        k_map: Mapping[str, ArrayIndex],
        algebraic_map: Mapping[str, ArrayIndex],
        algebraic_size: int,
    ) -> NDArray:
        """Compute the algebraic variables for the current state."""
        result = np.zeros(algebraic_size, dtype=float)
        volume = max(1e-6, float(y[y_map["V"]]))
        Cv = float(k[k_map["Cv"]])
        result[algebraic_map["F_out"]] = Cv * np.sqrt(volume)
        return result

    @staticmethod
    def differential_rhs(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        algebraic: NDArray,
        y_map: Mapping[str, ArrayIndex],
        u_map: Mapping[str, ArrayIndex],
        k_map: Mapping[str, ArrayIndex],
        algebraic_map: Mapping[str, ArrayIndex],
    ) -> NDArray:
        """Return the time derivatives for the differential states."""

        F_out = float(algebraic[algebraic_map["F_out"]])
        F_in = float(u[u_map["F_in"]])
        F_J_in = float(k[k_map["jacket_flow"]])

        k0 = float(k[k_map["k0"]])
        activation_energy = float(k[k_map["activation_energy"]])
        gas_constant = max(1e-12, float(k[k_map["gas_constant"]]))
        CA_in = float(k[k_map["CA_in"]])
        feed_temperature = float(k[k_map["T_in"]])
        reaction_enthalpy = float(k[k_map["reaction_enthalpy"]])
        rho_cp = max(1e-9, float(k[k_map["rho_cp"]]))
        overall_heat_transfer_coefficient = float(k[k_map["overall_heat_transfer_coefficient"]])
        heat_transfer_area = float(k[k_map["heat_transfer_area"]])
        jacket_volume = max(1e-9, float(k[k_map["jacket_volume"]]))
        jacket_rho_cp = max(1e-9, float(k[k_map["jacket_rho_cp"]]))
        jacket_inlet_temperature = float(u[u_map["T_J_in"]])

        UA = overall_heat_transfer_coefficient * heat_transfer_area

        volume = max(1e-6, float(y[y_map["V"]]))
        molarity_A = float(y[y_map["A"]])
        molarity_B = float(y[y_map["B"]])
        reactor_temperature = float(y[y_map["T"]])
        jacket_temperature = float(y[y_map["T_J"]])

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
            (F_J_in / jacket_volume) * (jacket_inlet_temperature - jacket_temperature)
            + UA * (reactor_temperature - jacket_temperature) / (jacket_rho_cp * jacket_volume)
        )

        return dy
