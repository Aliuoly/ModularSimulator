"""Gas-phase polymerization reactor expressed using the new ProcessModel API."""

from __future__ import annotations

import numpy as np
from typing import Annotated
from collections.abc import Mapping
from numpy.typing import NDArray
from pydantic import Field

from modular_simulation.measurables import ProcessModel, StateType as T, StateMetadata as M
from modular_simulation.utils.typing import ArrayIndex

R_GAS_KPA_L_PER_MOL_K = 8.31446261815324
ATM_PRESSURE_KPA = 101.325
GRAVITY_M_PER_S2 = 9.81
SMALL = 1.0e-12


class GasPhaseReactorProcessModel(ProcessModel):
    """Detailed gas-phase polymerization reactor with a reduced-order kinetic model."""

    # ---- Differential states -------------------------------------------------
    c_impurity: Annotated[float, M(type=T.DIFFERENTIAL, unit="mol/L")] = 0.0
    c_teal: Annotated[float, M(type=T.DIFFERENTIAL, unit="mol/L")] = 2.22e-4
    c_monomer: Annotated[float, M(type=T.DIFFERENTIAL, unit="mol/L")] = 0.2354
    c_comonomer: Annotated[float, M(type=T.DIFFERENTIAL, unit="mol/L")] = 0.08239
    c_hydrogen: Annotated[float, M(type=T.DIFFERENTIAL, unit="mol/L")] = 0.04707
    c_nitrogen: Annotated[float, M(type=T.DIFFERENTIAL, unit="mol/L")] = (
        738794.3 - 1000.0e3 * (0.04707 + 0.2354 + 0.08239)
    ) / 1000.0e3
    Nstar: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    N0: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    NdIH0: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    NdI0: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    NH0: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    Y0: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    Y1: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    Y2: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    X0: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    X1: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    X2: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    B: Annotated[
        NDArray[np.float64],
        M(type=T.DIFFERENTIAL, unit="")
    ] = Field(default_factory=lambda: np.array([1e-20, 1e-20], dtype=np.float64))
    V_poly: Annotated[float, M(type=T.DIFFERENTIAL, unit="L")] = 108.0e3
    V_gas: Annotated[float, M(type=T.DIFFERENTIAL, unit="L")] = 1000.0e3

    # ---- Controlled states ---------------------------------------------------
    F_cat: Annotated[float, M(type=T.CONTROLLED, unit="g/s")] = 1.0
    F_m1: Annotated[float, M(type=T.CONTROLLED, unit="g/s")] = 100.0
    F_m2: Annotated[float, M(type=T.CONTROLLED, unit="g/s")] = 100.0
    F_h2: Annotated[float, M(type=T.CONTROLLED, unit="g/s")] = 10.0
    F_n2: Annotated[float, M(type=T.CONTROLLED, unit="g/s")] = 10.0
    F_vent: Annotated[float, M(type=T.CONTROLLED, unit="L/s")] = 0.0
    F_teal: Annotated[float, M(type=T.CONTROLLED, unit="mol/s")] = 10.0
    F_impurity: Annotated[float, M(type=T.CONTROLLED, unit="g/s")] = 0.0
    discharge_valve_position: Annotated[float, M(type=T.CONTROLLED, unit="")] = 0.0

    # ---- Algebraic states ----------------------------------------------------
    hydrogen_rate: Annotated[float, M(type=T.ALGEBRAIC, unit="mol/s")] = 0.0
    monomer_rates: Annotated[
        NDArray[np.float64],
        M(type=T.ALGEBRAIC, unit="mol/s")
    ] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    vol_prod_rate: Annotated[float, M(type=T.ALGEBRAIC, unit="L/s")] = 0.0
    total_monomer_concentration: Annotated[float, M(type=T.ALGEBRAIC, unit="mol/L")] = 0.0
    total_gas_loss_rate: Annotated[float, M(type=T.ALGEBRAIC, unit="L/s")] = 0.0
    dV_poly: Annotated[float, M(type=T.ALGEBRAIC, unit="L/s")] = 0.0
    dV_gas: Annotated[float, M(type=T.ALGEBRAIC, unit="L/s")] = 0.0
    impurity_balance_A: Annotated[float, M(type=T.ALGEBRAIC, unit="1/s")] = 0.0
    impurity_balance_B: Annotated[float, M(type=T.ALGEBRAIC, unit="1/s")] = 0.0
    pseudo_kiT: Annotated[
        NDArray[np.float64],
        M(type=T.ALGEBRAIC, unit="1/s")
    ] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_khT: Annotated[
        NDArray[np.float64],
        M(type=T.ALGEBRAIC, unit="1/s")
    ] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_kpTT: Annotated[
        NDArray[np.float64],
        M(type=T.ALGEBRAIC, unit="1/s")
    ] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_kfmTT: Annotated[
        NDArray[np.float64],
        M(type=T.ALGEBRAIC, unit="1/s")
    ] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_kfhT: Annotated[
        NDArray[np.float64],
        M(type=T.ALGEBRAIC, unit="1/s")
    ] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_kfrT: Annotated[
        NDArray[np.float64],
        M(type=T.ALGEBRAIC, unit="1/s")
    ] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_kfsT: Annotated[
        NDArray[np.float64],
        M(type=T.ALGEBRAIC, unit="1/s")
    ] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    c_impurity_qssa: Annotated[float, M(type=T.ALGEBRAIC, unit="mol/L")] = 0.0
    impurity_qssa_applicable: Annotated[float, M(type=T.ALGEBRAIC, unit="")] = 0.0
    cumm_MI: Annotated[float, M(type=T.ALGEBRAIC, unit="")] = 2.0
    cumm_density: Annotated[float, M(type=T.ALGEBRAIC, unit="g/L")] = 918.0
    bed_level: Annotated[float, M(type=T.ALGEBRAIC, unit="m")] = 15.0
    bed_weight: Annotated[float, M(type=T.ALGEBRAIC, unit="tonne")] = 100.0
    yM1: Annotated[float, M(type=T.ALGEBRAIC, unit="")] = 0.0
    yM2: Annotated[float, M(type=T.ALGEBRAIC, unit="")] = 0.0
    yH2: Annotated[float, M(type=T.ALGEBRAIC, unit="")] = 0.0
    pressure: Annotated[float, M(type=T.ALGEBRAIC, unit="kPa")] = 0.0
    mass_prod_rate: Annotated[float, M(type=T.ALGEBRAIC, unit="tonne/h")] = 0.0
    # ---- Constants -----------------------------------------------------------
    cat_sites_per_gram: Annotated[float, M(type=T.CONSTANT, unit="1/g")] = 0.05
    cat_type1_site_fraction: Annotated[float, M(type=T.CONSTANT, unit="")] = 0.55
    discharge_efficiency: Annotated[float, M(type=T.CONSTANT, unit="")] = 0.80
    discharge_valve_constant: Annotated[float, M(type=T.CONSTANT, unit="L/(s*kPa**2)")] = 1e-5
    cross_section_area: Annotated[float, M(type=T.CONSTANT, unit="m2")] = 7.5
    temperature: Annotated[float, M(type=T.CONSTANT, unit="K")] = 85.0 + 273.0
    monomer_mw: Annotated[float, M(type=T.CONSTANT, unit="g/mol")] = 28.0
    comonomer_mw: Annotated[float, M(type=T.CONSTANT, unit="g/mol")] = 56.0
    hydrogen_mw: Annotated[float, M(type=T.CONSTANT, unit="g/mol")] = 2.0
    nitrogen_mw: Annotated[float, M(type=T.CONSTANT, unit="g/mol")] = 28.0
    kf: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/s"),
    ] = Field(default_factory=lambda: np.array([3.0e-3, 3.0e-3], dtype=np.float64))
    ka: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/s"),
    ] = Field(default_factory=lambda: np.array([3.0e-4, 3.0e-4], dtype=np.float64))
    kds: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/s"),
    ] = Field(default_factory=lambda: np.array([1.0e-5, 1.0e-5], dtype=np.float64))
    kdI: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/s"),
    ] = Field(default_factory=lambda: np.array([2000.0, 2000.0], dtype=np.float64))
    kti: Annotated[float, M(type=T.CONSTANT, unit="1/s")] = 1.0e2
    khr: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/s"),
    ] = Field(default_factory=lambda: np.array([20.0, 20.0], dtype=np.float64))
    ki1: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([1.0e-3, 1.0e-3], dtype=np.float64))
    ki2: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([1.4e-4, 1.4e-4], dtype=np.float64))
    kh1: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([6.0e-3, 6.0e-3], dtype=np.float64))
    kh2: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([6.0e-4, 6.0e-4], dtype=np.float64))
    kp11: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([17.0, 17.0], dtype=np.float64))
    kp12: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([0.4, 4.5], dtype=np.float64))
    kp21: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([12.8, 12.8], dtype=np.float64))
    kp22: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([0.45, 1.86], dtype=np.float64))
    kfm11: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([4.2e-4, 4.2e-4], dtype=np.float64))
    kfm12: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([1.2e-3, 2.2e-2], dtype=np.float64))
    kfm21: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([4.2e-4, 2.0e-4], dtype=np.float64))
    kfm22: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([1.2e-3, 2.2e-2], dtype=np.float64))
    kfh1: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([3.52e-2, 2.553e-1], dtype=np.float64))
    kfh2: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([3.696e-2, 2.405e-1], dtype=np.float64))
    kfr1: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([2.4e-3, 1.2e-2], dtype=np.float64))
    kfr2: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/(mol*s)"),
    ] = Field(default_factory=lambda: np.array([4.8e-3, 2.4e-2], dtype=np.float64))
    kfs1: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/s"),
    ] = Field(default_factory=lambda: np.array([1.0e-6, 1.0e-6], dtype=np.float64))
    kfs2: Annotated[
        NDArray[np.float64],
        M(type=T.CONSTANT, unit="1/s"),
    ] = Field(default_factory=lambda: np.array([1.0e-6, 1.0e-6], dtype=np.float64))
    min_production_rate_for_qssa: Annotated[float, M(type=T.CONSTANT, unit="kg/h")] = 5.0

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
        result = np.zeros(algebraic_size, dtype=np.float64)

        c_impurity = float(y[y_map["c_impurity"]])
        c_teal = float(y[y_map["c_teal"]])
        c_monomer = float(y[y_map["c_monomer"]])
        c_comonomer = float(y[y_map["c_comonomer"]])
        c_hydrogen = float(y[y_map["c_hydrogen"]])
        c_nitrogen = float(y[y_map["c_nitrogen"]])
        Nstar = y[y_map["Nstar"]]
        N0 = y[y_map["N0"]]
        NdIH0 = y[y_map["NdIH0"]]
        NdI0 = y[y_map["NdI0"]]
        NH0 = y[y_map["NH0"]]
        Y0 = y[y_map["Y0"]]
        Y1 = y[y_map["Y1"]]
        Y2 = y[y_map["Y2"]]
        X1 = y[y_map["X1"]]
        X2 = y[y_map["X2"]]
        B = y[y_map["B"]]
        V_poly = max(SMALL, float(y[y_map["V_poly"]]))

        temperature = float(k[k_map["temperature"]])
        cross_section_area = float(k[k_map["cross_section_area"]])
        discharge_valve_constant = float(k[k_map["discharge_valve_constant"]])
        discharge_efficiency = float(k[k_map["discharge_efficiency"]])
        monomer_mw = float(k[k_map["monomer_mw"]])
        comonomer_mw = float(k[k_map["comonomer_mw"]])
        kti = float(k[k_map["kti"]])

        kf = k[k_map["kf"]]
        ka = k[k_map["ka"]]
        kdI = k[k_map["kdI"]]
        khr = k[k_map["khr"]]
        ki1 = k[k_map["ki1"]]
        ki2 = k[k_map["ki2"]]
        kh1 = k[k_map["kh1"]]
        kh2 = k[k_map["kh2"]]
        kp11 = k[k_map["kp11"]]
        kp12 = k[k_map["kp12"]]
        kp21 = k[k_map["kp21"]]
        kp22 = k[k_map["kp22"]]
        kfm11 = k[k_map["kfm11"]]
        kfm12 = k[k_map["kfm12"]]
        kfm21 = k[k_map["kfm21"]]
        kfm22 = k[k_map["kfm22"]]
        kfh1 = k[k_map["kfh1"]]
        kfh2 = k[k_map["kfh2"]]
        kfr1 = k[k_map["kfr1"]]
        kfr2 = k[k_map["kfr2"]]
        kfs1 = k[k_map["kfs1"]]
        kfs2 = k[k_map["kfs2"]]

        F_teal = float(u[u_map["F_teal"]])
        F_impurity = float(u[u_map["F_impurity"]])
        F_vent = float(u[u_map["F_vent"]])
        discharge_valve_position = float(u[u_map["discharge_valve_position"]])

        total_monomer = max(SMALL, c_monomer + c_comonomer)
        f1 = c_monomer / total_monomer
        f2 = 1.0 - f1

        phi_den = kp12 * c_comonomer + kp21 * c_monomer + SMALL
        phi_1 = kp21 * c_monomer / phi_den
        phi_2 = 1.0 - phi_1

        kiT = f1 * ki1 + f2 * ki2
        khT = f1 * kh1 + f2 * kh2
        kpT1 = phi_1 * kp11 + phi_2 * kp21
        kpT2 = phi_1 * kp12 + phi_2 * kp22
        kpTT = f1 * kpT1 + f2 * kpT2
        kfmT1 = phi_1 * kfm11 + phi_2 * kfm21
        kfmT2 = phi_1 * kfm12 + phi_2 * kfm22
        kfmTT = f1 * kfmT1 + f2 * kfmT2
        kfhT = phi_1 * kfh1 + phi_2 * kfh2
        kfrT = phi_1 * kfr1 + phi_2 * kfr2
        kfsT = phi_1 * kfs1 + phi_2 * kfs2

        r1 = np.sum(c_monomer * Y0 * kpT1)
        r2 = np.sum(c_comonomer * Y0 * kpT2)
        monomer_rates = np.array([r1, r2], dtype=np.float64)

        cumm_Fbar2 = B[1] / max(SMALL, np.sum(B))
        cumm_mMWbar = monomer_mw * (1 - cumm_Fbar2) + comonomer_mw * cumm_Fbar2
        cumm_pMWbar = cumm_mMWbar * (X2 + Y2).sum() / max(SMALL, (X1 + Y1).sum())
        cumm_MI = (cumm_pMWbar / 111525.0) ** (1 / -0.288)
        cumm_density = 1000.0 * (0.966 - 0.20 * max(SMALL, cumm_Fbar2) ** 0.40)

        inst_Fbar2 = r1 / max(SMALL, r1 + r2)
        rho_p = 1000.0 * (0.966 - 0.20 * max(SMALL, inst_Fbar2) ** 0.40)
        volumetric_prod_rate = (monomer_mw * r1 + comonomer_mw * r2) / max(SMALL, rho_p)
        mass_prod_rate = volumetric_prod_rate * rho_p / 1e6 * 3600.0

        hydrogen_rate = np.sum(Y0 * c_hydrogen * kfhT)

        pressure = (
            (c_nitrogen + c_hydrogen + c_monomer + c_comonomer + c_teal + c_impurity)
            * R_GAS_KPA_L_PER_MOL_K
            * temperature
        )
        bed_weight = V_poly * cumm_density * 1e-6
        bed_level = V_poly / 1000.0 / max(SMALL, cross_section_area)

        discharge_pressure = pressure + cumm_density * bed_level * GRAVITY_M_PER_S2 / 1000.0
        delta_p = discharge_pressure - ATM_PRESSURE_KPA
        discharge_rate = discharge_valve_constant * delta_p ** 2
        polymer_discharge_rate = discharge_valve_position * discharge_rate
        dV_poly = volumetric_prod_rate - polymer_discharge_rate
        dV_gas = -dV_poly
        total_gas_loss = polymer_discharge_rate / discharge_efficiency - polymer_discharge_rate + F_vent

        kdI_sum = np.sum(kdI * (Y0 + N0 + NH0))
        ka_sum = np.sum(ka * (NdI0 + NdIH0))
        tempA = kdI_sum - ka_sum + total_gas_loss + dV_gas
        tempB = (
            np.sum(kf * Nstar)
            + np.sum(khr * NH0)
            + np.sum(kfrT * Y0)
            + polymer_discharge_rate
            + dV_poly
        )

        c_impurity_qssa = c_impurity
        impurity_qssa_applicable = 0.0
        min_prod_rate = float(k[k_map["min_production_rate_for_qssa"]])
        if mass_prod_rate > min_prod_rate:
            a = tempA * kti
            b = tempA * tempB + kti * F_teal - kti * F_impurity
            c_term = -F_impurity * tempB
            if abs(a) > SMALL:
                discriminant = max(0.0, b * b - 4.0 * a * c_term)
                root = (-b + np.sqrt(discriminant)) / (2.0 * a)
                c_impurity_qssa = max(1.0e-32, root)
                impurity_qssa_applicable = 1.0
            elif abs(b) > SMALL:
                root = -c_term / b
                c_impurity_qssa = max(1.0e-32, root)
                impurity_qssa_applicable = 1.0

        common_term = R_GAS_KPA_L_PER_MOL_K * temperature / pressure
        yM1 = c_monomer * common_term
        yM2 = c_comonomer * common_term
        yH2 = c_hydrogen * common_term

        result[algebraic_map["hydrogen_rate"]] = hydrogen_rate
        result[algebraic_map["monomer_rates"]] = monomer_rates
        result[algebraic_map["vol_prod_rate"]] = volumetric_prod_rate
        result[algebraic_map["mass_prod_rate"]] = mass_prod_rate
        result[algebraic_map["total_monomer_concentration"]] = total_monomer
        result[algebraic_map["total_gas_loss_rate"]] = total_gas_loss
        result[algebraic_map["dV_poly"]] = dV_poly
        result[algebraic_map["dV_gas"]] = dV_gas
        result[algebraic_map["impurity_balance_A"]] = tempA
        result[algebraic_map["impurity_balance_B"]] = tempB
        result[algebraic_map["pseudo_kiT"]] = kiT
        result[algebraic_map["pseudo_khT"]] = khT
        result[algebraic_map["pseudo_kpTT"]] = kpTT
        result[algebraic_map["pseudo_kfmTT"]] = kfmTT
        result[algebraic_map["pseudo_kfhT"]] = kfhT
        result[algebraic_map["pseudo_kfrT"]] = kfrT
        result[algebraic_map["pseudo_kfsT"]] = kfsT
        result[algebraic_map["c_impurity_qssa"]] = c_impurity_qssa
        result[algebraic_map["impurity_qssa_applicable"]] = impurity_qssa_applicable
        result[algebraic_map["cumm_MI"]] = cumm_MI
        result[algebraic_map["cumm_density"]] = cumm_density
        result[algebraic_map["bed_level"]] = bed_level
        result[algebraic_map["bed_weight"]] = bed_weight
        result[algebraic_map["yM1"]] = yM1
        result[algebraic_map["yM2"]] = yM2
        result[algebraic_map["pressure"]] = pressure
        result[algebraic_map["yH2"]] = yH2
        return result


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
        dy = np.zeros_like(y)
        monomer_mw = float(k[k_map["monomer_mw"]])
        comonomer_mw = float(k[k_map["comonomer_mw"]])
        hydrogen_mw = float(k[k_map["hydrogen_mw"]])
        nitrogen_mw = float(k[k_map["nitrogen_mw"]])
        c_impurity = float(y[y_map["c_impurity"]])
        c_teal = float(y[y_map["c_teal"]])
        c_monomer = float(y[y_map["c_monomer"]])
        c_comonomer = float(y[y_map["c_comonomer"]])
        c_hydrogen = float(y[y_map["c_hydrogen"]])
        c_nitrogen = float(y[y_map["c_nitrogen"]])
        Nstar = y[y_map["Nstar"]]
        N0 = y[y_map["N0"]]
        NdIH0 = y[y_map["NdIH0"]]
        NdI0 = y[y_map["NdI0"]]
        NH0 = y[y_map["NH0"]]
        Y0 = y[y_map["Y0"]]
        Y1 = y[y_map["Y1"]]
        Y2 = y[y_map["Y2"]]
        X0 = y[y_map["X0"]]
        X1 = y[y_map["X1"]]
        X2 = y[y_map["X2"]]
        B = y[y_map["B"]]
        V_poly = float(y[y_map["V_poly"]])
        V_gas = float(y[y_map["V_gas"]])

        V_poly_pos = max(SMALL, V_poly)
        V_gas_pos = max(SMALL, V_gas)

        cat_sites_per_gram = float(k[k_map["cat_sites_per_gram"]])
        cat_type1_fraction = float(k[k_map["cat_type1_site_fraction"]])
        #kti = k[k_map["kti"]])

        kf = k[k_map["kf"]]
        ka = k[k_map["ka"]]
        kds = k[k_map["kds"]]
        kdI = k[k_map["kdI"]]
        khr = k[k_map["khr"]]

        monomer_rates = algebraic[algebraic_map["monomer_rates"]]
        hydrogen_rate = float(algebraic[algebraic_map["hydrogen_rate"]])
        vol_prod_rate = float(algebraic[algebraic_map["vol_prod_rate"]])
        total_gas_loss = float(algebraic[algebraic_map["total_gas_loss_rate"]])
        dVpoly = float(algebraic[algebraic_map["dV_poly"]])
        dVgas = float(algebraic[algebraic_map["dV_gas"]])
        tempA = float(algebraic[algebraic_map["impurity_balance_A"]])
        tempB = float(algebraic[algebraic_map["impurity_balance_B"]])
        total_monomer = float(algebraic[algebraic_map["total_monomer_concentration"]])
        kiT = algebraic[algebraic_map["pseudo_kiT"]]
        khT = algebraic[algebraic_map["pseudo_khT"]]
        kpTT = algebraic[algebraic_map["pseudo_kpTT"]]
        kfmTT = algebraic[algebraic_map["pseudo_kfmTT"]]
        kfhT = algebraic[algebraic_map["pseudo_kfhT"]]
        kfrT = algebraic[algebraic_map["pseudo_kfrT"]]
        kfsT = algebraic[algebraic_map["pseudo_kfsT"]]

        c_impurity_qssa = float(algebraic[algebraic_map["c_impurity_qssa"]])
        impurity_qssa_applicable = float(algebraic[algebraic_map["impurity_qssa_applicable"]])

        F_cat = float(u[u_map["F_cat"]])
        F_m1 = float(u[u_map["F_m1"]]) / monomer_mw
        F_m2 = float(u[u_map["F_m2"]]) / comonomer_mw
        F_h2 = float(u[u_map["F_h2"]]) / hydrogen_mw
        F_n2 = float(u[u_map["F_n2"]]) / nitrogen_mw
        F_teal = float(u[u_map["F_teal"]])
        F_impurity = float(u[u_map["F_impurity"]]) 

        C1 = c_monomer
        C2 = c_comonomer

        cat_fraction = cat_type1_fraction
        cat_conversion = cat_sites_per_gram * np.array([cat_fraction, 1.0 - cat_fraction])
        F_cat_sites = F_cat * cat_conversion

        dC_teal = (F_teal - tempB * c_teal) / V_poly_pos
        dC_impurity = (F_impurity - tempA * c_impurity) / V_gas_pos

        if impurity_qssa_applicable > 0.5:
            c_impurity = c_impurity_qssa
            dC_impurity = 0.0

        C_MT = total_monomer
        dC_m1 = (
            F_m1
            - total_gas_loss * c_monomer
            - monomer_rates[0]
            - dVgas * C1
        ) / V_gas_pos
        dC_m2 = (
            F_m2
            - total_gas_loss * c_comonomer
            - monomer_rates[1]
            - dVgas * C2
        ) / V_gas_pos
        dC_h2 = (
            F_h2
            - total_gas_loss * c_hydrogen
            - hydrogen_rate
            - dVgas * c_hydrogen
        ) / V_gas_pos
        dC_n2 = (
            F_n2
            - total_gas_loss * c_nitrogen
            - dVgas * c_nitrogen
        ) / V_gas_pos

        dNstar = F_cat_sites - kf * Nstar - Nstar * vol_prod_rate / V_poly_pos
        dN0 = (
            kf * Nstar
            + ka * NdI0
            - N0
            * (
                kiT * C_MT
                + kds
                + kdI * c_impurity
                + vol_prod_rate / V_poly_pos
            )
        )
        dNdIH0 = c_impurity * NH0 * kdI - ka * NdIH0 - NdIH0 * vol_prod_rate / V_poly_pos
        dNdI0 = c_impurity * N0 * kdI - ka * NdI0 - NdI0 * vol_prod_rate / V_poly_pos
        dNH0 = (
            Y0 * (kfhT * c_hydrogen + kfsT)
            + ka * NdIH0
            - NH0
            * (
                khT * C_MT
                + kds
                + khr * c_teal
                + kdI * c_impurity
                + vol_prod_rate / V_poly_pos
            )
        )
        dY0 = (
            C_MT * (kiT * N0 + khT * NH0)
            + khr * NH0 * c_teal
            - Y0
            * (
                kfhT * c_hydrogen
                + kfsT
                + kds
                + kdI * c_impurity
                + vol_prod_rate / V_poly_pos
            )
        )
        dY1 = (
            C_MT * (kiT * N0 + khT * NH0)
            + khr * NH0 * c_teal
            + C_MT * kpTT * Y0
            + (Y0 - Y1) * (kfmTT * C_MT + kfrT * c_teal)
            - Y1
            * (
                kfhT * c_hydrogen
                + kfsT
                + kds
                + kdI * c_impurity
                + vol_prod_rate / V_poly_pos
            )
        )
        dY2 = (
            C_MT * (kiT * N0 + khT * NH0)
            + khr * NH0 * c_teal
            + C_MT * kpTT * (2.0 * Y1 - Y0)
            + (Y0 - Y2) * (kfmTT * C_MT + kfrT * c_teal)
            - Y2
            * (
                kfhT * c_hydrogen
                + kfsT
                + kds
                + kdI * c_impurity
                + vol_prod_rate / V_poly_pos
            )
        )
        dX0 = (
            Y0
            * (
                kfmTT * C_MT
                + kfrT * c_teal
                + kfhT * c_hydrogen
                + kfsT
                + kds
                + kdI * c_impurity
            )
            - X0 * vol_prod_rate / V_poly_pos
        )
        dX1 = (
            Y1
            * (
                kfmTT * C_MT
                + kfrT * c_teal
                + kfhT * c_hydrogen
                + kfsT
                + kds
                + kdI * c_impurity
            )
            - X1 * vol_prod_rate / V_poly_pos
        )
        dX2 = (
            Y2
            * (
                kfmTT * C_MT
                + kfrT * c_teal
                + kfhT * c_hydrogen
                + kfsT
                + kds
                + kdI * c_impurity
            )
            - X2 * vol_prod_rate / V_poly_pos
        )
        dB = monomer_rates - B * vol_prod_rate / V_poly_pos

        dy[y_map["c_impurity"]] = dC_impurity
        dy[y_map["c_teal"]] = dC_teal
        dy[y_map["c_monomer"]] = dC_m1
        dy[y_map["c_comonomer"]] = dC_m2
        dy[y_map["c_hydrogen"]] = dC_h2
        dy[y_map["c_nitrogen"]] = dC_n2
        dy[y_map["Nstar"]] = dNstar
        dy[y_map["N0"]] = dN0
        dy[y_map["NdIH0"]] = dNdIH0
        dy[y_map["NdI0"]] = dNdI0
        dy[y_map["NH0"]] = dNH0
        dy[y_map["Y0"]] = dY0
        dy[y_map["Y1"]] = dY1
        dy[y_map["Y2"]] = dY2
        dy[y_map["X0"]] = dX0
        dy[y_map["X1"]] = dX1
        dy[y_map["X2"]] = dX2
        dy[y_map["B"]] = dB
        dy[y_map["V_poly"]] = dVpoly
        dy[y_map["V_gas"]] = dVgas
        return dy
