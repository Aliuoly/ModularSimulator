"""Gas-phase polymerization reactor written for the framework interface."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from modular_simulation.framework import System
from modular_simulation.measurables import AlgebraicStates, Constants, ControlElements, States

R_GAS_KPA_L_PER_MOL_K = 8.31446261815324
ATM_PRESSURE_KPA = 101.325
GRAVITY_M_PER_S2 = 9.81
SMALL = 1.0e-12


class GasPhaseReactorStates(States):
    """Differential states for the polymerization reactor."""

    c_impurity: float
    c_teal: float
    c_monomer: NDArray[np.float64]
    c_hydrogen: float
    c_nitrogen: float
    Nstar: NDArray[np.float64]
    N0: NDArray[np.float64]
    NdIH0: NDArray[np.float64]
    NdI0: NDArray[np.float64]
    NH0: NDArray[np.float64]
    Y0: NDArray[np.float64]
    Y1: NDArray[np.float64]
    Y2: NDArray[np.float64]
    X0: NDArray[np.float64]
    X1: NDArray[np.float64]
    X2: NDArray[np.float64]
    B: NDArray[np.float64]
    V_poly: float
    V_gas: float


class GasPhaseReactorControlElements(ControlElements):
    """Externally manipulated feed and valve positions."""

    F_cat: float
    F_m1: float
    F_m2: float
    F_h2: float
    F_n2: float
    F_vent: float
    F_teal: float
    F_impurity: float
    discharge_valve_open: float


class GasPhaseReactorAlgebraicStates(AlgebraicStates):
    """Derived quantities evaluated from the current state."""

    impurity_rate: float = 0.0
    teal_rate: float = 0.0
    pressure: float = 0.0
    bed_weight: float = 0.0
    bed_level: float = 0.0
    hydrogen_rate: float = 0.0
    monomer_rates: NDArray[np.float64] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    vol_prod_rate: float = 0.0
    production_rate: float = 0.0
    residence_time: float = 0.0
    cumm_Fbar2: float = 0.0
    cumm_mMWbar: float = 0.0
    cumm_pMWbar: float = 0.0
    cumm_MI: float = 0.0
    cumm_density: float = 0.0
    discharge_rate: float = 0.0


class GasPhaseReactorConstants(Constants):
    """Physical and kinetic parameters for the reactor."""

    cat_sites_per_gram: float
    cat_type1_site_fraction: float
    discharge_efficiency: float
    discharge_valve_constant: float
    cross_section_area: float
    temperature: float
    monomer_mw: NDArray[np.float64]
    kf: NDArray[np.float64]
    ka: NDArray[np.float64]
    kds: NDArray[np.float64]
    kdI: NDArray[np.float64]
    kti: float
    khr: NDArray[np.float64]
    ki1: NDArray[np.float64]
    ki2: NDArray[np.float64]
    kh1: NDArray[np.float64]
    kh2: NDArray[np.float64]
    kp11: NDArray[np.float64]
    kp12: NDArray[np.float64]
    kp21: NDArray[np.float64]
    kp22: NDArray[np.float64]
    kfm11: NDArray[np.float64]
    kfm12: NDArray[np.float64]
    kfm21: NDArray[np.float64]
    kfm22: NDArray[np.float64]
    kfh1: NDArray[np.float64]
    kfh2: NDArray[np.float64]
    kfr1: NDArray[np.float64]
    kfr2: NDArray[np.float64]
    kfs1: NDArray[np.float64]
    kfs2: NDArray[np.float64]
    min_production_rate_for_qssa: float = 5.0


class GasPhaseReactorSystem(System):
    """System definition for the gas-phase polymerization reactor."""

    @staticmethod
    def calculate_algebraic_values(
        y: NDArray[np.float64],
        u: NDArray[np.float64],
        k: NDArray[np.float64],
        y_map: Mapping[str, slice],
        u_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
        algebraic_size: int,
    ) -> NDArray[np.float64]:
        result = np.zeros(algebraic_size, dtype=np.float64)

        c_impurity = float(y[y_map["c_impurity"]][0])
        c_teal = float(y[y_map["c_teal"]][0])
        c_monomer = np.asarray(y[y_map["c_monomer"]], dtype=np.float64)
        c_hydrogen = float(y[y_map["c_hydrogen"]][0])
        c_nitrogen = float(y[y_map["c_nitrogen"]][0])
        Nstar = np.asarray(y[y_map["Nstar"]], dtype=np.float64)
        N0 = np.asarray(y[y_map["N0"]], dtype=np.float64)
        NdIH0 = np.asarray(y[y_map["NdIH0"]], dtype=np.float64)
        NdI0 = np.asarray(y[y_map["NdI0"]], dtype=np.float64)
        NH0 = np.asarray(y[y_map["NH0"]], dtype=np.float64)
        Y0 = np.asarray(y[y_map["Y0"]], dtype=np.float64)
        Y1 = np.asarray(y[y_map["Y1"]], dtype=np.float64)
        Y2 = np.asarray(y[y_map["Y2"]], dtype=np.float64)
        X0 = np.asarray(y[y_map["X0"]], dtype=np.float64)
        X1 = np.asarray(y[y_map["X1"]], dtype=np.float64)
        X2 = np.asarray(y[y_map["X2"]], dtype=np.float64)
        B = np.asarray(y[y_map["B"]], dtype=np.float64)
        V_poly = max(SMALL, float(y[y_map["V_poly"]][0]))

        temperature = float(k[k_map["temperature"]][0])
        cross_section_area = float(k[k_map["cross_section_area"]][0])
        discharge_valve_constant = float(k[k_map["discharge_valve_constant"]][0])
        monomer_mw = np.asarray(k[k_map["monomer_mw"]], dtype=np.float64)
        kti = float(k[k_map["kti"]][0])

        kf = np.asarray(k[k_map["kf"]], dtype=np.float64)
        ka = np.asarray(k[k_map["ka"]], dtype=np.float64)
        kds = np.asarray(k[k_map["kds"]], dtype=np.float64)
        kdI = np.asarray(k[k_map["kdI"]], dtype=np.float64)
        khr = np.asarray(k[k_map["khr"]], dtype=np.float64)
        ki1 = np.asarray(k[k_map["ki1"]], dtype=np.float64)
        ki2 = np.asarray(k[k_map["ki2"]], dtype=np.float64)
        kh1 = np.asarray(k[k_map["kh1"]], dtype=np.float64)
        kh2 = np.asarray(k[k_map["kh2"]], dtype=np.float64)
        kp11 = np.asarray(k[k_map["kp11"]], dtype=np.float64)
        kp12 = np.asarray(k[k_map["kp12"]], dtype=np.float64)
        kp21 = np.asarray(k[k_map["kp21"]], dtype=np.float64)
        kp22 = np.asarray(k[k_map["kp22"]], dtype=np.float64)
        kfm11 = np.asarray(k[k_map["kfm11"]], dtype=np.float64)
        kfm12 = np.asarray(k[k_map["kfm12"]], dtype=np.float64)
        kfm21 = np.asarray(k[k_map["kfm21"]], dtype=np.float64)
        kfm22 = np.asarray(k[k_map["kfm22"]], dtype=np.float64)
        kfh1 = np.asarray(k[k_map["kfh1"]], dtype=np.float64)
        kfh2 = np.asarray(k[k_map["kfh2"]], dtype=np.float64)
        kfr1 = np.asarray(k[k_map["kfr1"]], dtype=np.float64)
        kfr2 = np.asarray(k[k_map["kfr2"]], dtype=np.float64)
        kfs1 = np.asarray(k[k_map["kfs1"]], dtype=np.float64)
        kfs2 = np.asarray(k[k_map["kfs2"]], dtype=np.float64)

        C1 = c_monomer[0]
        C2 = c_monomer[1]
        total_monomer = max(SMALL, C1 + C2)
        f1 = C1 / total_monomer
        f2 = 1.0 - f1

        phi_den = kp12 * C2 + kp21 * C1 + SMALL
        phi_1 = kp21 * C1 / phi_den
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

        r1 = float(np.sum(C1 * Y0 * kpT1))
        r2 = float(np.sum(C2 * Y0 * kpT2))
        monomer_rates = np.array([r1, r2], dtype=np.float64)

        cumm_Fbar2 = B[1] / max(SMALL, float(np.sum(B)))
        cumm_mMWbar = monomer_mw[0] * (1.0 - cumm_Fbar2) + monomer_mw[1] * cumm_Fbar2
        denominator = max(SMALL, float(np.sum(X1 + Y1)))
        cumm_pMWbar = cumm_mMWbar * float(np.sum(X2 + Y2)) / denominator
        cumm_MI = (cumm_pMWbar / 111525.0) ** (1.0 / -0.288)
        cumm_density = 1000.0 * (0.966 - 0.20 * max(SMALL, cumm_Fbar2) ** 0.40)

        inst_Fbar2 = r1 / max(SMALL, r1 + r2)
        rho_p = 1000.0 * (0.966 - 0.20 * max(SMALL, inst_Fbar2) ** 0.40)
        volumetric_prod_rate = (monomer_mw[0] * r1 + monomer_mw[1] * r2) / max(SMALL, rho_p)
        mass_prod_rate = volumetric_prod_rate * rho_p / 1e6 * 3600.0

        hydrogen_rate = float(np.sum(Y0 * c_hydrogen * kfhT))
        residence_time = V_poly / max(SMALL, volumetric_prod_rate)

        pressure = (
            (c_nitrogen + c_hydrogen + C1 + C2 + c_teal + c_impurity)
            * R_GAS_KPA_L_PER_MOL_K
            * temperature
        )
        bed_weight = V_poly * cumm_density * 1e-6
        bed_level = V_poly / 1000.0 / max(SMALL, cross_section_area)

        site_count = float(kdI.size)
        impurity_rate = float(
            c_impurity
            * (
                site_count * kti * c_teal
                + np.sum(kdI * (Y0 + N0 + NH0))
                - np.sum(ka * (NdI0 + NdIH0))
            )
        )
        teal_rate = float(
            c_teal
            * (
                site_count * kti * c_impurity
                + np.sum(kf * Nstar)
                + np.sum(khr * NH0)
                + np.sum(kfrT * Y0)
            )
        )

        discharge_pressure = pressure + cumm_density * bed_level * GRAVITY_M_PER_S2 / 1000.0
        delta_p = discharge_pressure - ATM_PRESSURE_KPA
        discharge_rate = discharge_valve_constant * delta_p ** 2

        result[algebraic_map["impurity_rate"]] = impurity_rate
        result[algebraic_map["teal_rate"]] = teal_rate
        result[algebraic_map["pressure"]] = pressure
        result[algebraic_map["bed_weight"]] = bed_weight
        result[algebraic_map["bed_level"]] = bed_level
        result[algebraic_map["hydrogen_rate"]] = hydrogen_rate
        result[algebraic_map["monomer_rates"]] = monomer_rates
        result[algebraic_map["vol_prod_rate"]] = volumetric_prod_rate
        result[algebraic_map["production_rate"]] = mass_prod_rate
        result[algebraic_map["residence_time"]] = residence_time
        result[algebraic_map["cumm_Fbar2"]] = cumm_Fbar2
        result[algebraic_map["cumm_mMWbar"]] = cumm_mMWbar
        result[algebraic_map["cumm_pMWbar"]] = cumm_pMWbar
        result[algebraic_map["cumm_MI"]] = cumm_MI
        result[algebraic_map["cumm_density"]] = cumm_density
        result[algebraic_map["discharge_rate"]] = discharge_rate
        return result

    @staticmethod
    def rhs(
        t: float,
        y: NDArray[np.float64],
        u: NDArray[np.float64],
        k: NDArray[np.float64],
        algebraic: NDArray[np.float64],
        y_map: Mapping[str, slice],
        u_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
    ) -> NDArray[np.float64]:
        dy = np.zeros_like(y)

        c_impurity = float(y[y_map["c_impurity"]][0])
        c_teal = float(y[y_map["c_teal"]][0])
        c_monomer = np.asarray(y[y_map["c_monomer"]], dtype=np.float64)
        c_hydrogen = float(y[y_map["c_hydrogen"]][0])
        c_nitrogen = float(y[y_map["c_nitrogen"]][0])
        Nstar = np.asarray(y[y_map["Nstar"]], dtype=np.float64)
        N0 = np.asarray(y[y_map["N0"]], dtype=np.float64)
        NdIH0 = np.asarray(y[y_map["NdIH0"]], dtype=np.float64)
        NdI0 = np.asarray(y[y_map["NdI0"]], dtype=np.float64)
        NH0 = np.asarray(y[y_map["NH0"]], dtype=np.float64)
        Y0 = np.asarray(y[y_map["Y0"]], dtype=np.float64)
        Y1 = np.asarray(y[y_map["Y1"]], dtype=np.float64)
        Y2 = np.asarray(y[y_map["Y2"]], dtype=np.float64)
        X0 = np.asarray(y[y_map["X0"]], dtype=np.float64)
        X1 = np.asarray(y[y_map["X1"]], dtype=np.float64)
        X2 = np.asarray(y[y_map["X2"]], dtype=np.float64)
        B = np.asarray(y[y_map["B"]], dtype=np.float64)
        V_poly = float(y[y_map["V_poly"]][0])
        V_gas = float(y[y_map["V_gas"]][0])

        V_poly_pos = max(SMALL, V_poly)
        V_gas_pos = max(SMALL, V_gas)

        cat_sites_per_gram = float(k[k_map["cat_sites_per_gram"]][0])
        cat_type1_fraction = float(k[k_map["cat_type1_site_fraction"]][0])
        discharge_efficiency = float(k[k_map["discharge_efficiency"]][0])
        kti = float(k[k_map["kti"]][0])

        kf = np.asarray(k[k_map["kf"]], dtype=np.float64)
        ka = np.asarray(k[k_map["ka"]], dtype=np.float64)
        kds = np.asarray(k[k_map["kds"]], dtype=np.float64)
        kdI = np.asarray(k[k_map["kdI"]], dtype=np.float64)
        khr = np.asarray(k[k_map["khr"]], dtype=np.float64)
        ki1 = np.asarray(k[k_map["ki1"]], dtype=np.float64)
        ki2 = np.asarray(k[k_map["ki2"]], dtype=np.float64)
        kh1 = np.asarray(k[k_map["kh1"]], dtype=np.float64)
        kh2 = np.asarray(k[k_map["kh2"]], dtype=np.float64)
        kp11 = np.asarray(k[k_map["kp11"]], dtype=np.float64)
        kp12 = np.asarray(k[k_map["kp12"]], dtype=np.float64)
        kp21 = np.asarray(k[k_map["kp21"]], dtype=np.float64)
        kp22 = np.asarray(k[k_map["kp22"]], dtype=np.float64)
        kfm11 = np.asarray(k[k_map["kfm11"]], dtype=np.float64)
        kfm12 = np.asarray(k[k_map["kfm12"]], dtype=np.float64)
        kfm21 = np.asarray(k[k_map["kfm21"]], dtype=np.float64)
        kfm22 = np.asarray(k[k_map["kfm22"]], dtype=np.float64)
        kfh1 = np.asarray(k[k_map["kfh1"]], dtype=np.float64)
        kfh2 = np.asarray(k[k_map["kfh2"]], dtype=np.float64)
        kfr1 = np.asarray(k[k_map["kfr1"]], dtype=np.float64)
        kfr2 = np.asarray(k[k_map["kfr2"]], dtype=np.float64)
        kfs1 = np.asarray(k[k_map["kfs1"]], dtype=np.float64)
        kfs2 = np.asarray(k[k_map["kfs2"]], dtype=np.float64)

        monomer_rates = np.asarray(algebraic[algebraic_map["monomer_rates"]], dtype=np.float64)
        hydrogen_rate = float(algebraic[algebraic_map["hydrogen_rate"]][0])
        vol_prod_rate = float(algebraic[algebraic_map["vol_prod_rate"]][0])
        production_rate = float(algebraic[algebraic_map["production_rate"]][0])
        discharge_rate_nominal = float(algebraic[algebraic_map["discharge_rate"]][0])

        F_cat = float(u[u_map["F_cat"]][0])
        F_m1 = float(u[u_map["F_m1"]][0])
        F_m2 = float(u[u_map["F_m2"]][0])
        F_h2 = float(u[u_map["F_h2"]][0])
        F_n2 = float(u[u_map["F_n2"]][0])
        F_vent = float(u[u_map["F_vent"]][0])
        F_teal = float(u[u_map["F_teal"]][0])
        F_impurity = float(u[u_map["F_impurity"]][0])
        discharge_valve_open = float(u[u_map["discharge_valve_open"]][0])

        C1 = c_monomer[0]
        C2 = c_monomer[1]
        total_monomer = max(SMALL, C1 + C2)
        f1 = C1 / total_monomer
        f2 = 1.0 - f1
        phi_den = kp12 * C2 + kp21 * C1 + SMALL
        phi_1 = kp21 * C1 / phi_den
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

        cat_fraction = cat_type1_fraction
        cat_conversion = cat_sites_per_gram * np.array([cat_fraction, 1.0 - cat_fraction])
        F_cat_sites = F_cat * cat_conversion

        F_poly = discharge_valve_open * discharge_rate_nominal
        dVpoly = vol_prod_rate - F_poly
        dVgas = -dVpoly
        total_gas_loss = F_poly / discharge_efficiency - F_poly + F_vent

        kdI_sum = np.sum(kdI * (Y0 + N0 + NH0))
        ka_sum = np.sum(ka * (NdI0 + NdIH0))
        tempA = kdI_sum - ka_sum + total_gas_loss + dVgas
        tempB = (
            np.sum(kf * Nstar)
            + np.sum(khr * NH0)
            + np.sum(kfrT * Y0)
            + F_poly
            + dVpoly
        )

        dC_teal = (F_teal - tempB * c_teal) / V_poly_pos
        dC_impurity = (F_impurity - tempA * c_impurity) / V_gas_pos

        min_production_rate_for_qssa = float(k[k_map["min_production_rate_for_qssa"]][0])
        if production_rate > min_production_rate_for_qssa:
            a = tempA * kti
            b = tempA * tempB + kti * F_teal - kti * F_impurity
            c_term = -F_impurity * tempB
            if abs(a) > SMALL:
                discriminant = max(0.0, b * b - 4.0 * a * c_term)
                root = (-b + np.sqrt(discriminant)) / (2.0 * a)
            elif abs(b) > SMALL:
                root = -c_term / b
            else:
                root = c_impurity
            c_impurity = max(1.0e-32, float(root))
            dC_impurity = 0.0

        C_MT = C1 + C2
        dC_m1 = (
            F_m1
            - total_gas_loss * C1
            - monomer_rates[0]
            - dVgas * C1
        ) / V_gas_pos
        dC_m2 = (
            F_m2
            - total_gas_loss * C2
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
        dy[y_map["c_monomer"]] = np.array([dC_m1, dC_m2], dtype=np.float64)
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
