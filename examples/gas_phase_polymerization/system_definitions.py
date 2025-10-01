"""Gas-phase polymerization reactor written for the framework interface."""

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

    c_impurity: float = 0.0
    c_teal: float = 2.22e-4
    c_monomer: float = 0.2354
    c_comonomer: float = 0.08239
    c_hydrogen: float = 0.04707
    c_nitrogen: float = (738794.3 - 1000.0e3*(0.04707 + 0.2354 + 0.08239))/1000.0e3
    Nstar: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    N0: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    NdIH0: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    NdI0: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    NH0: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    Y0: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    Y1: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    Y2: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    X0: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    X1: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    X2: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    B: NDArray[np.float64] = Field(default_factory=lambda : np.array([1e-20, 1e-20]))
    V_poly: float = 200.0e3 # L
    V_gas: float = 1000.0e3 # L




class GasPhaseReactorControlElements(ControlElements):
    """Externally manipulated feed and valve positions."""

    F_cat: float = 0.0
    F_m1: float = 0.0
    F_m2: float = 0.0
    F_h2: float = 0.0
    F_n2: float = 0.0
    F_vent: float = 0.0
    F_teal: float = 0.0
    F_impurity: float = 0.0
    discharge_valve_position: float = 0.0


class GasPhaseReactorAlgebraicStates(AlgebraicStates):
    """Derived quantities required by the differential balances."""

    hydrogen_rate: float = 0.0
    monomer_rates: NDArray[np.float64] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    vol_prod_rate: float = 0.0
    total_monomer_concentration: float = 0.0
    total_gas_loss_rate: float = 0.0
    dV_poly: float = 0.0
    dV_gas: float = 0.0
    impurity_balance_A: float = 0.0
    impurity_balance_B: float = 0.0
    pseudo_kiT: NDArray[np.float64] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_khT: NDArray[np.float64] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_kpTT: NDArray[np.float64] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_kfmTT: NDArray[np.float64] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_kfhT: NDArray[np.float64] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_kfrT: NDArray[np.float64] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pseudo_kfsT: NDArray[np.float64] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    c_impurity_qssa: float = 0.0
    impurity_qssa_applicable: float = 0.0
    cumm_MI: float = 2.0
    cumm_density: float = 918.0
    bed_level: float = 15.0
    bed_weight: float = 100.0
    yM1: float = 0.0
    yM2: float = 0.0
    yH2: float = 0.0
    pressure: float = 0.0
    mass_prod_rate: float = 0.0



class GasPhaseReactorConstants(Constants):
    """Physical and kinetic parameters for the reactor."""

    cat_sites_per_gram: float = 0.02718
    cat_type1_site_fraction: float = 0.55
    discharge_efficiency: float = 0.80
    discharge_valve_constant: float = 5e-6
    cross_section_area: float = 13.0 # m2
    temperature: float = 85.0 + 273.0
    monomer_mw: float = 28.0
    comonomer_mw: float = 56.0
    hydrogen_mw: float = 2.0
    nitrogen_mw: float = 28.0
    kf: NDArray[np.float64] = Field(
        default_factory=lambda: np.array([1.0, 1.0], dtype=np.float64)
        * np.array([3e-3, 3e-3], dtype=np.float64),
        description="Rate of active-site formation (per catalyst site).",
    )
    ka: NDArray[np.float64] = Field(
        default_factory=lambda: np.array([3e-4, 3e-4], dtype=np.float64),
        description="Reactivation of poisoned/impurity-adsorbed sites via desorption.",
    )
    kds: NDArray[np.float64] = Field(
        default_factory=lambda: np.array([1e-5, 1e-5], dtype=np.float64),
        description="Spontaneous active-site deactivation.",
    )
    kdI: NDArray[np.float64] = Field(
        default_factory=lambda: np.array([2000.0, 2000.0], dtype=np.float64),
        description="Active-site deactivation due to poison/impurity adsorption.",
    )
    kti: float = Field(
        default=1.0e2,
        description="TEAL scavenging of poisons (lumped rate constant).",
    )
    khr: NDArray[np.float64] = Field(
        default_factory=lambda: np.array([20.0, 20.0], dtype=np.float64),
        description="Site (re)labeling/activation by TEAL at H₂-labeled active sites.",
    )
    ki1: NDArray[np.float64] = Field(
        default_factory=lambda: np.array([1e-3, 1e-3], dtype=np.float64),
        description="Chain initiation with monomer 1 at an active site.",
    )
    ki2: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.14, 0.14], dtype=np.float64)
                                 * np.array([1e-3, 1e-3], dtype=np.float64)),
        description="Chain initiation with monomer 2 at an active site.",
    )
    kh1: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([1.0, 1.0], dtype=np.float64)
                                 * np.array([6e-3, 6e-3], dtype=np.float64)),
        description="Chain initiation with monomer 1 at an H₂-labeled site.",
    )
    kh2: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.1, 0.1], dtype=np.float64)
                                 * np.array([6e-3, 6e-3], dtype=np.float64)),
        description="Chain initiation with monomer 2 at an H₂-labeled site.",
    )
    kp11: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([85.0, 85.0], dtype=np.float64)
                                 * np.array([2e-1, 2e-1], dtype=np.float64)),
        description="Propagation with monomer 1 at a monomer-1-labeled site.",
    )
    kp12: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([2.0, 15.0], dtype=np.float64)
                                 * np.array([2e-1, 3e-1], dtype=np.float64)),
        description="Propagation with monomer 2 at a monomer-1-labeled site.",
    )
    kp21: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([64.0, 64.0], dtype=np.float64)
                                 * np.array([2e-1, 2e-1], dtype=np.float64)),
        description="Propagation with monomer 1 at a monomer-2-labeled site.",
    )
    kp22: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([1.5, 6.2], dtype=np.float64)
                                 * np.array([3e-1, 3e-1], dtype=np.float64)),
        description="Propagation with monomer 2 at a monomer-2-labeled site.",
    )
    kfm11: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.0021, 0.0021], dtype=np.float64)
                                 * np.array([2e-1, 2e-1], dtype=np.float64)),
        description="Chain transfer to monomer 1 at a monomer-1-labeled site.",
    )
    kfm12: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.006, 0.11], dtype=np.float64)
                                 * np.array([2e-1, 2e-1], dtype=np.float64)),
        description="Chain transfer to monomer 2 at a monomer-1-labeled site.",
    )
    kfm21: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.0021, 0.001], dtype=np.float64)
                                 * np.array([2e-1, 2e-1], dtype=np.float64)),
        description="Chain transfer to monomer 1 at a monomer-2-labeled site.",
    )
    kfm22: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.006, 0.11], dtype=np.float64)
                                 * np.array([2e-1, 2e-1], dtype=np.float64)),
        description="Chain transfer to monomer 2 at a monomer-2-labeled site.",
    )
    kfh1: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.088, 0.37], dtype=np.float64)
                                 * np.array([4.0e-1, 6.9e-1], dtype=np.float64)),
        description="Chain transfer to hydrogen at a monomer-1-labeled site.",
    )
    kfh2: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.088, 0.37], dtype=np.float64)
                                 * np.array([4.2e-1, 6.5e-1], dtype=np.float64)),
        description="Chain transfer to hydrogen at a monomer-2-labeled site.",
    )
    kfr1: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.024, 0.12], dtype=np.float64)
                                 * np.array([1e-1, 1e-1], dtype=np.float64)),
        description="Chain transfer to TEAL at a monomer-1-labeled site.",
    )
    kfr2: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.048, 0.24], dtype=np.float64)
                                 * np.array([1e-1, 1e-1], dtype=np.float64)),
        description="Chain transfer to TEAL at a monomer-2-labeled site.",
    )
    kfs1: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.0001, 0.0001], dtype=np.float64)
                                 * np.array([1e-2, 1e-2], dtype=np.float64)),
        description="Spontaneous chain transfer at a monomer-1-labeled site.",
    )
    kfs2: NDArray[np.float64] = Field(
        default_factory=lambda: (np.array([0.0001, 0.0001], dtype=np.float64)
                                 * np.array([1e-2, 1e-2], dtype=np.float64)),
        description="Spontaneous chain transfer at a monomer-2-labeled site.",
    )
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
        c_monomer = y[y_map["c_monomer"]][0]
        c_comonomer = y[y_map["c_comonomer"]][0]
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
        discharge_efficiency = float(k[k_map["discharge_efficiency"]][0])
        monomer_mw = k[k_map["monomer_mw"]][0]
        comonomer_mw = k[k_map["comonomer_mw"]][0]
        kti = float(k[k_map["kti"]][0])

        kf = np.asarray(k[k_map["kf"]], dtype=np.float64)
        ka = np.asarray(k[k_map["ka"]], dtype=np.float64)
        #kds = np.asarray(k[k_map["kds"]], dtype=np.float64)
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

        F_teal = float(u[u_map["F_teal"]][0])
        F_impurity = float(u[u_map["F_impurity"]][0])
        F_vent = float(u[u_map["F_vent"]][0])
        discharge_valve_position = float(u[u_map["discharge_valve_position"]][0])


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

        r1 = float(np.sum(c_monomer * Y0 * kpT1))
        r2 = float(np.sum(c_comonomer * Y0 * kpT2))
        monomer_rates = np.array([r1, r2], dtype=np.float64)

        cumm_Fbar2 = B[1] / max(SMALL, float(np.sum(B)))
        cumm_mMWbar = monomer_mw*(1-cumm_Fbar2) + comonomer_mw*cumm_Fbar2
        cumm_pMWbar = cumm_mMWbar*(X2 + Y2).sum() / max(SMALL, (X1 + Y1).sum())
        cumm_MI = (cumm_pMWbar/111525.)**(1/-0.288) # correlation from paper
        cumm_density = 1000.0 * (0.966 - 0.20 * max(SMALL, cumm_Fbar2) ** 0.40)

        inst_Fbar2 = r1 / max(SMALL, r1 + r2)
        rho_p = 1000.0 * (0.966 - 0.20 * max(SMALL, inst_Fbar2) ** 0.40)
        volumetric_prod_rate = (monomer_mw * r1 + comonomer_mw * r2) / max(SMALL, rho_p)
        mass_prod_rate = volumetric_prod_rate * rho_p / 1e6 * 3600.0

        hydrogen_rate = float(np.sum(Y0 * c_hydrogen * kfhT))

        pressure = (
            (c_nitrogen + c_hydrogen + c_monomer + c_comonomer + c_teal + c_impurity)
            * R_GAS_KPA_L_PER_MOL_K
            * temperature
        )
        bed_weight = V_poly * cumm_density * 1e-6  # in tons
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
        min_production_rate_for_qssa = float(k[k_map["min_production_rate_for_qssa"]][0])
        if mass_prod_rate > min_production_rate_for_qssa:
            a = tempA * kti
            b = tempA * tempB + kti * F_teal - kti * F_impurity
            c_term = -F_impurity * tempB
            if abs(a) > SMALL:
                discriminant = max(0.0, b * b - 4.0 * a * c_term)
                root = (-b + np.sqrt(discriminant)) / (2.0 * a)
                c_impurity_qssa = max(1.0e-32, float(root))
                impurity_qssa_applicable = 1.0
            elif abs(b) > SMALL:
                root = -c_term / b
                c_impurity_qssa = max(1.0e-32, float(root))
                impurity_qssa_applicable = 1.0

        # y(i)P = n(i)/V * R*T
        # y(i) = n(i)/V * R * T / P
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
        result[algebraic_map["dV_poly"]] = dV_poly  # I know, I know, this is a differential state
        result[algebraic_map["dV_gas"]] = dV_gas    # but it is very helpful to have it here for C_impurity QSSA calculation
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
        monomer_mw = k[k_map["monomer_mw"]][0]
        comonomer_mw = k[k_map["comonomer_mw"]][0]
        hydrogen_mw = k[k_map["hydrogen_mw"]][0]
        nitrogen_mw = k[k_map["nitrogen_mw"]][0]
        c_impurity = float(y[y_map["c_impurity"]][0])
        c_teal = float(y[y_map["c_teal"]][0])
        c_monomer = y[y_map["c_monomer"]][0]
        c_comonomer = y[y_map["c_comonomer"]][0]
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
        #kti = float(k[k_map["kti"]][0])

        kf = np.asarray(k[k_map["kf"]], dtype=np.float64)
        ka = np.asarray(k[k_map["ka"]], dtype=np.float64)
        kds = np.asarray(k[k_map["kds"]], dtype=np.float64)
        kdI = np.asarray(k[k_map["kdI"]], dtype=np.float64)
        khr = np.asarray(k[k_map["khr"]], dtype=np.float64)

        monomer_rates = np.asarray(algebraic[algebraic_map["monomer_rates"]], dtype=np.float64)
        hydrogen_rate = float(algebraic[algebraic_map["hydrogen_rate"]][0])
        vol_prod_rate = float(algebraic[algebraic_map["vol_prod_rate"]][0])
        total_gas_loss = float(algebraic[algebraic_map["total_gas_loss_rate"]][0])
        dVpoly = float(algebraic[algebraic_map["dV_poly"]][0])
        dVgas = float(algebraic[algebraic_map["dV_gas"]][0])
        tempA = float(algebraic[algebraic_map["impurity_balance_A"]][0])
        tempB = float(algebraic[algebraic_map["impurity_balance_B"]][0])
        total_monomer = float(algebraic[algebraic_map["total_monomer_concentration"]][0])
        kiT = np.asarray(algebraic[algebraic_map["pseudo_kiT"]], dtype=np.float64)
        khT = np.asarray(algebraic[algebraic_map["pseudo_khT"]], dtype=np.float64)
        kpTT = np.asarray(algebraic[algebraic_map["pseudo_kpTT"]], dtype=np.float64)
        kfmTT = np.asarray(algebraic[algebraic_map["pseudo_kfmTT"]], dtype=np.float64)
        kfhT = np.asarray(algebraic[algebraic_map["pseudo_kfhT"]], dtype=np.float64)
        kfrT = np.asarray(algebraic[algebraic_map["pseudo_kfrT"]], dtype=np.float64)
        kfsT = np.asarray(algebraic[algebraic_map["pseudo_kfsT"]], dtype=np.float64)

        c_impurity_qssa = float(algebraic[algebraic_map["c_impurity_qssa"]][0])
        impurity_qssa_applicable = float(algebraic[algebraic_map["impurity_qssa_applicable"]][0])

        F_cat = float(u[u_map["F_cat"]][0]) / 3600.
        F_m1 = float(u[u_map["F_m1"]][0]) / monomer_mw / 3600.
        F_m2 = float(u[u_map["F_m2"]][0]) / comonomer_mw / 3600.
        F_h2 = float(u[u_map["F_h2"]][0]) / hydrogen_mw / 3600.
        F_n2 = float(u[u_map["F_n2"]][0]) / nitrogen_mw / 3600.
        F_teal = float(u[u_map["F_teal"]][0]) #TODO this is assumed molar flow already
        F_impurity = float(u[u_map["F_impurity"]][0]) # this IS molar flow.. 

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