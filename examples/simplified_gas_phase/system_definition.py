"""Gas-phase polymerization reactor written for the framework interface."""

from typing import Annotated, Mapping
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from modular_simulation.framework import System
from modular_simulation.measurables import AlgebraicStates, Constants, ControlElements, States
from astropy.units import Unit

R_GAS_KPA_L_PER_KMOL_K = 8.31446261815324e3
ATM_PRESSURE_KPA = 101.325
GRAVITY_M_PER_S2 = 9.81
SMALL = 1.0e-12

class GasPhaseReactorStates(States):
    """Differential states for the polymerization reactor."""
    n_monomer: Annotated[float, Unit("mol")] = 800 * 0.4
    n_comonomer: Annotated[float, Unit("mol")] = 800 * 0.13
    n_total: Annotated[float, Unit("mol")] = 800.
    effective_cat: Annotated[float, Unit("kg")] = 6.0

class GasPhaseReactorControlElements(ControlElements):
    """Externally manipulated feed and valve positions."""

    F_cat: Annotated[float, Unit("kg/h")] = 6.0
    F_m1: Annotated[float, Unit("kg/h")] = 45000.0
    F_m2: Annotated[float, Unit("kg/h")] = 5000.0

class GasPhaseReactorAlgebraicStates(AlgebraicStates):
    """Derived quantities required by the differential balances."""

    monomer_rates: Annotated[NDArray[np.float64], Unit("kmol/s")] = Field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    mass_prod_rate: Annotated[float, Unit("kg/hr")] = 0.0
    yM1: Annotated[float, Unit("")] = 0.0
    yM2: Annotated[float, Unit("")] = 0.0
    pressure: Annotated[float, Unit("kPa")] = 0.0

class GasPhaseReactorConstants(Constants):
    """Physical and kinetic parameters for the reactor."""
    monomer_mw: Annotated[float, Unit("kg/kmol")] = 28.0
    comonomer_mw: Annotated[float, Unit("kg/kmol")] = 56.0
    cat_productivity: Annotated[float, Unit("1/hour")] = 9000.0
    cat_time_constant: Annotated[float, Unit("hour")] = 0.5
    volume: Annotated[float, Unit("L")] = 900.0e3
    temperature: Annotated[float, Unit("K")] = 85.0 + 273.0
    reactivity_ratio_1: Annotated[float, Unit("")] = 8.0
    reactivity_ratio_2: Annotated[float, Unit("")] = 0.01

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

        n_monomer = y[y_map["n_monomer"]][0]
        n_comonomer = y[y_map["n_comonomer"]][0]
        n_total = y[y_map["n_total"]][0]
        eff_cat = y[y_map["effective_cat"]][0]
        volume = k[k_map["volume"]][0]
        temperature = k[k_map["temperature"]][0]
        monomer_mw = k[k_map["monomer_mw"]][0]
        comonomer_mw = k[k_map["comonomer_mw"]][0]
        cat_prod = k[k_map["cat_productivity"]][0]
        r1 = k[k_map["reactivity_ratio_1"]][0]
        r2 = k[k_map["reactivity_ratio_2"]][0]

        pressure = n_total * R_GAS_KPA_L_PER_KMOL_K * temperature / volume
        y_monomer = n_monomer / n_total
        y_comonomer = n_comonomer / n_total
        temp = (1.5 * y_monomer + 0.2 * y_comonomer) * pressure
        mass_prod_rate = cat_prod * eff_cat * temp / (2200*0.53) # kg/hr

        x = y_comonomer / (SMALL + y_monomer)
        f1 = (r1 + x) / (r1 + 2 * x + r2 * x ** 2)
        pm1 = f1 * monomer_mw + (1-f1) * comonomer_mw
        molar_prod_rate = mass_prod_rate / pm1 # kmol/hr
        monomer_rates = np.array([f1 * molar_prod_rate, (1-f1) * molar_prod_rate], dtype=np.float64) / 3600.


        result[algebraic_map["monomer_rates"]] = monomer_rates
        result[algebraic_map["mass_prod_rate"]] = mass_prod_rate
        result[algebraic_map["yM1"]] = n_monomer / n_total
        result[algebraic_map["yM2"]] = n_comonomer / n_total
        result[algebraic_map["pressure"]] = pressure
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
        eff_cat = y[y_map["effective_cat"]][0]

        monomer_rates = algebraic[algebraic_map["monomer_rates"]]
        cat_time_constant = k[k_map["cat_time_constant"]][0]

        F_cat = u[u_map["F_cat"]][0] # keep kg/hr
        F_m1 = u[u_map["F_m1"]][0] / monomer_mw / 3600. # from kg/h to kmol/s
        F_m2 = u[u_map["F_m2"]][0] / comonomer_mw / 3600.  # from kg/h to kmol/s


        deff_cat = 1/cat_time_constant * (F_cat - eff_cat)
        dn_monomer = F_m1 - monomer_rates[0]
        dn_comonomer = F_m2 - monomer_rates[1]
        dn_total = dn_monomer + dn_comonomer

        dy[y_map["effective_cat"]] = deff_cat
        dy[y_map["n_monomer"]] = dn_monomer
        dy[y_map["n_comonomer"]] = dn_comonomer
        dy[y_map["n_total"]] = dn_total
        return dy