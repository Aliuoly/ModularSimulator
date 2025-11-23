import numpy as np
from typing import Annotated, override, cast
from collections.abc import Mapping
from numpy.typing import NDArray
from pydantic import Field
from modular_simulation.measurables import ProcessModel, StateType as T, StateMetadata as M
from modular_simulation.utils.typing import ArrayIndex
from functools import partial

R_GAS_KPA_L_PER_KMOL_K = 8.31446261815324e3
ATM_PRESSURE_KPA = 101.325
GRAVITY_M_PER_S2 = 9.81
SMALL = 1.0e-12

DIFFERENTIAL = partial(M, type=T.DIFFERENTIAL)
ALGEBRAIC = partial(M, type=T.ALGEBRAIC)
CONTROLLED = partial(M, type=T.CONTROLLED)
CONSTANT = partial(M, type=T.CONSTANT)


class GasPhaseReactorProcessModel(ProcessModel):
    n_monomer: Annotated[float, DIFFERENTIAL(unit="mol", description="mols of monomer")] = 800 * 0.4
    n_comonomer: Annotated[float, DIFFERENTIAL(unit="mol", description="mols of comonomer")] = (
        800 * 0.13
    )
    n_total: Annotated[float, DIFFERENTIAL(unit="mol", description="total mols of gas")] = 800.0
    effective_cat: Annotated[
        float,
        DIFFERENTIAL(
            unit="kg/h", description="effective catalyst feed rate for production rate calculation"
        ),
    ] = 6.0

    F_cat: Annotated[float, CONTROLLED(unit="kg/h", description="catalyst feed rate")] = 6.0
    F_m1: Annotated[float, CONTROLLED(unit="kg/h", description="monomer feed rate")] = 45000.0
    F_m2: Annotated[float, CONTROLLED(unit="kg/h", description="comonomer feed rate")] = 5000.0

    monomer_rates: Annotated[
        NDArray[np.float64],
        ALGEBRAIC(unit="kmol/s", description="consumption rate of monomer and comonomer"),
    ] = Field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    mass_prod_rate: Annotated[
        float, ALGEBRAIC(unit="kg/hr", description="mass production rate")
    ] = 0.0
    yM1: Annotated[float, ALGEBRAIC(unit="", description="mole fraction of monomer in gas")] = 0.0
    yM2: Annotated[float, ALGEBRAIC(unit="", description="mole fraction of comonomer in gas")] = 0.0
    pressure: Annotated[float, ALGEBRAIC(unit="kPa", description="total gas pressure")] = 0.0

    monomer_mw: Annotated[
        float, CONSTANT(unit="kg/kmol", description="molar weight of monomer")
    ] = 28.0
    comonomer_mw: Annotated[
        float, CONSTANT(unit="kg/kmol", description="molar weight of comonomer")
    ] = 56.0
    cat_productivity: Annotated[
        float, CONSTANT(unit="1/hour", description="kinda like a rate constant")
    ] = 9000.0
    cat_time_constant: Annotated[
        float,
        CONSTANT(
            unit="hour",
            description="time constant for calculating effective cat for rate calculation",
        ),
    ] = 0.5
    volume: Annotated[float, CONSTANT(unit="L", description="reaction volume")] = 900.0e3
    temperature: Annotated[float, CONSTANT(unit="K", description="reaction temperature")] = (
        85.0 + 273.0
    )
    reactivity_ratio_1: Annotated[
        float, CONSTANT(unit="", description="see 'copolymer equation'")
    ] = 8.0
    reactivity_ratio_2: Annotated[
        float, CONSTANT(unit="", description="see 'copolymer equation'")
    ] = 0.01

    @override
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

        n_monomer = y[y_map["n_monomer"]]
        n_comonomer = y[y_map["n_comonomer"]]
        n_total = y[y_map["n_total"]]
        eff_cat = y[y_map["effective_cat"]]
        volume = k[k_map["volume"]]
        temperature = k[k_map["temperature"]]
        monomer_mw = k[k_map["monomer_mw"]]
        comonomer_mw = k[k_map["comonomer_mw"]]
        cat_prod = k[k_map["cat_productivity"]]
        r1 = k[k_map["reactivity_ratio_1"]]
        r2 = k[k_map["reactivity_ratio_2"]]

        pressure = n_total * R_GAS_KPA_L_PER_KMOL_K * temperature / volume
        y_monomer = n_monomer / n_total
        y_comonomer = n_comonomer / n_total
        temp = (1.5 * y_monomer + 0.2 * y_comonomer) * pressure
        mass_prod_rate = cat_prod * eff_cat * temp / (2200 * 0.53)  # kg/hr

        x = y_comonomer / (SMALL + y_monomer)
        f1 = (r1 + x) / (r1 + 2 * x + r2 * x**2)
        pm1 = f1 * monomer_mw + (1 - f1) * comonomer_mw
        molar_prod_rate = mass_prod_rate / pm1  # kmol/hr
        monomer_rates = (
            np.array([f1 * molar_prod_rate, (1 - f1) * molar_prod_rate], dtype=np.float64) / 3600.0
        )

        result[algebraic_map["monomer_rates"]] = monomer_rates
        result[algebraic_map["mass_prod_rate"]] = mass_prod_rate
        result[algebraic_map["yM1"]] = n_monomer / n_total
        result[algebraic_map["yM2"]] = n_comonomer / n_total
        result[algebraic_map["pressure"]] = pressure
        return result

    @override
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
        monomer_mw = k[k_map["monomer_mw"]]
        comonomer_mw = k[k_map["comonomer_mw"]]
        eff_cat = y[y_map["effective_cat"]]

        monomer_rates = algebraic[algebraic_map["monomer_rates"]]
        cat_time_constant = k[k_map["cat_time_constant"]] * 3600  # convert to seconds

        F_cat = u[u_map["F_cat"]]  # keep kg/hr
        F_m1 = u[u_map["F_m1"]] / monomer_mw / 3600.0  # from kg/h to kmol/s
        F_m2 = u[u_map["F_m2"]] / comonomer_mw / 3600.0  # from kg/h to kmol/s

        deff_cat = 1 / cat_time_constant * (F_cat - eff_cat)
        dn_monomer = cast(np.float64, F_m1 - monomer_rates[0])
        dn_comonomer = cast(np.float64, F_m2 - monomer_rates[1])
        dn_total = dn_monomer + dn_comonomer

        dy[y_map["effective_cat"]] = deff_cat
        dy[y_map["n_monomer"]] = dn_monomer
        dy[y_map["n_comonomer"]] = dn_comonomer
        dy[y_map["n_total"]] = dn_total
        return dy
