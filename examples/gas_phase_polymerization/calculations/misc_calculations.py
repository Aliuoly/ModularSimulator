from modular_simulation.interfaces import (
    CalculationBase,
    TagMetadata,
    TagType,
)
import numpy as np
from typing import Any, Annotated
from astropy.units import Unit
from pydantic import PrivateAttr
SMALL = 1e-12

class MoleRatioCalculation(CalculationBase):
    """
    Calculates the mole ratio of comonomer and hydrogen to monomer.
    All are technically unitless unless I want to define
    units like "mol h2 / mol m1", which I don't need to and don't want to.
    """
    rM2_tag: Annotated[str, TagMetadata(TagType.OUTPUT, Unit())]
    rH2_tag: Annotated[str, TagMetadata(TagType.OUTPUT, Unit())]

    yM1_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit())]
    yM2_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit())]
    yH2_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit())]

    def _calculation_algorithm(self, t: float, inputs_dict: dict[str, Any]) -> dict[str, float]: 
        yM1 = inputs_dict[self.yM1_tag]
        yM2 = inputs_dict[self.yM2_tag]
        yH2 = inputs_dict[self.yH2_tag]
        ratios = np.array([yM2, yH2]) / max(SMALL, yM1)
        return {
            self.rM2_tag: ratios[0],
            self.rH2_tag: ratios[1]
        }
    
class Monomer1PartialPressure(CalculationBase):
    """
    Calculates the mole ratio of comonomer and hydrogen to monomer. \n
    Output tags MUST be ordered as follows \n
        1. 'partial pressure of monomer 1'
    Inputs are:
        1. mole fraction of monomer
        2. total pressure - could be a filtered version from another calculation or raw
    """
    pM1_tag: Annotated[str, TagMetadata(TagType.OUTPUT, Unit("kPa"))]

    pressure_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit("kPa"))]
    yM1_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit())]

    def _calculation_algorithm(self, t: float, inputs_dict: dict[str, Any]): 
        yM1 = inputs_dict[self.yM1_tag]
        pressure = inputs_dict[self.pressure_tag]
        return {self.pM1_tag: yM1 * pressure}

class ResidenceTimeCalculation(CalculationBase):
    """
    Calculates the residence time of the polymer resin in the reactor. \n
    Output tag MUST have size 1 and refer to 'residence time' \n
    Input tag MUST have the following order \n
        1. 'production rate of resin in tonne/hr'
        2. 'bed weight of reactor in tonne'
    """
    residence_time_tag: Annotated[str, TagMetadata(TagType.OUTPUT, Unit("hour"))]

    bed_weight_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit("tonne"))]
    mass_prod_rate_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit("tonne/hour"))]

    def _calculation_algorithm(self, t, inputs_dict):
        bw = inputs_dict[self.bed_weight_tag]
        pr = inputs_dict[self.mass_prod_rate_tag]
        return {self.residence_time_tag: bw / max(SMALL, pr)}


class CatInventoryEstimator(CalculationBase):
    
    cat_inventory_tag: Annotated[str, TagMetadata(TagType.OUTPUT, Unit("kg"))]

    mass_prod_rate_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit("tonne/second"))]
    bed_weight_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit("tonne"))]
    F_cat_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit("kg/second"))]
    
    _inventory: float = PrivateAttr(default = 0.0)
    _t: float = PrivateAttr(default = 0.0)

    def _calculation_algorithm(self, t, inputs_dict: dict[str, Any]):
        pr = inputs_dict[self.mass_prod_rate_tag]
        bw = inputs_dict[self.bed_weight_tag]
        F_cat = inputs_dict[self.F_cat_tag]
        
        dt = t - self._t
        self._inventory = self.integrate_inventory(self._inventory, F_cat, pr, bw, dt)
        self._t = t

        return {self.cat_inventory_tag: self._inventory}
    
    @staticmethod
    def ode_rhs(t, y, u, pr, bw):
        return u - y/bw*pr

    @staticmethod
    def integrate_inventory(inventory, fcat, pr, bw, dt):
        # just first degree forward euler
        dinvdt = fcat - inventory/bw*pr
        return inventory + dinvdt * dt
    
    def model(self, F_cat):
        pr = self._last_input_value_dict[self.mass_prod_rate_tag]
        bw = self._last_input_value_dict[self.bed_weight_tag]
        
        dt = 600. # hardcoded 10 minutes -> 600 seconds from now for now. 
        return self.integrate_inventory(self._inventory, F_cat, pr, bw, dt)


class AlTiRatioEstimator(CalculationBase):
    AlTi_ratio_tag: Annotated[str, TagMetadata(TagType.OUTPUT, Unit())]

    F_cat_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit("kg/hour"))]
    F_teal_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit("kmol/hour"))]

    cat_Ti_weight_frac: Annotated[float, TagMetadata(TagType.CONSTANT, Unit())] = 0.0065
    mwTi: Annotated[float, TagMetadata(TagType.CONSTANT, Unit("kmol/kg"))] = 47.8

    def _calculation_algorithm(self, t:float, inputs_dict: dict[str, Any]):
        F_cat = inputs_dict[self.F_cat_tag]
        Fteal = inputs_dict[self.F_teal_tag] # TODO this is assumed molar flow right now
        FTi = F_cat*self.cat_Ti_weight_frac/self.mwTi
        return {self.AlTi_ratio_tag: Fteal / max(SMALL, FTi)}
    
    def AlTi_model(self, Fteal):
        F_cat = self._last_input_value_dict[self.F_cat_tag]
        FTi = F_cat*self.cat_Ti_weight_frac/self.mwTi
        return Fteal / max(1e-20, FTi)