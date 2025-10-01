from modular_simulation.usables import (
    Calculation, 
    Constant, 
    CalculatedTag, 
    MeasuredTag, 
    OutputTag
)
import numpy as np
from typing import Any, Dict
from scipy.integrate import odeint
from pydantic import PrivateAttr, Field
SMALL = 1e-12

class MoleRatioCalculation(Calculation):
    """
    Calculates the mole ratio of comonomer and hydrogen to monomer.
    """
    rM2_tag: OutputTag = Field(...)
    rH2_tag: OutputTag = Field(...)

    yM1_tag: MeasuredTag = Field(...)
    yM2_tag: MeasuredTag = Field(...)
    yH2_tag: MeasuredTag = Field(...)

    def _calculation_algorithm(self, t: float, inputs_dict: Dict[str, Any]) -> Dict[str, float]: 
        yM1 = inputs_dict[self.yM1_tag]
        yM2 = inputs_dict[self.yM2_tag]
        yH2 = inputs_dict[self.yH2_tag]
        ratios = np.array([yM2, yH2]) / max(SMALL, yM1)
        return {
            self.rM2_tag: ratios[0],
            self.rH2_tag: ratios[1]
        }
    
class Monomer1PartialPressure(Calculation):
    """
    Calculates the mole ratio of comonomer and hydrogen to monomer. \n
    Output tags MUST be ordered as follows \n
        1. 'partial pressure of monomer 1'
    Inputs are:
        1. mole fraction of monomer
        2. total pressure - could be a filtered version from another calculation or raw
    """
    pM1_tag: OutputTag

    pressure_tag: MeasuredTag
    yM1_tag: MeasuredTag

    def _calculation_algorithm(self, t: float, inputs_dict: Dict[str, Any]): 
        yM1 = inputs_dict[self.yM1_tag]
        pressure = inputs_dict[self.pressure_tag]
        return {self.pM1_tag: yM1 * pressure}

class ResidenceTimeCalculation(Calculation):
    """
    Calculates the residence time of the polymer resin in the reactor. \n
    Output tag MUST have size 1 and refer to 'residence time' \n
    Input tag MUST have the following order \n
        1. 'production rate of resin in ton/hr'
        2. 'bed weight of reactor in ton'
    """
    residence_time_tag: OutputTag

    bed_weight_tag: MeasuredTag
    mass_prod_rate_tag: MeasuredTag

    def _calculation_algorithm(self, t, inputs_dict):
        bw = inputs_dict[self.bed_weight_tag]
        pr = inputs_dict[self.mass_prod_rate_tag]
        return {self.residence_time_tag: bw / max(SMALL, pr)}


class CatInventoryEstimator(Calculation):
    
    cat_inventory_tag: OutputTag

    mass_prod_rate_tag: MeasuredTag
    bed_weight_tag: MeasuredTag
    F_cat_tag: MeasuredTag
    
    _inventory: float = PrivateAttr(default = 0.0)
    _t: float = PrivateAttr(default = 0.0)

    def _calculation_algorithm(self, t, inputs_dict: Dict[str, Any]):
        pr = inputs_dict[self.mass_prod_rate_tag] / 3600. # convert to ton/s
        bw = inputs_dict[self.bed_weight_tag] # in ton
        F_cat = inputs_dict[self.F_cat_tag]/3600. * 1000 # in kg/h -> g/s
        
        dt = t - self._t
        self._inventory = CatInventoryEstimator.integrate_inventory(self._inventory, F_cat, pr, bw, dt)
        self._t = t

        return {self.cat_inventory_tag: self._inventory}
    
    @classmethod
    def ode_rhs(cls, t, y, u, pr, bw):
        return u - y/bw*pr

    @classmethod
    def integrate_inventory(cls, y0, fcat, pr, bw, dt):
        sol = odeint(CatInventoryEstimator.ode_rhs, y0 = [y0], t = [0, dt], args = (fcat, pr, bw), tfirst = True)
        return sol.ravel()[-1]
    
    def discrete_model(self, F_cat):
        """discretized, the model is just
            inventory_new = dt * (u - inventory / bw * pr) + inventory"""
        pr = self._last_input_value_dict[self.mass_prod_rate_tag] / 3600 # ton/s
        bw = self._last_input_value_dict[self.bed_weight_tag]
        
        dt = 600. # hardcoded 10 minutes from now for now. 
        return dt * (F_cat - self._inventory / bw * pr) + self._inventory


class AlTiRatioEstimator(Calculation):
    AlTi_ratio_tag: OutputTag

    F_cat_tag: MeasuredTag
    F_teal_tag: MeasuredTag

    cat_Ti_weight_frac: Constant = 0.0065
    mwTi: Constant = 47.8

    def _calculation_algorithm(self, t:float, inputs_dict: Dict[str, Any]):
        F_cat = inputs_dict[self.F_cat_tag] / 3600. * 1000
        Fteal = inputs_dict[self.F_teal_tag] # TODO this is assumed molar flow right now
        FTi = F_cat*self.cat_Ti_weight_frac/self.mwTi
        return {self.AlTi_ratio_tag: Fteal / max(SMALL, FTi)}
    
    def AlTi_model(self, Fteal):
        F_cat = self._last_input_value_dict[self.F_cat_tag] / 3600. * 1000
        FTi = F_cat*self.cat_Ti_weight_frac/self.mwTi
        return Fteal / max(1e-20, FTi)