from modular_simulation.usables import Calculation
import numpy as np
from typing import Any, Dict
from scipy.integrate import odeint
SMALL = 1e-12

class MoleRatioCalculation(Calculation):
    """
    Calculates the mole ratio of comonomer and hydrogen to monomer. \n
    Output tags MUST be ordered as follows \n
        1. 'mole ratio of comonomer:monomer'
        2. 'mole ratio of hydrogen:monomer'
    Inputs are:
        1. mole fraction of monomer
        2. mole fraction of comonomer
        3. mole fraction of hydrogen
    """

    def _calculation_algorithm(self, t: float, inputs_dict: Dict[str, Any]): 
        yM1 = inputs_dict["yM1"]
        yM2 = inputs_dict["yM2"]
        yH2 = inputs_dict["yH2"]
        return np.array([yM2, yH2]) / max(SMALL, yM1)
    
class Monomer1PartialPressure(Calculation):
    """
    Calculates the mole ratio of comonomer and hydrogen to monomer. \n
    Output tags MUST be ordered as follows \n
        1. 'partial pressure of monomer 1'
    Inputs are:
        1. mole fraction of monomer
        2. total pressure - could be a filtered version from another calculation or raw
    """

    def _calculation_algorithm(self, t: float, inputs_dict: Dict[str, Any]): 
        yM1 = inputs_dict["yM1"]
        pressure = inputs_dict["pressure"]
        return yM1 * pressure

class ResidenceTimeCalculation(Calculation):
    """
    Calculates the residence time of the polymer resin in the reactor. \n
    Output tag MUST have size 1 and refer to 'residence time' \n
    Input tag MUST have the following order \n
        1. 'production rate of resin in ton/hr'
        2. 'bed weight of reactor in ton'
    """
    def _calculation_algorithm(self, t, inputs_dict):
        return inputs_dict["bed_weight"] / max(SMALL, inputs_dict["mass_prod_rate"])


class CatInventoryEstimator(Calculation):
    inventory: float = 0.0
    t: float = 0.0
        
    def _calculation_algorithm(self, t, inputs_dict: Dict[str, Any]):
        pr = inputs_dict["mass_prod_rate"] / 3600. # convert to ton/s
        bw = inputs_dict["bed_weight"] # in ton
        F_cat = inputs_dict["F_cat"] # in g/s
        
        dt = t - self.t
        self.inventory = CatInventoryEstimator.integrate_inventory(self.inventory, F_cat, pr, bw, dt)
        self.t = t

        return self.inventory
    
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
        pr = self._last_input_value_dict["mass_prod_rate"] / 3600 # ton/s
        bw = self._last_input_value_dict["bed_weight"]
        
        dt = 600. # hardcoded 10 minutes from now for now. 
        return dt * (F_cat - self.inventory / bw * pr) + self.inventory


class AlTiRatioEstimator(Calculation):
    cat_Ti_weight_frac: float = 0.0065
    mwTi: float = 47.8
    def _calculation_algorithm(self, t:float, inputs_dict: Dict[str, Any]):
        F_cat = inputs_dict["F_cat"]
        Fteal = inputs_dict["F_teal"] # TODO this is assumed molar flow right now
        FTi = F_cat*self.cat_Ti_weight_frac/self.mwTi
        return Fteal / max(SMALL, FTi)
    
    def AlTi_model(self, Fteal):
        F_cat = self._last_input_value_dict["F_cat"]
        FTi = F_cat*self.cat_Ti_weight_frac/self.mwTi
        return Fteal / max(1e-20, FTi)