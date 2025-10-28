from modular_simulation.usables import (
    CalculationBase, 
    TagMetadata,
    TagType, 
)
import numpy as np
from typing import Any, Annotated
from astropy.units import Unit
SMALL = 1e-12

class MoleRatioCalculation(CalculationBase):
    """
    Calculates the mole ratio of comonomer and hydrogen to monomer.
    All are technically unitless unless I want to define
    units like "mol h2 / mol m1", which I don't need to and don't want to.
    """
    rM2_tag: Annotated[str, TagMetadata(TagType.OUTPUT, Unit())]

    yM1_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit())]
    yM2_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit())]

    def _calculation_algorithm(self, t: float, inputs_dict: dict[str, Any]) -> dict[str, float]: 
        yM1 = inputs_dict[self.yM1_tag]
        yM2 = inputs_dict[self.yM2_tag]
        ratios = np.array([yM2]) / max(SMALL, yM1)
        return {
            self.rM2_tag: ratios[0]
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
    
    
