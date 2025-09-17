from typing import Dict, List, TYPE_CHECKING
from pydantic import  PrivateAttr, BaseModel, ConfigDict

from modular_simulation.usables import Calculation, Sensor, Measurement
if TYPE_CHECKING:
    from modular_simulation.quantities import MeasurableQuantities

UsableResults = Dict[str, Measurement]
class UsableQuantities(BaseModel):
    """
    1. Defines how measurements and calculations are obtained through
        - measurement_definition: Dict[str, Sensor]
        - calculation_definition: Dict[str, Calculation]
    2. Saves the current snapshot of measurements and calculations
        - results: Dict[str, Any]
    """
    measurement_definitions: Dict[str, Sensor]
    calculation_definitions: Dict[str, Calculation]

    _tag_list: List[str] = PrivateAttr(default_factory=list)
    _usable_results: UsableResults = PrivateAttr(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # validation handled in the validation module
    
    def update(
            self, 
            measurable_quantities: "MeasurableQuantities",
            t: float
            ) -> UsableResults:
        
        for tag, sensor in self.measurement_definitions.items():
            self._usable_results[tag] = sensor.measure(measurable_quantities, t)
        for tag, calculation in self.calculation_definitions.items():
            self._usable_results[tag] = calculation.calculate(self._usable_results)

        return self._usable_results

