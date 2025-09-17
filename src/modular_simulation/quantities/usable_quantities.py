from typing import Dict, List, TYPE_CHECKING
from pydantic import  PrivateAttr

if TYPE_CHECKING:
    from modular_simulation.usables import Calculation, Sensor, Measurement
    from modular_simulation.quantities import MeasurableQuantities

UsableResults = Dict[str, "Measurement"]
class UsableQuantities:
    """
    1. Defines how measurements and calculations are obtained through
        - measurement_definition: Dict[str, Sensor]
        - calculation_definition: Dict[str, Calculation]
    2. Saves the current snapshot of measurements and calculations
        - results: Dict[str, Any]
    """
    measurement_definitions: Dict[str, "Sensor"]
    calculation_definitions: Dict[str, "Calculation"]
    measurable_quantities: "MeasurableQuantities"  # Explicit dependency

    _tag_list: List[str] = PrivateAttr(default_factory=list)
    
    #model_config = dict(arbitrary_types_allowed=True)

    def __init__(self,
                 measurement_definitions: Dict[str, "Sensor"],
                 calculation_definitions: Dict[str, "Calculation"],
                 measurable_quantities: "MeasurableQuantities",):
        self.measurement_definitions = measurement_definitions
        self.calculation_definitions = calculation_definitions
        self._tag_list = list(self.measurement_definitions.keys()) \
                      + list(self.calculation_definitions.keys())

        self._usable_results = {}
    
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

