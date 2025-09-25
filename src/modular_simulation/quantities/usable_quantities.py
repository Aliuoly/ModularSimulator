from typing import Dict, List
from pydantic import  PrivateAttr, BaseModel, ConfigDict, Field
from modular_simulation.usables import Calculation, Sensor, TimeValueQualityTriplet

class UsableQuantities(BaseModel):
    """
    1. Defines how measurements and calculations are obtained through
        - measurement_definition: Dict[str, Sensor]
        - calculation_definition: Dict[str, Calculation]
    2. Saves the current snapshot of measurements and calculations
        - results: Dict[str, Any]
    """
    sensors: List[Sensor]
    calculations: List[Calculation] = Field(default_factory = list)

    _usable_results: Dict[str, TimeValueQualityTriplet] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def update(self, t: float) -> Dict[str, TimeValueQualityTriplet]:
        """
        updates the measurements and performs calculations with the latest info.
        Results are automatically linked to the controllers that depend on these 
        but a dictionary of results is still returned for tracking. 
        """

        self._usable_results.update(
            {sensor.measurement_tag: sensor.measure(t) for sensor in self.sensors}
        )
        self._usable_results.update(
            {calculation.output_tag: calculation.calculate(t) for calculation in self.calculations}
        )
        return self._usable_results
    
    @property
    def usable_results(self):
        if len(self._usable_results.keys()) == 0:
            self.update(t = 0)
        return self._usable_results

