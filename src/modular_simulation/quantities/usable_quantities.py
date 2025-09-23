from typing import Dict, List
from pydantic import  PrivateAttr, BaseModel, ConfigDict
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
    calculations: List[Calculation]

    _usable_results: Dict[str, TimeValueQualityTriplet] = PrivateAttr(default_factory=dict)
    _sensor_history: Dict[str, List[TimeValueQualityTriplet]] = PrivateAttr(default_factory=dict)
    _calculation_history: Dict[str, List[TimeValueQualityTriplet]] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def update(self, t: float) -> Dict[str, TimeValueQualityTriplet]:
        """
        updates the measurements and performs calculations with the latest info.
        Results are automatically linked to the controllers that depend on these 
        but a dictionary of results is still returned for tracking. 
        """

        for sensor in self.sensors:
            result = sensor.measure(t)
            tag = sensor.measurement_tag
            self._usable_results[tag] = result
            self._sensor_history.setdefault(tag, []).append(result)
        for calculation in self.calculations:
            result = calculation.calculate(t)
            tag = calculation.output_tag
            self._usable_results[tag] = result
            self._calculation_history.setdefault(tag, []).append(result)

        return self._usable_results

    @property
    def history(self) -> Dict[str, Dict[str, List[TimeValueQualityTriplet]]]:
        return {
            "sensors": {tag: list(entries) for tag, entries in self._sensor_history.items()},
            "calculations": {tag: list(entries) for tag, entries in self._calculation_history.items()},
        }

