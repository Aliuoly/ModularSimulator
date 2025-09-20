from typing import List, Dict
from dataclasses import dataclass, field
from modular_simulation.usables import TimeValueQualityTriplet
from modular_simulation.control_system.controller import Controller
@dataclass(slots = True)
class ControllableQuantities:
    """Container for the controllers acting on the system's control elements."""
    controllers: List[Controller]

    _control_outputs: Dict[str, TimeValueQualityTriplet] = field(init = False, default_factory = dict)
    
    def update(self, t: float) -> Dict[str, TimeValueQualityTriplet]:
        """updates the controllers available. Controllers are linked to the instance of ControlElement
        internally, so the results are reflected in the simulation automatically without having
        to return anything here. However, it is still returned for tracking purposes."""
        for controller in self.controllers:
            self._control_outputs[controller.mv_tag] = controller.update(t)

        return self._control_outputs
