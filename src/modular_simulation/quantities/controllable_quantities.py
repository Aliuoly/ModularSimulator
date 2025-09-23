from typing import List, Dict
from dataclasses import dataclass, field
from modular_simulation.usables import TimeValueQualityTriplet
from modular_simulation.control_system.controllers.controller import Controller
from modular_simulation.control_system.controllers.cascade_controller import CascadeController
@dataclass(slots = True)
class ControllableQuantities:
    """Container for the controllers acting on the system's control elements."""
    controllers: List[Controller | CascadeController]

    _control_outputs: Dict[str, TimeValueQualityTriplet] = field(init = False, default_factory = dict)
    _setpoint_history: Dict[str, List[TimeValueQualityTriplet]] = field(init = False, default_factory = dict)

    def update(self, t: float) -> Dict[str, TimeValueQualityTriplet]:
        """updates the controllers available. Controllers are linked to the instance of ControlElement
        internally, so the results are reflected in the simulation automatically without having
        to return anything here. However, it is still returned for tracking purposes."""
        for controller in self.controllers:
            result = controller.update(t)
            self._control_outputs[controller.mv_tag] = result

            if isinstance(controller, CascadeController):
                sp_value = controller.active_sp_trajectory().current_value(t)
            else:
                sp_value = controller._last_sp_command
                if sp_value is None:
                    sp_value = controller.sp_trajectory(t)
            history_entry = TimeValueQualityTriplet(t=t, value=sp_value, ok=True)
            self._setpoint_history.setdefault(controller.cv_tag, []).append(history_entry)

        return self._control_outputs

    @property
    def setpoint_history(self) -> Dict[str, List[TimeValueQualityTriplet]]:
        return {tag: list(entries) for tag, entries in self._setpoint_history.items()}
