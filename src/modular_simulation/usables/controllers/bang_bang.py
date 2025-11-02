from __future__ import annotations
from pydantic import Field, PrivateAttr
from modular_simulation.usables.controllers.controller_base import ControllerBase
from typing import TYPE_CHECKING
import logging
from modular_simulation.utils.typing import Seconds, StateValue
if TYPE_CHECKING:
    from modular_simulation.framework.system import System
logger = logging.getLogger(__name__)

class BangBangController(ControllerBase):
    """bang-bang controller for either on or off controllers. 
        Default returns on_value starting from when the measurement goes below UL 
        until when the measurement goes above HL. 
        Vice versa, will return off_value when measurement goes above HL
         until when the measurement goes below UL.
        The mv range MUST have off value < on value, but 
        the default behavior can be INVERTED with the inverted flag.
        """
    deadband: float = Field(
        ...,
        description = (
            "The deadband of the bang bang controller around the setpoint. "
            "The bang bang controller will try to maintaince the cv in the range of "
            "setpoint - deadband to setpoint + deadband."
        )
    )
    alpha: float = Field(
        gt = 0.0,
        le = 1.0,
        default = 1.0,
        description = "low pass filter factor on measurement. 1.0 = no filter"
    )
    mv_range: tuple[float, float] = Field(
        default = (0., 1.),
        description = "limits assumed to correspond to off (first element) or on (second element)"
    )
    inverted: bool = Field(
        default = False,
        description = "If True, the controller inverts the on/off logic described in the documentation."
    )
    _cv_filtered: float = PrivateAttr() 
    _state: bool = PrivateAttr(default = False)

    def _post_commission(self, system: System):
        self._cv_filtered = system.tag_info_dict[self.cv_tag].data.value
    def _control_algorithm(
        self,
        t: Seconds,
        cv: StateValue,
        sp: StateValue,
        ) -> tuple[StateValue, bool]:
        
        self._cv_filtered = self.alpha * cv + (1 - self.alpha) * self._cv_filtered

        HL = sp + self.deadband
        LL = sp - self.deadband
        if self._cv_filtered >= HL:
            self._state = False # if above high limit, turn OFF controller
        elif self._cv_filtered <= LL:
            self._state = True # if below low limit, turn ON controller
        if self.inverted:
            self._state = not self._state
        
        if self._state:
            return self.mv_range[1], True
        else:
            return self.mv_range[0], True