from pydantic import Field, PrivateAttr
from modular_simulation.control_system.controller import Controller
from typing import Tuple
import logging
logger = logging.getLogger(__name__)

class BangBangController(Controller):
    """bang-bang controller for either on or off controllers. 
        Default returns on_value starting from when the measurement goes below UL 
        until when the measurement goes above HL. 
        Vice versa, will return off_value when measurement goes above HL
         until when the measurement goes below UL.
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
    mv_range: Tuple[float, float] = Field(
        default = (0., 1.),
        description = "limits assumed to correspond to off (first element) or on (second element)"
    )
    
    _cv_filtered: float|None = PrivateAttr(default = None) 
    _state: bool = PrivateAttr(default = False)

    def _control_algorithm(
        self,
        t: float,
        cv: float,
        sp: float,
        ) -> float:
        
        if self._cv_filtered is None:
            self._cv_filtered = cv
        else:
            self._cv_filtered = self.alpha * cv + (1 - self.alpha) * self._cv_filtered

        HL = sp + self.deadband
        LL = sp - self.deadband
        if self._cv_filtered >= HL:
            self._state = False # if above high limit, turn OFF controller
        elif self._cv_filtered <= LL:
            self._state = True # if below low limit, turn ON controller
        
        if self._state:
            return self.mv_range[1]
        else:
            return self.mv_range[0]