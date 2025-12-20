from __future__ import annotations
from pydantic import Field, PrivateAttr
from modular_simulation.usables.control_system.controller_base import ControllerBase
from typing import TYPE_CHECKING, Callable, override, cast
from astropy.units import UnitBase
from modular_simulation.usables.tag_info import TagData
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
        description=(
            "The deadband of the bang bang controller around the setpoint. "
            "The bang bang controller will try to maintaince the cv in the range of "
            "setpoint - deadband to setpoint + deadband."
        ),
    )
    alpha: float = Field(
        gt=0.0,
        le=1.0,
        default=1.0,
        description="low pass filter factor on measurement. 1.0 = no filter",
    )
    on_action: float = Field(
        default=1.0,
        description="The action to take when the measurement goes above the setpoint deadband.",
    )
    off_action: float = Field(
        default=0.0,
        description="The action to take when the measurement goes below the setpoint deadband.",
    )
    inverted: bool = Field(
        default=False,
        description="If True, the controller inverts the on/off logic described in the documentation.",
    )
    _cv_filtered: float = PrivateAttr()
    _state: bool = PrivateAttr(default=False)

    @override
    def _post_initialization(
        self,
        system: System,
        mv_getter: Callable[[], TagData],
        mv_range: tuple[StateValue, StateValue],
        mv_tag: str,
        mv_unit: UnitBase,
    ) -> bool:
        cv_value = system.tag_store[self.cv_tag].data.value
        try:
            cv_value = float(cv_value)
        except ValueError:
            logger.warning(
                f"BangBangController: mv = {mv_tag} cv ={self.cv_tag} - cv must be a scalar value."
            )
            return False
        self._cv_filtered = cv_value
        return True

    @override
    def _control_algorithm(
        self,
        t: Seconds,
        cv: StateValue,
        sp: StateValue,
    ) -> tuple[StateValue, bool]:
        self._cv_filtered = self.alpha * cast(float, cv) + (1 - self.alpha) * self._cv_filtered

        HL = sp + self.deadband
        LL = sp - self.deadband
        if self._cv_filtered >= HL:
            self._state = False  # if above high limit, turn OFF controller
        elif self._cv_filtered <= LL:
            self._state = True  # if below low limit, turn ON controller
        if self.inverted:
            self._state = not self._state

        if self._state:
            return self.on_action, True
        else:
            return self.off_action, True
