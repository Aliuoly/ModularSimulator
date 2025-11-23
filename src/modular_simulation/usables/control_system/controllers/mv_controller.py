from typing import override
from modular_simulation.usables.control_system.controller_base import ControllerBase
from modular_simulation.utils.typing import StateValue, Seconds
import logging

logger = logging.getLogger(__name__)


class MVController(ControllerBase):
    """
    Sets the mv (controller output) to be the setpoint, whereever it comes from.
    """

    @override
    def _control_algorithm(
        self,
        t: Seconds,
        cv: StateValue,
        sp: StateValue,
    ) -> tuple[StateValue, bool]:
        return sp, True
