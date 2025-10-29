from pydantic import Field, PrivateAttr
import numpy as np
from modular_simulation.usables.controllers.controller_base import ControllerBase
from numpy.typing import NDArray
import logging
logger = logging.getLogger(__name__)
class MVController(ControllerBase):

    """
    Sets the mv (controller output) to be the setpoint, whereever it comes from.
    """

    def _control_algorithm(
            self,
            t: float,
            cv: float | NDArray[np.float64],
            sp: float | NDArray[np.float64],
            ) -> float | NDArray[np.float64]:
        
        return sp