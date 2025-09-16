from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING
from numpy.typing import NDArray
import numpy as np
from pydantic import BaseModel, PrivateAttr, Field, ConfigDict
if TYPE_CHECKING:
    from modular_simulation.quantities import UsableResults
    from modular_simulation.control_system import Trajectory
    from modular_simulation.usables import Sensor


class Controller(BaseModel, ABC):

    pv_tag: str = Field(..., description="The measurement tag of the sensor providing the process variable (PV) for this controller.")
    sp_trajectory: "Trajectory" = Field(..., description="A Trajectory instance defining the setpoint (SP) over time.")

    _pv_sensor: Union["Sensor", None] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def link_pv_sensor(self, sensor: "Sensor") -> None:
        """Stores a direct reference to the sensor providing the PV."""
        self._pv_sensor = sensor

    def update(
            self, 
            usable_results: "UsableResults",
            t: float,
            ) -> Union[float, NDArray[np.float64]]:
        # 1. get pv
        pv = self._get_pv_value()
        # 2. get sp
        sp = self.sp_trajectory(t)
        # 3. compute control output
        control_output = self._control_algorithm(pv, sp, usable_results, t)
        return control_output

    @abstractmethod
    def _control_algorithm(
            self,
            pv_value: Union[float, NDArray[np.float64]],
            sp_value: Union[float, NDArray[np.float64]],
            usable_results: "UsableResults",
            t: float
            ) -> Union[float, NDArray[np.float64]]:
        """The actual control algorithm. To be implemented by subclasses."""
        pass

    def _get_pv_value(self) -> Union[float, NDArray[np.float64]]:
        if self._pv_sensor is None:
            raise ValueError("No PV sensor linked to controller. Somehow, \
                             the linking step in ControllableQuantities failed and\
                             you didnt get an error. Funny.")
        if self._pv_sensor._last_measurement is None:
            raise ValueError("PV sensor has not yet taken a measurement. \
                              Make sure this sensor actually saves the _last_measurement attribute correctly. \
                              This would not occur unless you overloaded the public facing 'measure' method \
                              of the Sensor class.")
        return self._pv_sensor._last_measurement.value