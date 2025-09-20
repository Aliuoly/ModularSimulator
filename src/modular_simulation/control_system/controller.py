from abc import ABC, abstractmethod
from typing import Union, List, Callable, Tuple, TYPE_CHECKING
from numpy.typing import NDArray
import numpy as np
from pydantic import BaseModel, PrivateAttr, Field, ConfigDict
from modular_simulation.control_system.trajectory import Trajectory
from modular_simulation.measurables import ControlElements
from modular_simulation.usables.time_value_quality_triplet import TimeValueQualityTriplet
if TYPE_CHECKING:
    from modular_simulation.quantities import UsableQuantities
import logging
logger = logging.getLogger(__name__)

class Controller(BaseModel, ABC):

    mv_tag: str = Field(
        ..., 
        description="The tag of the ControlElement corresponding to the " \
            "manipulated variable (MV) for this controller.")
    cv_tag: str = Field(
        ...,
        description = "The tag of the UsableQuantities corresponding to the" \
                        "measured or calculated controlled variable (CV) for this controller"
    )
    sp_trajectory: "Trajectory" = Field(
        ..., 
        description="A Trajectory instance defining the setpoint (SP) over time.")
    mv_range: Tuple[float, float] = Field(
        ...,
        description = "Lower and upper bound of the manipulated variable, in that order."
    )
    _cv_getter: Callable[[], TimeValueQualityTriplet] | None = PrivateAttr(default=None)
    _mv_setter: Callable[[float|NDArray], None] | None = PrivateAttr(default = None)
    _usables: Union["UsableQuantities", None] = PrivateAttr(default = None)
    _last_value: TimeValueQualityTriplet | None = PrivateAttr(default = None)
    _u0: float | NDArray = PrivateAttr(default = 0.)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra = "forbid")
    
    def _initialize(
            self,
            usable_quantities: "UsableQuantities",
            control_elements: ControlElements,
        ) -> None:
        """
        establish the link between sensor corresponding to the CV and 
            between controlelement of the system
        """
        self._usables = usable_quantities
        sensors = usable_quantities.sensors
        calculations = usable_quantities.calculations
        found = False
        for sensor in sensors:
            if sensor.measurement_tag == self.cv_tag:
                # fun bug for people if they ever see this
                # originally, this line was 'self._cv_getter = lambda : sensor._last_value
                # we then continue to loop over sensors to find if there are duplicates. 
                # welp, turns out, the sensor object referenced in the lambda
                # was not 'frozen' to the one we wanted. 
                # instead, as we iterated over sensors and saved the object into 'sensor',
                # the _cv_getter's referencing sensor was changing as well.
                # Anyways, now I relegate the duplicate checking to the usables validation
                # instead of here, and we break early to avoid this. 
                self._cv_getter = lambda : sensor._last_value
                found = True
                break
        for calculation in calculations:
            if calculation.output_tag == self.cv_tag:
                self._cv_getter = lambda : calculation._last_value
                found = True
                break
        if not found:
            raise AttributeError(
                f"{self.cv_tag} controller could not be initialized. The tag '{self.cv_tag}' is not measured. "
                f"Available measurements are: {', '.join([s.measurement_tag for s in sensors])}."
            )
        
        found = False
        for control_element_name in control_elements.__class__.model_fields:
            if control_element_name == self.mv_tag:
                # set the '0 point' value with whatever the measurement is
                self._u0 = getattr(control_elements, control_element_name)
                self._mv_setter = lambda value : setattr(control_elements, self.mv_tag, value)
                found = True
        if not found:
            raise AttributeError(
                f"{self.cv_tag} controller's could not be initialized. "
                f"The specified manipulated variable tag '{self.mv_tag}' is not defined as a control element."
                f"Available control elements are: {', '.join([ce for ce in control_elements.__class__.model_fields])}."
            )
        self._last_value = TimeValueQualityTriplet(t = 0, value = self._u0, ok = True)

    def update(self, t: float) -> Union[float, NDArray[np.float64]]:
        # 1. get pv
        cv = self._get_cv_value()
        # 2. get sp
        sp = self.sp_trajectory(t)
        # 3. compute control output
        control_output = self._control_algorithm(cv, sp, t)

        control_output.value = np.clip(control_output.value + self._u0, *self.mv_range)
        self._last_value = control_output
        if self._mv_setter is not None:
            if control_output.ok:
                self._mv_setter(control_output.value)
            else:
                logger.info("%s controller update skipped due to bad quality. Controller output (%s): %s",
                            self.cv_tag, self.mv_tag, control_output)
        else:
            raise RuntimeError(
                f"{self.cv_tag} controller not yet initialized. "
                "Make sure system was initialized with the create_system function."
                )
        return self._last_value
    
    def update_trajectory(
            self,
            t: float,
            value: float
            ) -> None:
        self.sp_trajectory.set_now(t, value)
    
    def track_cv(
            self,
            t: float
            ) -> None:
        self.sp_trajectory.set_now(t, self._get_cv_value().value)

    @abstractmethod
    def _control_algorithm(
            self,
            cv_value: TimeValueQualityTriplet,
            sp_value: TimeValueQualityTriplet | float | NDArray,
            t: float
            ) -> TimeValueQualityTriplet:
        """The actual control algorithm. To be implemented by subclasses."""
        pass

    def _get_cv_value(self) -> TimeValueQualityTriplet:
        if self._cv_getter is None:
            raise RuntimeError(
                "No PV sensor linked to controller. Something went wrong during system initialization."
                )
        cv = self._cv_getter()
        if cv is None:
            raise ValueError("PV sensor has not yet taken a measurement. \
                    Make sure this sensor actually saves the _last_value attribute correctly. \
                    This would not occur unless you overloaded the public facing 'measure' method \
                    of the Sensor class.")
        return cv