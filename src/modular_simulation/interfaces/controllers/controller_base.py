from __future__ import annotations # without this cant type hint the cascade controller properly
from abc import ABC, abstractmethod
import math
from typing import Any, Callable, TYPE_CHECKING, Optional
from numpy.typing import NDArray
import numpy as np
from enum import IntEnum
from pydantic import BaseModel, PrivateAttr, Field, ConfigDict, SerializeAsAny
from modular_simulation.interfaces.tag_info import TagData, TagInfo
from modular_simulation.interfaces.controllers.trajectory import Trajectory
from modular_simulation.validation.exceptions import ControllerConfigurationError
from astropy.units import Quantity, UnitBase, UnitConversionError #type: ignore
if TYPE_CHECKING:
    from collections.abc import Sequence
    from modular_simulation.core.dynamic_model import DynamicModel
    from modular_simulation.interfaces.calculations.calculation_base import CalculationBase
    from modular_simulation.interfaces.sensors.sensor_base import SensorBase
import logging
logger = logging.getLogger(__name__)

def do_nothing_mv_setter(value):
    pass

def make_mv_setter(control_elements: Any, tag: str, unit_converter:Callable[[float],float]):
    
    def mv_setter(value) -> None:
        setattr(control_elements, tag, unit_converter(value))
    return mv_setter

def make_cv_getter(raw_tag_info: TagInfo, desired_tag_info: TagInfo):
    if raw_tag_info.unit != desired_tag_info.unit:
        if raw_tag_info.unit.is_equivalent(desired_tag_info.unit):
            converter = raw_tag_info.unit.get_converter(desired_tag_info.unit)
            def cv_getter() -> TagData:
                return TagData(
                    raw_tag_info.data.time, 
                    converter(raw_tag_info.data.value),
                    raw_tag_info.data.ok
                )
            return cv_getter
        else:
            raise ControllerConfigurationError(
                f"Tried to convert tag '{raw_tag_info.tag}' from '{raw_tag_info.unit}' to '{desired_tag_info.unit}' and failed. "
                "Make sure these units are compatible. "
            )
    else:
        def cv_getter() -> TagData:
            return raw_tag_info.data
        return cv_getter

def wrap_cv_getter_as_sp_getter(cv_getter):
    """
    Turns a cv_getter, which takes no arguments, 
        into a sp_getter, which takes t as argument
    """
    def sp_getter(t: float):
        return cv_getter()
    return sp_getter


class ControllerMode(IntEnum):
    TRACKING = -1
    AUTO    = 1
    CASCADE = 2

class ControllerBase(BaseModel, ABC):
    """Shared infrastructure for feedback controllers operating on usable tags.

    The base class translates between system measurements, setpoint
    trajectories, cascade controllers, and manipulated variables.  Subclasses
    provide the actual control law via :meth:`_control_algorithm`, while this
    class handles unit conversion, MV range enforcement, first-order setpoint
    ramping, historization, and automatic mode transitions between AUTO,
    CASCADE, and TRACKING.
    """
    mv_tag: str = Field(
        ..., 
        description="The tag of the ControlElement corresponding to the " \
            "manipulated variable (MV) for this controller.")
    cv_tag: str = Field(
        ...,
        description=(
            "The tag corresponding to the measured or calculated controlled variable (CV) for this controller"
        ),
    )
    sp_trajectory: Trajectory = Field(
        ..., 
        description="A Trajectory instance defining the setpoint (SP) over time.")
    mv_range: tuple[float, float] = Field(
        ...,
        description = (
            "Lower and upper bound of the manipulated variable, in that order. "
            "The unit is assumed to be the same unit as the mv_tag's unit. "
            "If you want to specify some other unit, consider changing the "
            "measured unit of mv_tag or making a conversion calculation separately."
        )
    )
    ramp_rate: float | None = Field(
        default=None,
        description=(
            "Optional rate limit for the controller setpoint in units per second. "
            "When provided, the requested setpoint from the trajectory will be ramped "
            "towards the target at no more than this rate during each update."
        ),
    )
    cascade_controller: SerializeAsAny[ControllerBase | None] = Field(
        default = None,
        description = (
            "If provided, and when controller mode is CASCADE, the setpoint source "
            "of this controller will be provided by the provided cascade controller. "
        )
    )
    mode: ControllerMode = Field(
        # default to CASCADE so highest loop gets used. 
        # Highest loop with no cascade controller will automatically change to AUTO.
        default = ControllerMode.CASCADE, 
        description = (
            "ControllerBase's mode - if AUTO, setpoint comes from the sp_trajectory provided. "
            "If CASCADE, setpoint comes from the cascade controller, if provided. "
            "If no cascade controller is provided, this mode will fall back to AUTO with a warning. "
        )
    )
    _is_final_control_element: bool = PrivateAttr(default = True)
    _sp_getter: Callable[[float], TagData] = PrivateAttr()
    _cv_getter: Callable[[], TagData] = PrivateAttr()
    _mv_setter: Callable[[float|NDArray], None] = PrivateAttr()
    _sp_tag_info: TagInfo = PrivateAttr()
    _last_output: TagData = PrivateAttr()
    _u0: float | NDArray = PrivateAttr(default = 0.)
    _sp_history: list[TagData] = PrivateAttr(default_factory = list)
    _mv_system_unit: UnitBase = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True, extra = "forbid")

    def _make_sp_tag_info(self, tag_infos: list[TagInfo]):
        """Construct SP tag metadata and convert MV limits into system units."""
        for tag_info in tag_infos:
            if tag_info.tag == self.cv_tag:
                self._sp_tag_info = TagInfo(
                    tag = f"{self.cv_tag}.sp", 
                    unit = self.sp_trajectory.unit,
                    description = f"setpoint for {self.cv_tag}"
                )
            if tag_info.tag == self.mv_tag: 
                self._mv_system_unit = tag_info.unit
                
        return self._sp_tag_info
    
    def _initialize_cv_getter(
        self,
        tag_infos: list[TagInfo]
        ) -> None:
        # validation already done during quantity initiation. No error checking here. 
        for tag_info in tag_infos:
            if tag_info.tag == self.cv_tag:
                self._cv_getter = make_cv_getter(
                    raw_tag_info = tag_info,
                    desired_tag_info = self._sp_tag_info
                    )
                break
    def _initialize_non_final_control_element_mv_setter(
        self,
        tag_infos : list[TagInfo]
        ) -> None:
        # validation already done during quantity initiation. No error checking here. 
        for tag_info in tag_infos:
            if tag_info.tag == self.mv_tag:
                # set the '0 point' value with whatever the measurement is
                self._u0 = tag_info.data.value
                self._mv_setter = do_nothing_mv_setter
                break
        
        self._last_output = TagData(time = 0, value = self._u0, ok = True)
        
    def _initialize_mv_setter(
        self,
        control_elements: Any,
        tag_infos: list[TagInfo],
        ) -> None:
        # validation already done during quantity initiation. No error checking here. 

        for control_element_name in control_elements.model_dump():
            if control_element_name == self.mv_tag:
                # set the '0 point' value with whatever the measurement is
                
                for tag_info in tag_infos:
                    if tag_info.tag == self.mv_tag:
                        unit_converter = tag_info.unit.get_converter(
                            control_elements.tag_unit_info[tag_info.tag]
                        )
                        backwards_converter = control_elements.tag_unit_info[tag_info.tag].get_converter(
                            tag_info.unit
                        )
                        self._u0 = backwards_converter(getattr(control_elements, control_element_name))
                        self._mv_setter = make_mv_setter(control_elements, self.mv_tag, unit_converter)
                        break
                
        self._last_output = TagData(time = 0, value = self._u0, ok = True)
    
    def change_control_mode(self, mode: ControllerMode | str) -> None:
        """public facing method for changing controller mode"""
        self._change_control_mode(mode, initialization = False)

    def _change_control_mode(self, mode: ControllerMode | str, initialization: bool = False) -> None:
        """private method for changing controller mode, with optional argument initialization,
                which stops logging the CASCADE fallback to AUTO for INFO level."""
        if isinstance(mode, str):
            mode = mode.lower().strip()
            if mode == "auto":
                mode = ControllerMode.AUTO
            elif mode == "cascade":
                mode = ControllerMode.CASCADE
            elif mode == "tracking":
                mode = ControllerMode.TRACKING
            else:
                raise ValueError(
                    f"Unrecognized controller mode '{mode}'. Please pick from 'auto' or 'cascade'"
                )
        logger.debug(f"'{self.cv_tag}' controller mode is changed from {self.mode.name} --> {mode.name}")
        if mode == ControllerMode.CASCADE:
            if self.cascade_controller is None:
                if not initialization:
                    logger.info(
                        f"Attemped to change '{self.cv_tag}' controller mode to CASCADE; however, "
                        "no cascade controller was provided. Falling back to AUTO mode. "
                    )
                else:
                    logger.debug(
                        f"Attemped to change '{self.cv_tag}' controller mode to CASCADE; however, "
                        "no cascade controller was provided. Falling back to AUTO mode. "
                    )
                self._change_control_mode(ControllerMode.AUTO, initialization)
            else:
                # if has cascade controller, the setpoint of this "inner loop" controller
                # will be provided by the .update method of the cascade controller. 
                self._sp_getter = self.cascade_controller.update
                self.mode = ControllerMode.CASCADE
                # since inner loop now cascades to cascade controller,
                # the mode for cascade should switch to AUTO
                # the if logic is for initialization time where the cascade controller
                # Might already be defaulted to CASCADE - in which case, keep it CASCADE.
                # This ensures the cascade ladder goes all the way to the outer-most loop
                # during initialization. 
                if self.cascade_controller.mode != ControllerMode.CASCADE:
                    self.cascade_controller._change_control_mode(ControllerMode.AUTO, initialization)
        elif mode == ControllerMode.AUTO: 
            self._sp_getter = self.sp_trajectory
            self.mode = ControllerMode.AUTO
            # if has cascade controller, it must shed to TRACKING mode now
            if self.cascade_controller is not None:
                self.cascade_controller._change_control_mode(ControllerMode.TRACKING, initialization)
        elif mode == ControllerMode.TRACKING:
            self._sp_getter = wrap_cv_getter_as_sp_getter(self._cv_getter)
            self.mode = ControllerMode.TRACKING
            # if has cascade controller, it must shed to TRACKING mode now
            if self.cascade_controller is not None:
                self.cascade_controller._change_control_mode(ControllerMode.TRACKING, initialization)
        
        
            
        # if no cascade controller, the setpoint of this controller
        # will be provided by the sp_trajectory
    def _initialize(
        self,
        tag_infos: list[TagInfo],
        sensors: "Sequence[SensorBase]",
        calculations: "Sequence[CalculationBase]",
        control_elements: Any,
        is_final_control_element: bool = True,
        ) -> None:
        """Wire the controller into orchestrated quantities and validate modes.

        The container guarantees all referenced tags exist, so this method
        simply creates the getter/setter callables, configures cascade
        relationships, and promotes the controller into the highest valid
        operating mode for the supplied configuration.
        """
        logger.debug(f"Initializing '{self.cv_tag}' controller.")
        
        # A. check if is final control element and initialize mv setter if is
        if not is_final_control_element:
            self._is_final_control_element = False
            self._initialize_non_final_control_element_mv_setter(tag_infos)
        else:
            self._initialize_mv_setter(control_elements, tag_infos)
        
        # B. initialize cv_getter
        self._initialize_cv_getter(tag_infos)

        # C. do control mode validation for initialization (i.e., change from cascade to appropriate one if CASCADE not applicable.)
        self._change_control_mode(self.mode, initialization=True)

        # D. check if has cascade controller and initialize it if so
        if self.cascade_controller is not None:
            logger.debug(f"'{self.cv_tag}' controller is configured to cascade to '{self.cascade_controller.cv_tag}' controller.")
            # if the inner loop is NOT in CASCADE, then force the outer loop to be in TRACKING mode. 
            if self.mode == ControllerMode.AUTO or self.mode == ControllerMode.TRACKING:
                self.cascade_controller._change_control_mode(ControllerMode.TRACKING)
                logger.debug(f"'{self.cv_tag}' controller not in CASCADE mode -> cascade controller forced to TRACKING. ")
            
            self.cascade_controller._initialize(
                tag_infos=tag_infos,
                sensors=sensors,
                calculations=calculations,
                control_elements=control_elements,
                is_final_control_element=False,
            )
        

    def update(self, t: float) -> TagData:
        """Run one control-cycle update and return the applied MV value.

        The routine pulls the latest controlled-variable reading, sources the
        appropriate setpoint (local trajectory, cascade input, or tracking),
        evaluates the subclass control law, applies MV ramp limits, and writes
        the result back into the system if this is the final controller in the
        cascade chain.
        """
        # 0. check dt -> if 0, skip all and return _last_output
        if t - self._last_output.time < 1e-12:
            return self._last_output
        # 1. get controlled variable (aka pv). Is always a Triplet
        cv = self._cv_getter()
        # IF TRACKING - set sp = pv and return last control output exactly. 
        # and append to history since that gets skipped. 
        # also handle the case more cascade levels are present
        # in which case, they all get skipped as well since the
        # .update method never gets called (since it is used as the _sp_getter only when lower loop is CASCADE)
        if self.mode == ControllerMode.TRACKING:
            self.sp_trajectory.set_now(t = t, value = cv.value)
            self._sp_history.append(cv)
            if self.cascade_controller is not None:
                self.cascade_controller.update(t)
            return self._last_output
        
        if cv is None:
            raise RuntimeError(
                f"ControllerBase '{self.cv_tag}' not initialized. "
            )
        # 2. get set point. If is a from a cascade controller, is a Triplet
        #       if is not from a cascade controller, is a float or NDArray from sp_trajectory(t)
        sp = self._sp_getter(t)
        
        if isinstance(sp, TagData):
            # this means we are in cascade control mode
            # so we go ahead and update trajectory with cascade setpoint
            sp_val = sp.value
            proceed = sp.ok and cv.ok
            self._sp_history.append(sp)
            self.sp_trajectory.set_now(t = t, value = sp_val)
        else:
            sp_val = sp
            proceed = cv.ok
            #setpoint from sp_trajectory quality is always good (user sets it)
            self._sp_history.append(TagData(t, sp_val, True)) 
        
        # if setpoint or cv quality is bad, skip controller update and return last value
        if not proceed:
            return self._last_output

        # compute control output
        control_output = self._control_algorithm(t = t, cv = cv.value, sp = sp_val)
        if np.isnan(control_output):
            raise ValueError(
                f"{self.cv_tag} controller output is NAN!"
            )
        control_output = np.clip(control_output, *self.mv_range)
        ramp_output = self._apply_mv_ramp(
            t0 = self._last_output.time, mv0 = self._last_output.value, mv_target = control_output, t_now = t
            )
        if self._is_final_control_element:
            self._mv_setter(ramp_output)

        # do misc things like set _last_output, make sp track PV (if TRACKING mode), update sp_trajectory with cascade sp (if CASCADE mode)
        self._last_output = TagData(time = t, value = ramp_output, ok = True) # TODO: figure out what to do with quality for MV - seems to always be OK.

        return self._last_output

    def _apply_mv_ramp(
            self,
            t0: float,
            mv0: float | NDArray,
            mv_target: float | NDArray,
            t_now: float
            ) -> float | NDArray:
        """Ramp-limit the manipulated variable according to ``ramp_rate``."""
        if self.ramp_rate is None:
            return mv_target

        dt = t_now - t0
        if dt <= 0.0:
            return mv0

        max_delta = self.ramp_rate * dt
        delta = mv_target - mv0
        if np.absolute(delta) <= max_delta:
            return mv_target
        else:
            return mv0 + math.copysign(max_delta, delta)

    @abstractmethod
    def _control_algorithm(
            self,
            t: float,
            cv: float | NDArray,
            sp: float | NDArray,
            ) -> float | NDArray:
        """The actual control algorithm. To be implemented by subclasses."""
        pass
    
    @property
    def sp_history(self) -> dict[str, list[TagData]]:
        """Return a mapping of cascade level to historized setpoint samples."""
        history: dict[str, list[TagData]] = {}
        controller: Optional["ControllerBase"] = self
        while controller is not None:
            history[f"{controller.cv_tag}.sp"] = controller._sp_history
            controller = controller.cascade_controller
        return history
