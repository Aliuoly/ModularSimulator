from __future__ import annotations # without this cant type hint the cascade controller properly
from abc import ABC, abstractmethod
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated
import numpy as np
from enum import IntEnum
from astropy.units import Quantity
from pydantic import BaseModel, PrivateAttr, Field, ConfigDict, SerializeAsAny, BeforeValidator, PlainSerializer
from modular_simulation.usables.tag_info import TagData, TagInfo
from modular_simulation.usables.controllers.trajectory import Trajectory
from modular_simulation.validation.exceptions import ControllerConfigurationError
from modular_simulation.utils.typing import Seconds, StateValue, PerSeconds, SerializableUnit
from modular_simulation.utils.wrappers import second, second_value
from modular_simulation.measurables.process_model import ProcessModel
if TYPE_CHECKING:
    from modular_simulation.framework import System
import logging
logger = logging.getLogger(__name__)

def do_nothing_mv_setter(value):
    pass

def wrap_cv_getter_as_sp_getter(cv_getter):
    """
    Turns a cv_getter, which takes no arguments, 
        into a sp_getter, which takes t as argument
    """
    def sp_getter(t: Seconds):
        return cv_getter()
    
    return sp_getter

class ControllerMode(IntEnum):
    """
    TRACKING: CV's SP = CV's PV Always. SP cannot be changed - always follows PV.
    AUTO    : CV's SP is provided by sp_trajectory. 
    CASCADE : CV's SP is provided by a cascade controller.
    """
    TRACKING = -1 
    AUTO    = 1
    CASCADE = 2

class ControllerBase(BaseModel, ABC):
    """
    Shared infrastructure for feedback controllers operating on usable tags.

    This base class implements the common infrastructure underlying all feedback
    controllers. It mediates between system measurements, setpoint trajectories,
    cascade controller outputs, and process actuators. Subclasses define only the
    core control law via :meth:`_control_algorithm`, while the base class manages
    the rest — including unit conversion, manipulated variable (MV) range
    enforcement, setpoint ramping, historization, and automatic mode transitions
    between AUTO, CASCADE, and TRACKING.

    Under the hood, the base class provides the following mechanisms:

    **1. Measurement and Actuation Interface**

    - **PV retrieval:** via linkage to bound sensors or calculation outputs.
    - **SP retrieval:** via local :class:`Trajectory` or the output of a cascade controller.
    - **MV actuation:** via a bound setter to the :class:`ProcessModel`'s controlled state.

    If the controller's MV unit differs from the process model's internal state unit,
    the setter automatically performs the necessary conversion.

    **2. Model-Based Control Support - should be done in the controller subclass

    The class should natively supports model-based controllers (e.g., IMC) by managing
    the conversion between controller units and internal model units. This enables
    operations such as:

        controller_input (MV, controller units)
            → model_input (model units)
            → model prediction (PV, model units)
            → converter
            → PV (controller units)
            → residual = SP - PV_prediction

    When the final control output is computed, it is automatically converted and
    written to the process model.

    **3. Converter Infrastructure**

    The following converters are automatically configured and applied as needed:

    +---------------------------------------------+----------------------------------------------+
    | Conversion Path                             | Purpose                                      |
    +=============================================+==============================================+
    | MV (measured unit) → MV (process model unit) | For writing the final control output to the  |
    |                                             | process model.                               |
    +---------------------------------------------+----------------------------------------------+
    | MV (measured unit) → MV (model unit)         | For use in model-based control calculations. |
    +---------------------------------------------+----------------------------------------------+
    | PV (model unit) → PV (measured unit)         | For reconciling model predictions with the   |
    |                                             | controller's observed process values.        |
    +---------------------------------------------+----------------------------------------------+

    In essence, the base class abstracts away all translation and coordination
    between physical measurements, model-based predictions, and manipulated
    variables, allowing controller subclasses to focus solely on implementing
    their specific control law.
    """

    mv_tag: str = Field(
        ..., 
        description="The tag of the ControlElement corresponding to the " \
            "manipulated variable (MV) for this controller.")
    cv_tag: str = Field(
        ...,
        description = "The tag of the UsableQuantities corresponding to the" \
                        "measured or calculated controlled variable (CV) for this controller"
    )
    sp_trajectory: Trajectory = Field(
        ..., 
        description="A Trajectory instance defining the setpoint (SP) over time.")
    mv_range: tuple[StateValue, StateValue] = Field(
        ...,
        description = (
            "Lower and upper bound of the manipulated variable, in that order. "
            "The unit is assumed to be the same unit as the mv_tag's unit. "
            "If you want to specify some other unit, consider changing the "
            "measured unit of mv_tag or making a conversion calculation separately."
        )
    )
    ramp_rate: PerSeconds | None = Field(
        default=None,
        description=(
            "Optional ramp rate limit for the controller setpoint. "
            "Must be provided as a 'per time' unitized Quantity, where "
            "the numerator of the Quantity is assumed to be the unit of the " 
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
    period: Seconds = Field(
        default = 1e-12,
        description = (
            "minimum execution period of the controller. Controlelr will execute "
            "as frequently as possible such that the time between execution is as "
            "close to this value as possible. "
        )
    )
    _is_final_control_element: bool = PrivateAttr(default = True)
    _sp_getter: Callable[[Seconds], TagData] = PrivateAttr()
    _cv_getter: Callable[[], TagData] = PrivateAttr()
    _mv_setter: Callable[[StateValue], None] = PrivateAttr()
    _initialized: bool = PrivateAttr(default = False)
    _control_action: TagData = PrivateAttr()
    _sp_tag_info: TagInfo = PrivateAttr()
    _is_scalar:bool = PrivateAttr(default = False)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra = "forbid")
    
    def _make_cv_getter(self, available_tag_info_dict: dict[str, TagInfo]) -> None:
        cv_tag_info = available_tag_info_dict.get(self.cv_tag)
        if cv_tag_info is None:
            raise ControllerConfigurationError(
                f"Could not find controlled variable '{self.cv_tag}' for '{self.cv_tag}' controller "
                "in measurements nor calculation outputs. "
            )
        self._cv_getter = cv_tag_info.make_converted_data_getter(cv_tag_info.unit)
        
    def _make_mv_setter(
        self,
        process: ProcessModel,
        available_tag_info_dict: dict[str, TagInfo],
        is_final_control_element: bool
        ) -> None:
        mv_tag_info = available_tag_info_dict.get(self.mv_tag)
        if mv_tag_info is None:
            raise ControllerConfigurationError(
                f"Could not find manipuated variable '{self.mv_tag}' for '{self.cv_tag}' controller "
                "in measurements nor calculation outputs. "
            )
        if is_final_control_element:
            # search for the raw tag in case the mv tag is an alias
            control_element_name = mv_tag_info._raw_tag
            mv_state_metadata = process.state_metadata_dict.get(control_element_name)
            if mv_state_metadata is None:
                raise ControllerConfigurationError(
                    f"Could not find control element '{self.mv_tag}' for '{self.cv_tag}' controller in process model."
                )
            self._mv_setter = process.make_converted_setter(control_element_name, mv_tag_info.unit)
        else:
            self._mv_setter = do_nothing_mv_setter
            
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
    def commission(
        self,
        system: System,
        is_final_control_element: bool = True,
        ) -> None:
        """Wire the controller into orchestrated quantities and validate modes.

        The container guarantees all referenced tags exist, so this method
        simply creates the getter/setter callables, configures cascade
        relationships, and promotes the controller into the highest valid
        operating mode for the supplied configuration.
        """
        logger.debug(f"Initializing '{self.cv_tag}' controller.")
        # initialization happens at highest cascade level first, and then
        # changes mode back to the configured mode. 
        controller_mode = self.mode
        self.mode = ControllerMode.CASCADE
        
        if not is_final_control_element:
            self._is_final_control_element = False
        self._make_mv_setter(system.process_model, system.tag_info_dict, is_final_control_element)
        self._make_cv_getter(system.tag_info_dict)

        # C. do control mode validation for initialization (i.e., change from cascade to appropriate one if CASCADE not applicable.)
        self._change_control_mode(self.mode, initialization=True)

        # D. check if has cascade controller and initialize it if so
        if self.cascade_controller is not None:
            logger.debug(f"'{self.cv_tag}' controller is configured to cascade to '{self.cascade_controller.cv_tag}' controller.")
            # if the inner loop is NOT in CASCADE, then force the outer loop to be in TRACKING mode. 
            if self.mode == ControllerMode.AUTO or self.mode == ControllerMode.TRACKING:
                self.cascade_controller._change_control_mode(ControllerMode.TRACKING)
                logger.debug(f"'{self.cv_tag}' controller not in CASCADE mode -> cascade controller forced to TRACKING. ")
            
            self.cascade_controller.commission(
                system = system,
                is_final_control_element=False,
                )
        
        #revert back to configured mode
        self.change_control_mode(controller_mode)

        #now run post commission hook in case it is implemented
        postfix = ".sp" if is_final_control_element else ".csp"
        mv_type = "final control element" if is_final_control_element else "cascade controller"
        self._sp_tag_info = TagInfo(
            tag = f"{self.cv_tag}{postfix}", 
            unit = system.tag_info_dict[self.cv_tag].unit,
            description = f"setpoint for {self.cv_tag} ({mv_type})"
        )
        system.update_tag_info_dict(self._sp_tag_info.tag, self._sp_tag_info)

        # make the control action default to starting state value
        mv_tag_info = system.tag_info_dict[self.mv_tag]
        mv_state_unit = system.process_model.state_metadata_dict[mv_tag_info._raw_tag].unit
        converted_data = mv_tag_info.make_converted_data_getter(mv_state_unit)()
        converted_data.time = system.time
        converted_data.ok = True
        self._control_action = converted_data
        
        self._post_commission(system)
        self._initialized = True
        if np.isscalar(converted_data.value):
            self._is_scalar = True
        
    def _post_commission(self, system: System):
        pass

    def update(self, t: Seconds) -> TagData:
        """Run one control-cycle update and return the applied MV value.

        The routine pulls the latest controlled-variable reading, sources the
        appropriate setpoint (local trajectory, cascade input, or tracking),
        evaluates the subclass control law, applies MV ramp limits, and writes
        the result back into the system if this is the final controller in the
        cascade chain.
        """
        # 0. check dt -> if 0, skip all and return _last_output
        if not self._initialized:
            raise RuntimeError(f"{self.cv_tag} controller not commissioned.")
        
        last_control_action = self._control_action
        if t - last_control_action.time < self.period:
            return last_control_action
        # 1. get controlled variable (aka pv). Is always a Triplet
        cv = self._cv_getter()
        
        if self.mode == ControllerMode.TRACKING:
            self.sp_trajectory.set_now(t = t, value = cv.value)
            self._sp_tag_info.data = cv
            if self.cascade_controller is not None:
                self.cascade_controller.update(t)
            return last_control_action

        sp = self._sp_getter(t)
        
        if isinstance(sp, TagData):
            # this means we are in cascade control mode
            # so we go ahead and update trajectory with cascade setpoint
            sp_val = sp.value
            proceed = sp.ok and cv.ok
            self._sp_tag_info.data = sp
            self.sp_trajectory.set(t = t, value = sp_val)
        else:
            sp_val = sp
            proceed = cv.ok
            #setpoint from sp_trajectory quality is always good (user sets it)
            self._sp_tag_info.data = TagData(t, sp_val, True)
        
        # if setpoint or cv quality is bad, skip controller update and return last value
        if not proceed:
            last_control_action.ok = False
            return last_control_action

        # compute control output
        control_output = self._control_algorithm(t = t, cv = cv.value, sp = sp_val)
        if np.isnan(control_output):
            logger.warning(
                f"'{self.cv_tag}' controller returned an invalid value '{control_output}'. "
                f"Holding previous control action ({last_control_action.value}) "
            )
            last_control_action.ok = False
            return last_control_action
        if self._is_scalar:
            control_output = max(min(control_output, self.mv_range[1]), self.mv_range[0])
        else:
            control_output = np.clip(control_output, *self.mv_range)
        ramp_output = self._apply_mv_ramp(
            t0 = last_control_action.time, mv0 = last_control_action.value, mv_target = control_output, t_now = t
            )
        if self._is_final_control_element:
            self._mv_setter(ramp_output)

        self._control_action = TagData(time = t, value = ramp_output, ok = True) # TODO: figure out what to do with quality for MV - seems to always be OK.

        return self._control_action

    def _apply_mv_ramp(
            self,
            t0: Seconds,
            mv0: StateValue,
            mv_target: StateValue,
            t_now: Seconds
            ) -> StateValue:
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
            t: Seconds,
            cv: StateValue,
            sp: StateValue,
            ) -> StateValue:
        """The actual control algorithm. To be implemented by subclasses."""
        pass
    
    @property
    def sp_history(self) -> dict[str, TagData]:
        return {self._sp_tag_info.tag: self._sp_tag_info.history}

    @property
    def t(self) -> Seconds:
        return self._control_action.time
    