from __future__ import annotations  # without this cant type hint the cascade controller properly
from abc import ABC, abstractmethod
import math
from collections.abc import Callable
from typing import TYPE_CHECKING
from astropy.units import UnitBase
import numpy as np
from .controller_mode import ControllerMode
from pydantic import BaseModel, PrivateAttr, Field, ConfigDict, SerializeAsAny
from modular_simulation.usables.tag_info import TagData, TagInfo
from .trajectory import Trajectory
from modular_simulation.validation.exceptions import ControllerConfigurationError
from modular_simulation.utils.typing import Seconds, StateValue, PerSeconds
from .mode_manager import ControllerModeManager

if TYPE_CHECKING:
    from modular_simulation.framework import System
import logging

logger = logging.getLogger(__name__)


class ControllerBase(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """ """

    cv_tag: str = Field(
        ...,
    )
    cv_range: tuple[StateValue, StateValue] = Field(
        ...,
        description=(
            "Lower and upper bound of the manipulated variable, in that order. "
            "The unit is assumed to be the same unit as the mv_tag's unit. "
            "If you want to specify some other unit, consider changing the "
            "measured unit of mv_tag or making a conversion calculation separately."
        ),
    )
    ramp_rate: PerSeconds | None = Field(
        default=None,
        description=(
            "Optional ramp rate limit for the controller setpoint. "
            "Must be provided as a 'per time' unitized Quantity, where "
            "the numerator of the Quantity is assumed to be the unit of the "
        ),
    )
    sp_trajectory: Trajectory | None = Field(
        default=None, description="A Trajectory instance defining the setpoint (SP) over time."
    )
    cascade_controller: SerializeAsAny[ControllerBase | None] = Field(
        default=None,
        description=(
            "If provided, and when controller mode is CASCADE, the setpoint source "
            "of this controller will be provided by the provided cascade controller. "
        ),
    )
    mode: ControllerMode = Field(
        # default to CASCADE so highest loop gets used.
        # Highest loop with no cascade controller will automatically change to AUTO.
        default=ControllerMode.CASCADE,
        description=(
            "ControllerBase's mode - if AUTO, setpoint comes from the sp_trajectory provided. "
            "If CASCADE, setpoint comes from the cascade controller, if provided. "
            "If no cascade controller is provided, this mode will fall back to AUTO with a warning. "
        ),
    )
    period: Seconds = Field(
        default=1e-12,
        description=(
            "minimum execution period of the controller. Controlelr will execute "
            "as frequently as possible such that the time between execution is as "
            "close to this value as possible. "
        ),
    )
    _mode_manager: ControllerModeManager = PrivateAttr()
    _mv_trajectory: Trajectory = PrivateAttr()
    _mv_range: tuple[StateValue, StateValue] = PrivateAttr()
    _cv_getter: Callable[[], TagData] = PrivateAttr()
    _initialized: bool = PrivateAttr(default=False)
    _control_action: TagData = PrivateAttr()
    _sp_tag_info: TagInfo = PrivateAttr()
    _is_scalar: bool = PrivateAttr(default=False)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    def change_control_mode(self, mode: ControllerMode | str) -> None:
        """public facing method for mode changing."""
        mode = ControllerMode.from_value(mode)
        logger.info(
            "'%s' controller mode is changed from %s --> %s", self.cv_tag, self.mode.name, mode.name
        )
        self.mode = self._mode_manager.change_mode(mode)

    def commission(
        self,
        system: System,
        mv_getter: Callable[[], TagData],
        mv_range: tuple[StateValue, StateValue],
        mv_tag: str,
        mv_unit: UnitBase,
    ) -> bool:
        """Wire the controller into orchestrated quantities and validate modes.

        The container guarantees all referenced tags exist, so this method
        simply creates the getter/setter callables, configures cascade
        relationships, and promotes the controller into the highest valid
        operating mode for the supplied configuration.
        """
        logger.debug(f"Initializing '{self.cv_tag}' controller.")
        self._mv_range = mv_range
        # make the trajectory if not provided
        cv_tag_info = system.tag_store.get(self.cv_tag)
        if cv_tag_info is None:
            raise ControllerConfigurationError(
                f"Could not find the control variable '{self.cv_tag}'. "
                + "The control variable of a controller must be a MEASURED or CALCULATED result."
            )
        self._cv_getter = system.tag_store.make_converted_data_getter(self.cv_tag)
        if self.sp_trajectory is None:
            self.sp_trajectory = Trajectory(cv_tag_info.data.value, cv_tag_info.data.time)

        # commission the controllers starting from the outer loop first.
        if self.cascade_controller is not None:
            # mv tag of the cascade controller is the cv tag of the inner controller
            successful = self.cascade_controller.commission(
                system=system,
                mv_getter=self._cv_getter,
                mv_range=self.cv_range,
                mv_tag=self.cv_tag,
                mv_unit=cv_tag_info.unit,
            )
            if not successful:
                return False
        elif self.mode == ControllerMode.CASCADE:
            # if no cascade controller but mode is CASCADE, change to AUTO
            # typically this is done by the mode manager when the mode was actively
            # changed to CASCADE. However, during "commissioning", the mode defaults
            # to CASCADE, so we do this silently.
            self.mode = ControllerMode.AUTO

        # prepare sp and mv tag info first
        # sp tag info's data is initialized with the sp_trajectory's value
        # at the commissioning time.
        self._sp_tag_info = TagInfo(
            tag=f"{self.cv_tag}.sp",
            unit=cv_tag_info.unit,
            type="setpoint",
            description=f"setpoint for {self.cv_tag}",
        )
        self._sp_tag_info.data = TagData(
            time=system.time, value=self.sp_trajectory(system.time), ok=True
        )
        mv_data = mv_getter()
        self._control_action = mv_data
        self._mv_trajectory = Trajectory(y0=mv_data.value, t0=system.time)
        # make the mode manager
        self._mode_manager = ControllerModeManager(
            mode=self.mode,
            manual_mv_source=self._mv_trajectory,
            auto_sp_source=self.sp_trajectory,
            cascade_sp_source=self.cascade_controller,
            sp_getter=self._sp_tag_info.make_converted_data_getter(),
            cv_getter=self._cv_getter,
            mv_getter=mv_getter,
            cv_tag=self.cv_tag,
        )
        # now run post commission hook in case it is implemented
        successful = self._post_commission(system, mv_getter, mv_range, mv_tag, mv_unit)
        if not successful:
            return False
        self._initialized = True
        if np.isscalar(self._control_action.value):
            self._is_scalar = True
        return True

    def _post_commission(
        self,
        system: System,
        mv_getter: Callable[[], TagData],
        mv_range: tuple[StateValue, StateValue],
        mv_tag: str,
        mv_unit: UnitBase,
    ) -> bool:
        """hook for additional logic involving system after self.commission"""
        return True

    def update(self, t: Seconds) -> TagData:
        """Run one control-cycle update and return the applied MV value.

        The routine pulls the latest controlled-variable reading, sources the
        appropriate setpoint (local trajectory, cascade input, or tracking),
        evaluates the subclass control law, applies MV ramp limits, and writes
        the result back into the system if this is the final controller in the
        cascade chain.
        """
        if not self._initialized:
            raise RuntimeError(f"{self.cv_tag} controller not commissioned.")

        last_control_action = self._control_action

        if t - last_control_action.time < self.period:
            return last_control_action

        cv = self._cv_getter()
        sp = self._mode_manager.get_setpoint(t)
        self._sp_tag_info.data = sp

        if self.mode == ControllerMode.TRACKING:
            raise RuntimeError(
                f"'{self.cv_tag}' controller is in TRACKING mode yet its `.update` method was called. "
                + "This is a bug in the controller. Please report it."
            )
        elif self.mode == ControllerMode.MANUAL:
            return self._mode_manager.get_control_action(t)

        proceed = cv.ok and sp.ok
        if not proceed:
            last_control_action.ok = False
            return last_control_action

        control_output, successful = self._control_algorithm(t=t, cv=cv.value, sp=sp.value)
        if np.isnan(control_output) or not successful:
            logger.warning(
                f"'{self.cv_tag}' controller algorithm failed. "
                + f"Holding previous control action ({last_control_action.value}) "
            )
            last_control_action.ok = False
            return last_control_action

        ramp_output = self._apply_mv_ramp(
            t0=last_control_action.time,
            mv0=last_control_action.value,
            mv_target=control_output,
            t_now=t,
        )
        if self._is_scalar:
            ramp_output = min(max(ramp_output, self._mv_range[0]), self._mv_range[1])
        else:
            ramp_output = np.clip(ramp_output, self._mv_range[0], self._mv_range[1])
        self._control_action = TagData(time=t, value=ramp_output, ok=True)

        return self._control_action

    def _apply_mv_ramp(
        self, t0: Seconds, mv0: StateValue, mv_target: StateValue, t_now: Seconds
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
    ) -> tuple[StateValue, bool]:
        """The actual control algorithm. To be implemented by subclasses."""
        pass

    @property
    def sp_tag_info_dict(self) -> dict[str, TagInfo]:
        c = self
        sp_tag_info_dict = {self._sp_tag_info.tag: self._sp_tag_info}
        while c.cascade_controller is not None:
            c = c.cascade_controller
            sp_tag_info_dict.update(c.sp_tag_info_dict)
        return sp_tag_info_dict

    @property
    def t(self) -> Seconds:
        return self._control_action.time
