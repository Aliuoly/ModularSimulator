from __future__ import annotations  # without this cant type hint the cascade controller properly
from abc import abstractmethod
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, override
from dataclasses import asdict
from astropy.units import UnitBase
import numpy as np
from .controller_mode import ControllerMode
from pydantic import PrivateAttr, Field, ConfigDict, SerializeAsAny
from modular_simulation.components.point import DataValue, Point
from .trajectory import Trajectory
from modular_simulation.validation.exceptions import ControllerConfigurationError
from modular_simulation.components.abstract_component import (
    AbstractComponent,
    ComponentUpdateResult,
)
from modular_simulation.utils.typing import Seconds, StateValue, PerSeconds
from .mode_manager import ControllerModeManager

if TYPE_CHECKING:
    from modular_simulation.framework import System
import logging

logger = logging.getLogger(__name__)


class AbstractController(AbstractComponent):
    """Abstract base class for all controllers in the modular simulation framework."""

    cv_tag: str = Field(..., description="The tag of the controlled variable (CV).")
    cv_range: tuple[StateValue, StateValue] = Field(
        ...,
        description=(
            "Lower and upper bound of the controlled variable, in that order. "
            "The unit is assumed to be the same unit as the cv_tag's unit. "
        ),
    )
    ramp_rate: PerSeconds | None = Field(
        default=None,
        description=(
            "Optional ramp rate limit for the controller setpoint. "
            "Must be provided as a 'per time' unitized Quantity."
        ),
    )
    sp_trajectory: Trajectory | None = Field(
        default=None, description="A Trajectory instance defining the setpoint (SP) over time."
    )
    cascade_controller: SerializeAsAny[AbstractController | None] = Field(
        default=None,
        description=(
            "If provided, and when controller mode is CASCADE, the setpoint source "
            "of this controller will be provided by the provided cascade controller. "
        ),
    )
    mode: ControllerMode = Field(
        default=ControllerMode.CASCADE,
        description=(
            "Controller's mode - if AUTO, setpoint comes from the sp_trajectory provided. "
            "If CASCADE, setpoint comes from the cascade controller, if provided. "
        ),
    )
    period: Seconds = Field(
        default=1e-12,
        description=("minimum execution period of the controller."),
    )

    _mode_manager: ControllerModeManager = PrivateAttr()
    _mv_trajectory: Trajectory = PrivateAttr()
    _mv_range: tuple[StateValue, StateValue] = PrivateAttr()
    _cv_getter: Callable[[], DataValue] = PrivateAttr()
    _control_action: DataValue = PrivateAttr()
    _sp_point: Point = PrivateAttr()
    _is_scalar: bool = PrivateAttr(default=False)
    _system: "System" = PrivateAttr()
    _initialized: bool = PrivateAttr(default=False)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    # -------- AbstractComponent Interface --------

    @abstractmethod
    def _control_algorithm(
        self,
        t: Seconds,
        cv: StateValue,
        sp: StateValue,
    ) -> tuple[StateValue, bool]:
        """The actual control algorithm. To be implemented by subclasses."""
        pass

    @override
    def _install(self, system: System) -> list[Exception]:
        """Base commissioning for the controller.

        Resolves CV points, sets up SP points, commissions cascade controllers.
        Element-specific wiring happens later in wire_to_element().
        """
        exceptions: list[Exception] = []
        logger.debug(f"Commissioning '{self.cv_tag}' controller.")

        # 1. Get CV point from system's point registry
        cv_point = system.point_registry.get(self.cv_tag)
        if cv_point is None:
            exceptions.append(
                ControllerConfigurationError(
                    f"Could not find the control variable '{self.cv_tag}'. "
                    + "The control variable of a controller must be a MEASURED or CALCULATED result."
                )
            )
            return exceptions

        # 2. Get CV getter from system's point registry
        self._cv_getter = system.point_registry.make_converted_data_getter(self.cv_tag)

        # 3. Initialize SP trajectory if necessary
        if self.sp_trajectory is None:
            self.sp_trajectory = Trajectory(y0=cv_point.data.value, t0=cv_point.data.time)

        # 4. Commission cascade controllers recursively if it exists
        if self.cascade_controller is not None:
            cascade_exceptions = self.cascade_controller.install(system)
            if cascade_exceptions:
                exceptions.extend(cascade_exceptions)
                return exceptions
        elif self.mode == ControllerMode.CASCADE:
            # Fallback to AUTO if CASCADE requested but no controller provided
            self.mode = ControllerMode.AUTO

        # Prepare setpoint point
        self._sp_point = Point(
            tag=f"{self.cv_tag}.sp",
            unit=cv_point.unit,
            type="setpoint",
            description=f"setpoint for {self.cv_tag}",
        )
        self._sp_point.data = DataValue(
            time=system.time, value=self.sp_trajectory(system.time), ok=True
        )

        self._system = system

        if not exceptions:
            self._initialized = True

        return exceptions

    @override
    def _should_update(self, t: Seconds) -> bool:
        # control_action time is the time of the *last* action
        return t >= (self._control_action.time + self.period)

    @override
    def _update(self, t: Seconds) -> ComponentUpdateResult:
        """Run one control-cycle update and return result."""
        try:
            data_value = self._control_update(t)
            return ComponentUpdateResult(data_value=data_value, exceptions=[])
        except Exception as e:
            return ComponentUpdateResult(data_value=self._control_action, exceptions=[e])

    # -------- Wiring Logic --------

    def wire_to_element(
        self,
        system: System,
        mv_getter: Callable[[], DataValue],
        mv_range: tuple[StateValue, StateValue],
        mv_tag: str,
        mv_unit: UnitBase,
    ) -> bool:
        """Wire the controller to its manipulated variable (MV) source/sink.

        This is called by ControlElement during its commissioning.
        """
        self._mv_range = mv_range
        mv_data = mv_getter()
        # Initialize control action with 'never' timestamp to ensure immediate update at t=0
        self._control_action = DataValue(value=mv_data.value, time=float("-inf"), ok=mv_data.ok)
        self._mv_trajectory = Trajectory(y0=mv_data.value, t0=system.time)

        # Recurse for cascade controllers
        if self.cascade_controller is not None:
            # For cascade, MV of outer is CV of inner
            if not self.cascade_controller.wire_to_element(
                system=system,
                mv_getter=self._cv_getter,
                mv_range=self.cv_range,
                mv_tag=self.cv_tag,
                mv_unit=self._sp_point.unit,  # Sp point has CV unit
            ):
                return False

        # Create mode manager
        self._mode_manager = ControllerModeManager(
            mode=self.mode,
            manual_mv_source=self._mv_trajectory,
            auto_sp_source=self.sp_trajectory,
            cascade_sp_source=self.cascade_controller,
            sp_getter=self._sp_point.make_converted_data_getter(),
            cv_getter=self._cv_getter,
            mv_getter=mv_getter,
            cv_tag=self.cv_tag,
        )

        # Call subclass hook
        if not self.post_install(system, mv_getter, mv_range, mv_tag, mv_unit):
            return False

        if np.isscalar(self._control_action.value):
            self._is_scalar = True

        return True

    def post_install(
        self,
        system: System,
        mv_getter: Callable[[], DataValue],
        mv_range: tuple[StateValue, StateValue],
        mv_tag: str,
        mv_unit: UnitBase,
    ) -> bool:
        """Hook for additional logic involving system after wire_to_element."""
        return True

    # -------- Control Logic --------

    def change_control_mode(self, mode: ControllerMode | str) -> None:
        """Public facing method for mode changing."""
        mode = ControllerMode.from_value(mode)
        logger.info(
            "'%s' controller mode is changed from %s --> %s", self.cv_tag, self.mode.name, mode.name
        )
        self.mode = self._mode_manager.change_mode(mode)

    def _control_update(self, t: Seconds) -> DataValue:
        """Internal control update method."""
        if not self._initialized:
            raise RuntimeError(f"Controller '{self.cv_tag}' has not been commissioned.")

        last_control_action = self._control_action

        # Base _should_update already handles the period, but we keep a safety check here if needed
        # though it's redundant now.

        cv = self._cv_getter()
        sp = self._mode_manager.get_setpoint(t)
        self._sp_point.data = sp

        if self.mode == ControllerMode.TRACKING:
            raise RuntimeError(
                f"'{self.cv_tag}' controller is in TRACKING mode yet its `.update` method was called. "
            )
        elif self.mode == ControllerMode.MANUAL:
            return self._mode_manager.get_control_action(t)

        proceed = cv.ok and sp.ok
        if not proceed:
            last_control_action.ok = False
            return last_control_action

        control_output, successful = self._control_algorithm(t=t, cv=cv.value, sp=sp.value)
        if (
            np.isnan(control_output).any()
            if isinstance(control_output, np.ndarray)
            else np.isnan(control_output) or not successful
        ):
            logger.warning(f"'{self.cv_tag}' controller algorithm failed. Holding previous value.")
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
        self._control_action = DataValue(time=t, value=ramp_output, ok=True)

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

    @override
    def _get_configuration_dict(self) -> dict[str, Any]:
        config = self.model_dump(exclude={"cascade_controller"})
        if self.cascade_controller is not None:
            config["cascade_controller"] = self.cascade_controller.save()
        return config

    @override
    def _get_runtime_state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.name,
            "sp": asdict(self._sp_point.data),
            "control_action": asdict(self._control_action),
        }

    @classmethod
    @override
    def _load_configuration(cls, data: dict[str, Any]) -> "AbstractController":
        config = dict(data)

        # Handle Controller
        if "cascade_controller" in config and config["cascade_controller"] is not None:
            # We use cls.load instead of AbstractController.load to maintain type
            # but actually AbstractController.load is safer for dispatch
            cascade_controller = AbstractController.load(config["cascade_controller"])
            config["cascade_controller"] = cascade_controller

        return cls(**config)

    @override
    def _load_runtime_state(self, state: dict[str, Any]) -> None:
        if "mode" in state:
            self.mode = ControllerMode[state["mode"]]
        # sp and control_action are usually cold-started or updated by first step
        # but could be restored here if needed.

    # -------- Properties --------

    @property
    def sp_point_dict(self) -> dict[str, Point]:
        sp_point_dict = {self._sp_point.tag: self._sp_point}
        if self.cascade_controller is not None:
            sp_point_dict.update(self.cascade_controller.sp_point_dict)
        return sp_point_dict

    @property
    def t(self) -> Seconds:
        return self._control_action.time
