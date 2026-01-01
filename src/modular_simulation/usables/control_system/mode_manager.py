from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from collections.abc import Callable
from .controller_mode import ControllerMode
from modular_simulation.usables.point import DataValue

if TYPE_CHECKING:
    from .abstract_controller import AbstractController
    from .trajectory import Trajectory
    from modular_simulation.usables.abstract_component import ComponentUpdateResult
import logging

logger = logging.getLogger(__name__)


class ControlElementModeManager:
    manual_mv_source: Trajectory
    """
    For a control element:
        manual_mv_source corresponds to the mv_trajectory. 
    """
    auto_mv_source: AbstractController | None
    """
    For a control element:
        auto_mv_source corresponds to its associated controller's `update` method,
        which provides said controller's control output. 
        The control element, when in AUTO mode, simply passes the controller's control output
        to its CONTROLLED state using its `_mv_setter` callable private attribute. 
    """
    mode: ControllerMode
    mv_getter: Callable[[], DataValue]
    mv_tag: str

    def __init__(
        self,
        mode: ControllerMode,
        manual_mv_source: Trajectory,
        auto_mv_source: AbstractController | None,
        mv_getter: Callable[[], DataValue],
        mv_tag: str,
    ):
        self.mode = mode
        self.manual_mv_source = manual_mv_source
        self.auto_mv_source = auto_mv_source
        self.mv_getter = mv_getter
        self.mv_tag = mv_tag

    def change_mode(
        self,
        mode: ControllerMode,
    ) -> ControllerMode:
        if mode == ControllerMode.MANUAL:
            return self._change_mode_to_manual()
        elif mode == ControllerMode.AUTO:
            return self._change_mode_to_auto()
        else:
            raise ValueError(
                "Error in mode change for control element '%s'. "
                + "For control elements, only MANUAL and AUTO mode are accepted. ",
                self.mv_tag,
            )

    def _change_mode_to_manual(self) -> ControllerMode:
        """
        When changing to MANUAL mode, the setpoint source becomes the control element's mv_trajectory.
        Bumpless transfer is achieved by updating said trajectory with the last output's value.
        """
        self.mode = ControllerMode.MANUAL
        if self.auto_mv_source is not None:
            self.auto_mv_source.change_control_mode(ControllerMode.TRACKING)

        mv_data = self.mv_getter()
        if np.isnan(mv_data.value):
            raise RuntimeError(
                "NAN encoutered during mode change for control element '%s'.",
                self.mv_tag,
            )
        # bumpless transfer
        _ = self.manual_mv_source.set(value=mv_data.value, t=mv_data.time)
        return self.mode

    def _change_mode_to_auto(self) -> ControllerMode:
        """
        When changing to AUTO mode, the setpoint source becomes the controller's `update` method.
        The controller's mode will also update from TRACKING to AUTO mode.

        The bumpless transfer logic is delegated to the controller's change_control_mode.

        Note that changing to AUTO instead of MANUAL mode is a design choice, not out of necessity.
        """
        if self.auto_mv_source is None:
            logger.info(
                "Tried to change mode for Controller element '%s' to AUTO but failed:"
                + " no controller assigned. Returning to MANUAL mode.",
                self.mv_tag,
            )
            return self._change_mode_to_manual()
        else:
            self.mode = ControllerMode.AUTO
            self.auto_mv_source.change_control_mode(ControllerMode.AUTO)
            return self.mode

    def get_control_action(self, t: float) -> DataValue:
        if self.mode == ControllerMode.MANUAL:
            return DataValue(time=t, value=self.manual_mv_source(t), ok=True)
        elif self.mode == ControllerMode.AUTO and self.auto_mv_source is not None:
            result = self.auto_mv_source.update(t)
            return result.data_value  # type: ignore
        else:
            raise RuntimeError("Error in mode change for control element '%s'.")


class ControllerModeManager:
    """
    Internal helper class that manages the mode of a control element or controller.

    Handles logic for mode changes, bumpless transfer, and setpoint source selection.

    doesnt have a tracking_sp_source since the controller's cv getter provides it.
    getters are used for retrieving the data instead of the tag info objects
    during bumpless transfer to avoid any possible modification of the underlying
    tag info. This is just me being clear that only data retrieval is needed
    by the mode manager.
    """

    manual_mv_source: Trajectory
    """
    For a control element:
        manual_mv_source corresponds to the mv_trajectory. 
        Functionally equivalent to the inner controller's AUTO mode, 
        but kept separate for conformity to standard control system behavior, 
        where it is possible for controller A -> controller B cascade loop to have 
            1. A in CASCADE mode, B in MANUAL mode,
                here, B's control action = A's SP is dictated by a trajectory (B.mv_trajectory)
            2. A in AUTO mode, B in TRACKING mode 
                here, A's SP is dictated by a trajectory (A.sp_trajectory)
        even though both options above are functionally equivalent. 
    """
    auto_sp_source: Trajectory
    """
    For a control element:
        auto_sp_source corresponds to its associated controller's `update` method,
        which provides said controller's control output. 
        The control element, when in AUTO mode, simply passes the controller's control output
        to its CONTROLLED state using its `_mv_setter` callable private attribute. 
    """
    cascade_sp_source: AbstractController | None
    """
    for a controller:
        corresponds to the cascade controller's `update` method, which provides
        said cascade controller's control output.
    """
    mode: ControllerMode
    mv_getter: Callable[[], DataValue]
    cv_getter: Callable[[], DataValue]
    sp_getter: Callable[[], DataValue]
    cv_tag: str

    def __init__(
        self,
        mode: ControllerMode,
        manual_mv_source: Trajectory,
        auto_sp_source: Trajectory,
        cascade_sp_source: AbstractController | None,
        mv_getter: Callable[[], DataValue],
        cv_getter: Callable[[], DataValue],
        sp_getter: Callable[[], DataValue],
        cv_tag: str,
    ):
        self.mode = mode
        self.manual_mv_source = manual_mv_source
        self.auto_sp_source = auto_sp_source
        self.cascade_sp_source = cascade_sp_source
        self.mv_getter = mv_getter
        self.cv_getter = cv_getter
        self.sp_getter = sp_getter
        self.cv_tag = cv_tag

    def change_mode(self, mode: ControllerMode):
        """Changes the mode of the controller."""
        if mode == ControllerMode.TRACKING:
            return self._change_mode_to_tracking()
        elif mode == ControllerMode.MANUAL:
            return self._change_mode_to_manual()
        elif mode == ControllerMode.AUTO:
            return self._change_mode_to_auto()
        elif mode == ControllerMode.CASCADE:
            return self._change_mode_to_cascade()
        else:
            raise RuntimeError("Error in mode change.")

    def _change_mode_to_tracking(self) -> ControllerMode:
        """
        Nothing is needed for tracking mode other than changing the mode.
        """
        self.mode = ControllerMode.TRACKING
        return self.mode

    def _change_mode_to_manual(self) -> ControllerMode:
        """
        When changing to MANUAL mode, the setpoint source becomes the control element's mv_trajectory.
        Bumpless transfer is achieved by updating said trajectory with the last output's value.
        """
        self.mode = ControllerMode.MANUAL
        mv_data = self.mv_getter()
        _ = self.manual_mv_source.set(value=mv_data.value, t=mv_data.time)
        return self.mode

    def _change_mode_to_auto(self) -> ControllerMode:
        """
        When changing to AUTO mode, the setpoint source becomes the local sp_trajectory.
        However, to ensure bumpless transfer, the local sp_trajectory
        is updated with the last setpoint, wherever it came from.
        """
        self.mode = ControllerMode.AUTO
        if self.cascade_sp_source is not None:
            self.cascade_sp_source.change_control_mode(ControllerMode.TRACKING)
        sp_data = self.sp_getter()
        _ = self.auto_sp_source.set(value=sp_data.value, t=sp_data.time)
        return self.mode

    def _change_mode_to_cascade(self) -> ControllerMode:
        """
        When changing to CASCADE mode, the setpoint source becomes the cascade controller's
        .update method. At the same time, the cascade controller's mode, previously
        in TRACKING mode (unless initializing), is automatically changed to AUTO.
        The mode change to AUTO also serves as a bumpless transfer logic - see
        the _change_mode_to_auto method.
        If no cascade controller is available, will fall back to AUTO mode with a warning.
        """
        if self.cascade_sp_source is None:
            logging.info(
                f"Attemped to change '{self.cv_tag}' controller mode to CASCADE; however, "
                + "no cascade controller was provided. Falling back to AUTO mode. "
            )
            return self._change_mode_to_auto()
        else:
            self.mode = ControllerMode.CASCADE
            self.cascade_sp_source.change_control_mode(ControllerMode.AUTO)
            return self.mode

    def get_setpoint(self, t: float) -> DataValue:
        if self.mode == ControllerMode.TRACKING:
            return self.cv_getter()
        elif self.mode == ControllerMode.AUTO:
            return DataValue(time=t, value=self.auto_sp_source(t), ok=True)
        elif self.mode == ControllerMode.CASCADE and self.cascade_sp_source is not None:
            result = self.cascade_sp_source.update(t)
            return result.data_value  # type: ignore
        else:
            raise RuntimeError(
                f"Invalid mode '{self.mode}' for get_setpoint. This is a bug. Please report it."
            )

    def get_control_action(self, t: float) -> DataValue:
        if self.mode == ControllerMode.MANUAL:
            return DataValue(time=t, value=self.manual_mv_source(t), ok=True)
        else:
            raise RuntimeError(
                "Not in MANUAL mode but get_control_action is called. This is a bug. Please report it."
            )
