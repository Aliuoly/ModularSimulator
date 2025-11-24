from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Any
from collections.abc import Callable
from dataclasses import asdict
import importlib
from pydantic import BaseModel, PrivateAttr, Field, ConfigDict, SerializeAsAny
from modular_simulation.usables.control_system.trajectory import Trajectory
from modular_simulation.usables.control_system.mode_manager import ControlElementModeManager
from modular_simulation.usables.control_system.controller_mode import ControllerMode
from modular_simulation.validation.exceptions import ControllerConfigurationError
from modular_simulation.usables.control_system.controller_base import ControllerBase
from modular_simulation.utils.typing import Seconds, StateValue

if TYPE_CHECKING:
    from modular_simulation.framework import System
    from modular_simulation.measurables.process_model import ProcessModel
    from modular_simulation.usables.tag_info import TagData, TagInfo
import logging

logger = logging.getLogger(__name__)


def get_mv_setter(mv_tag_info: TagInfo, process: ProcessModel) -> Callable[[StateValue], None]:
    # search for the raw tag in case the mv tag is an alias
    control_element_name = mv_tag_info.raw_tag
    mv_state_metadata = process.state_metadata_dict.get(control_element_name)
    if mv_state_metadata is None:
        raise ControllerConfigurationError(
            f"Could not find control element '{mv_tag_info.tag} (raw tag = {mv_tag_info.raw_tag})' in process model."
        )
    return process.make_converted_setter(control_element_name, mv_tag_info.unit)


class ControlElement(BaseModel):
    """
    Interfaces with a process state, making it manipulatable.

    Think of this as a control valve placed on a process stream, though
    it is technically possible to "place" a control element on a
    non-stream state - doing so will allow direct manipulation of
    said states.

    More often, a controller is placed on a control element to
    achieve control objectives.
    """

    mv_tag: str = Field(
        ...,
        description=(
            "The tag of the CONTROLLED state corresponding to the "
            + "manipulated variable (MV) of this control element."
        ),
    )
    mv_trajectory: Trajectory | None = Field(
        default=None,
        description=(
            "A Trajectory instance defining the manipulated variable (MV) over time. "
            + "Only applicable if in mode MANUAL."
        ),
    )
    mv_range: tuple[StateValue, StateValue] = Field(
        ...,
        description=(
            "Lower and upper bound of the manipulated variable, in that order. "
            "The unit is assumed to be the same unit as the mv_tag's unit. "
            "If you want to specify some other unit, consider changing the "
            "measured unit of mv_tag or making a conversion calculation separately."
        ),
    )
    controller: SerializeAsAny[ControllerBase | None] = Field(
        default=None,
        description=(
            "If provided, and when controller mode is AUTO, the setpoint source "
            "of this controller will be provided by the provided controller. "
        ),
    )
    mode: ControllerMode = Field(
        default=ControllerMode.AUTO,
        description=(
            "control element's mode - if AUTO, output comes from the controller provided. "
            "If MANUAL, output comes from the mv trajectory. "
            "Reusing ControllerMode, which also has CASCADE and TRACKING - NOT applicable for control element. "
        ),
    )
    period: Seconds = Field(
        default=1e-12,
        description=(
            "minimum execution period of the controller. Controller will execute "
            "as frequently as possible such that the time between execution is as "
            "close to this value as possible. "
        ),
    )

    _mv_tag_info: TagInfo = PrivateAttr()
    _mv_setter: Callable[[StateValue], None] = PrivateAttr()
    _mode_manager: ControlElementModeManager = PrivateAttr()
    _initialized: bool = PrivateAttr(default=False)
    _is_scalar: bool = PrivateAttr(default=False)
    _control_action: TagData = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    def commission(self, system: System) -> bool:
        """Wire the control element into orchestrated quantities and validate modes.

        The container guarantees all referenced tags exist, so this method
        simply creates the getter/setter callables, configures cascade
        relationships, and promotes the controller into the highest valid
        operating mode for the supplied configuration.
        """
        logger.debug(f"Initializing '{self.mv_tag}' control element.")

        mv_tag_info = system.tag_store.get(self.mv_tag)
        if mv_tag_info is None:
            raise ControllerConfigurationError(
                f"Could not find the control element '{self.mv_tag}'. "
                + "The manipulated variable of a control elemnt must be a MEASURED CONTROLLED state."
            )
        self._mv_tag_info = mv_tag_info
        # commission the control element & its controllers (if any)
        if self.controller is not None:
            mv_getter = mv_tag_info.make_converted_data_getter()
            if not self.controller.commission(
                system=system,
                mv_getter=mv_getter,
                mv_range=self.mv_range,
                mv_tag=self.mv_tag,
                mv_unit=self._mv_tag_info.unit,
            ):
                return False  # failed to initialize due to controller commissioning error
        else:
            # silently force mode to MANUAL in case it was AUTO during initialization
            # if no controller is provided
            self.mode = ControllerMode.MANUAL
        # make the control action default to starting measurement
        # also set the mv trajectory if not provided.
        self._control_action = mv_tag_info.data
        if self.mv_trajectory is None:
            self.mv_trajectory = Trajectory(y0=self._control_action.value, t0=system.time)

        self._mv_setter = get_mv_setter(mv_tag_info, system.process_model)
        self._mode_manager = ControlElementModeManager(
            mode=self.mode,
            manual_mv_source=self.mv_trajectory,
            auto_mv_source=self.controller,
            mv_getter=mv_tag_info.make_converted_data_getter(),
            mv_tag=self.mv_tag,
        )

        self._is_scalar = np.isscalar(self._control_action.value)
        self._initialized = True
        return True

    def change_control_mode(self, mode: ControllerMode | str) -> None:
        """public facing method for mode changing."""
        mode = ControllerMode.from_value(mode)
        logger.debug(
            "control element '%s' mode is changed from %s --> %s",
            self.mv_tag,
            self.mode.name,
            mode.name,
        )
        self.mode = self._mode_manager.change_mode(mode=mode)

    def update(self, t: Seconds) -> TagData:
        if not self._initialized or self.mv_trajectory is None:
            raise RuntimeError("Control element is not initialized.")
        additional_info = ""
        control_action = self._mode_manager.get_control_action(t)
        if control_action.ok:
            self._mv_setter(control_action.value)
            self._mv_tag_info.data = control_action
        else:
            additional_info += (
                "control action getter output marked bad, keeping previous control action. "
            )
        logger.debug(
            "control element '%s' updated to %f. %s",
            self.mv_tag,
            self._mv_tag_info.data.value,
            additional_info,
        )

        return self._mv_tag_info.data

    def save(self) -> dict[str, Any]:
        """Persist minimal configuration and runtime state."""

        controller_payload = (
            self.controller.save() if self.controller is not None else None
        )

        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "config": self.model_dump(exclude={"controller"}),
            "controller": controller_payload,
            "state": self._save_runtime_state(),
        }

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "ControlElement":
        """Recreate a control element instance from serialized configuration."""

        module = importlib.import_module(payload["module"])
        control_element_cls = getattr(module, payload["type"])
        if not issubclass(control_element_cls, cls):
            raise TypeError(f"{control_element_cls} is not a subclass of {cls}")

        controller_payload = payload.get("controller")
        controller = ControllerBase.load(controller_payload) if controller_payload else None

        config = dict(payload["config"])
        config["controller"] = controller

        control_element = control_element_cls(**config)
        control_element._load_runtime_state(payload.get("state") or {})
        return control_element

    def _save_runtime_state(self) -> dict[str, Any]:
        """Hook for subclasses to extend saved runtime state."""

        return {
            "mode": self.mode.name,
            "mv": asdict(self._control_action),
            "mv_tag": self.mv_tag,
        }

    def _load_runtime_state(self, state: dict[str, Any]) -> None:  # pyright: ignore[reportUnusedParameter]
        """Subclass hook to restore any persisted runtime state."""

        if "mode" in state:
            self.mode = ControllerMode[state["mode"]]

    @property
    def mv_tag_info(self) -> TagInfo:
        return self._mv_tag_info

    @property
    def controller_sp_tag_info_dict(self) -> dict[str, TagInfo]:
        if self.controller is None:
            return {}
        return self.controller.sp_tag_info_dict
