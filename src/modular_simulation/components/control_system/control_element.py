from __future__ import annotations
from typing import TYPE_CHECKING, Any, override
import numpy as np
import importlib
from dataclasses import asdict
from pydantic import PrivateAttr, ConfigDict, Field
from modular_simulation.components.point import Point, DataValue
from modular_simulation.validation.exceptions import (
    ControlElementConfigurationError,
    ControllerConfigurationError,
)
from modular_simulation.components.abstract_component import (
    AbstractComponent,
    ComponentUpdateResult,
)
from modular_simulation.utils.typing import Seconds, StateValue
from .controller_mode import ControllerMode
from .mode_manager import ControlElementModeManager

from .trajectory import Trajectory
from .abstract_controller import AbstractController
import logging

if TYPE_CHECKING:
    from modular_simulation.framework.system import System

logger = logging.getLogger(__name__)


class ControlElement(AbstractComponent):
    """A control element (e.g. a valve, pump, or heater) that can be controlled by a controller."""

    mv_tag: str = Field(..., description="Tag name of the manipulated variable (MV).")
    mv_trajectory: Trajectory = Field(
        ..., description="A Trajectory instance for manual control of the element."
    )
    mv_range: tuple[StateValue, StateValue] = Field(
        ..., description="The physical range of the manipulated variable."
    )
    controller: AbstractController | None = Field(
        default=None, description="The controller assigned to this element."
    )
    mode: ControllerMode = Field(
        default=ControllerMode.MANUAL,
        description="Initialization mode for the control element (AUTO or MANUAL).",
    )
    period: Seconds = Field(
        default=1e-12, description="The minimum execution period of the control element."
    )

    _mv_point: Point = PrivateAttr()
    _mv_setter: Any = PrivateAttr()
    _mode_manager: ControlElementModeManager = PrivateAttr()
    _last_update_time: Seconds = PrivateAttr(default=float("-inf"))

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        validate_assignment=False,
    )

    # -------- AbstractComponent Interface --------

    @override
    def _initialize(self, system: System) -> list[Exception]:
        """Bind the control element to the system and wire its controller."""
        exceptions: list[Exception] = []
        # Try to find the point in TagStore.
        # If it's a process state, System registers it with a specific prefix.
        mv_point = None
        if self.mv_tag in system.process_model.state_metadata_dict:
            metadata = system.process_model.state_metadata_dict[self.mv_tag]
            # Reconstruct the prefix used by System.py
            # TODO: This coupling to System's naming convention is fragile.
            # Ideally TagStore handles this or System provides a helper.
            prefixed_tag = f"(raw, {metadata.type.name})_{self.mv_tag}"
            mv_point = system.tag_store.get(prefixed_tag)

        if mv_point is None:
            # Fallback 1: Direct lookup
            mv_point = system.tag_store.get(self.mv_tag)

        if mv_point is None:
            # Fallback 2: Search for suffix match (handling System's internal naming)
            # This handles cases where metadata.type.name might differ or be casing dependent
            possible_matches = [
                t
                for t in system.tag_store
                if t.endswith(f"_{self.mv_tag}") and t.startswith("(raw,")
            ]
            if len(possible_matches) == 1:
                mv_point = system.tag_store.get(possible_matches[0])

        if mv_point is None:
            exceptions.append(
                ControlElementConfigurationError(
                    f"Could not find the manipulated variable '{self.mv_tag}'."
                )
            )
            return exceptions

        self._mv_point = mv_point

        # Determine if MV maps to a process model state (direct control)
        # or just a registry point (e.g. calculation input)
        try:
            if self.mv_tag in system.process_model.state_metadata_dict:
                from astropy.units import Unit

                metadata = system.process_model.state_metadata_dict[self.mv_tag]
                target_unit = Unit(metadata.unit)
                # If mv_point came from a sensor, use its unit, otherwise use metadata unit
                source_unit = Unit(self._mv_point.unit) if self._mv_point.unit else target_unit

                converter = source_unit.get_converter(target_unit)

                def setter(data: DataValue) -> None:
                    # Write directly to process model
                    val = float(converter(data.value))
                    setattr(system.process_model, self.mv_tag, val)
                    # Also update the point for history/visibility
                    self._mv_point.data = data

                self._mv_setter = setter
            else:
                self._mv_setter = system.tag_store.make_converted_data_setter(self.mv_tag)

        except Exception as e:
            exceptions.append(ControlElementConfigurationError(f"Failed to bind MV setter: {e}"))
            return exceptions

        # Initialize internal state
        if np.isscalar(mv_point.data.value):
            self._is_scalar = True

        # initialize the control element & its controllers (if any)
        if self.controller is not None:
            if "mode" not in self.model_fields_set:
                self.mode = ControllerMode.AUTO

            # Phase 1: Base component commissioning
            controller_exceptions = self.controller.initialize(system)
            if controller_exceptions:
                exceptions.extend(controller_exceptions)
                return exceptions

            # Phase 2: Element-specific wiring (commissioning)
            mv_getter = mv_point.make_converted_data_getter()
            if not self.controller.wire_to_element(
                system=system,
                mv_getter=mv_getter,
                mv_range=self.mv_range,
                mv_tag=self.mv_tag,
                mv_unit=self._mv_point.unit,
            ):
                exceptions.append(
                    ControllerConfigurationError(
                        f"Controller for '{self.mv_tag}' failed to wire to element."
                    )
                )
                return exceptions
        else:
            # silently force mode to MANUAL in case it was AUTO during commissioning
            self.mode = ControllerMode.MANUAL

        self._mode_manager = ControlElementModeManager(
            mode=self.mode,
            manual_mv_source=self.mv_trajectory,
            auto_mv_source=self.controller,
            mv_getter=mv_point.make_converted_data_getter(),
            mv_tag=self.mv_tag,
        )

        if not exceptions:
            self._initialized = True

        return exceptions

    @override
    def _should_update(self, t: Seconds) -> bool:
        start_time = self._last_update_time
        return t >= (start_time + self.period)

    @override
    def _update(self, t: Seconds) -> ComponentUpdateResult:
        """Run one control update cycle and return result."""
        try:
            # Update internal timer
            self._last_update_time = t

            # Original logic from _control_update moved here
            control_action = self._mode_manager.get_control_action(t)

            if control_action.ok:
                if self._is_scalar:
                    control_action.value = min(
                        max(control_action.value, self.mv_range[0]), self.mv_range[1]
                    )
                else:
                    control_action.value = np.clip(
                        control_action.value, self.mv_range[0], self.mv_range[1]
                    )

            self._mv_setter(control_action)
            return ComponentUpdateResult(data_value=control_action, exceptions=[])
        except Exception as e:
            return ComponentUpdateResult(data_value=self._mv_point.data, exceptions=[e])

    @override
    def _load_runtime_state(self, state: dict[str, Any]) -> None:
        if "mode" in state:
            self.mode = ControllerMode[state["mode"]]
        if "mv" in state:
            # Optionally restore mv point data if needed, but usually it's driven by PM
            pass

    # -------- Control Logic --------

    def change_control_mode(self, mode: ControllerMode | str) -> None:
        """Change the control mode of the element."""
        mode = ControllerMode.from_value(mode)
        logger.info(
            "'%s' control element mode is changed from %s --> %s",
            self.mv_tag,
            self.mode.name,
            mode.name,
        )
        self.mode = self._mode_manager.change_mode(mode)

    @override
    def _get_configuration_dict(self) -> dict[str, Any]:
        config = self.model_dump(exclude={"controller", "mv_trajectory"})
        config["mv_trajectory"] = self.mv_trajectory.save()
        if self.controller is not None:
            config["controller"] = self.controller.save()
        return config

    @override
    def _get_runtime_state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.name,
            "mv": asdict(self._mv_point.data),
        }

    @classmethod
    @override
    def _load_configuration(cls, data: dict[str, Any]) -> "ControlElement":
        config = dict(data)

        # Handle Trajectory
        if "mv_trajectory" in config:
            from .trajectory import Trajectory

            config["mv_trajectory"] = Trajectory.load(config["mv_trajectory"])

        # Handle Controller
        if "controller" in config and config["controller"] is not None:
            from .abstract_controller import AbstractController

            config["controller"] = AbstractController.load(config["controller"])

        return cls(**config)

    # -------- Serialization (Legacy) --------

    # -------- Properties --------

    @property
    def controller_sp_point_dict(self) -> dict[str, Point]:
        if self.controller is None:
            return {}
        return self.controller.sp_point_dict


ControlElement.model_rebuild()
