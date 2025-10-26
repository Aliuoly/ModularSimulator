"""Operator-facing interface wiring sensors, calculations, and controllers."""
from __future__ import annotations

from functools import cached_property
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from modular_simulation.interfaces.tag_info import TagInfo
from modular_simulation.validation.exceptions import (
    CalculationConfigurationError,
    ControllerConfigurationError,
    SensorConfigurationError,
)

if TYPE_CHECKING:
    from modular_simulation.core.dynamic_model import DynamicModel
    from modular_simulation.interfaces.calculations.calculation_base import CalculationBase
    from modular_simulation.interfaces.controllers.controller_base import ControllerBase
    from modular_simulation.interfaces.sensors.sensor_base import SensorBase

logger = logging.getLogger(__name__)


class ModelInterface(BaseModel):
    """Collection of operator-facing components connected to a :class:`DynamicModel`."""

    sensors: list["SensorBase"] = Field(default_factory=list)
    calculations: list["CalculationBase"] = Field(default_factory=list)
    controllers: list["ControllerBase"] = Field(default_factory=list)

    _initialized: bool = PrivateAttr(default=False)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> "ModelInterface":
        exception_group = []
        exception_group.extend(self._validate_duplicate_tag())
        exception_group.extend(self._validate_calculation_inputs())
        exception_group.extend(self._validate_controller_tags())

        if exception_group:
            raise ExceptionGroup(
                "errors encountered during model interface instantiation:",
                exception_group,
            )
        return self

    def _validate_duplicate_tag(self) -> list[SensorConfigurationError]:
        exception_group: list[SensorConfigurationError] = []
        seen_tags: list[str] = []
        duplicate_tags: list[str] = []
        for tag in (tag_info.tag for tag_info in self._tag_infos):
            if tag in seen_tags and ".sp" not in tag:
                duplicate_tags.append(tag)
            else:
                seen_tags.append(tag)
        if duplicate_tags:
            exception_group.append(
                SensorConfigurationError(
                    "The following duplicate tag(s) found: "
                    + ", ".join(sorted(set(duplicate_tags)))
                    + "."
                )
            )
        return exception_group

    def _validate_calculation_inputs(self) -> list[CalculationConfigurationError]:
        exception_group: list[CalculationConfigurationError] = []
        available_tags = {tag_info.tag for tag_info in self._tag_infos}
        for calculation in self.calculations:
            missing_tags = [
                info.tag
                for info in calculation._input_tag_info_dict.values()
                if info.tag not in available_tags
            ]
            if missing_tags:
                exception_group.append(
                    CalculationConfigurationError(
                        "The following input tag(s) required by "
                        f"'{calculation.__class__.__name__}' are not available: "
                        + ", ".join(sorted(set(missing_tags)))
                        + "."
                    )
                )
        return exception_group

    def _validate_controller_tags(self) -> list[ControllerConfigurationError]:
        exception_group: list[ControllerConfigurationError] = []
        available_tags = {tag_info.tag for tag_info in self._tag_infos}
        missing_cv_tags: list[str] = []
        missing_mv_tags: list[str] = []
        for controller in self.controllers:
            if controller.cv_tag not in available_tags:
                missing_cv_tags.append(controller.cv_tag)
            if controller.mv_tag not in available_tags:
                missing_mv_tags.append(controller.mv_tag)
        if missing_cv_tags:
            exception_group.append(
                ControllerConfigurationError(
                    "The following controlled variables are not available as "
                    "either measurements or calculations: "
                    + ", ".join(sorted(set(missing_cv_tags)))
                    + "."
                )
            )
        if missing_mv_tags:
            exception_group.append(
                ControllerConfigurationError(
                    "The following manipulated variables are not available as "
                    "either measurements or calculations: "
                    + ", ".join(sorted(set(missing_mv_tags)))
                    + "."
                )
            )
        return exception_group

    def _initialize(self, dynamic_model: "DynamicModel") -> None:
        for sensor in self.sensors:
            sensor._initialize(dynamic_model)
        for calculation in self.calculations:
            calculation._initialize(self._tag_infos)
        for controller in self.controllers:
            controller._initialize(
                self._tag_infos,
                self,
                dynamic_model.control_elements,
            )
        self._initialized = True

    def update(self, t: float) -> None:
        """Refresh all sensors, calculations, and controllers for the time ``t``."""

        if not self._initialized:
            raise RuntimeError(
                "model interface is not initialized. Ensure this interface instance "
                "belongs to a System, and the system has been constructed."
            )
        for sensor in self.sensors:
            sensor.measure(t)
        for calculation in self.calculations:
            calculation.calculate(t)
        for controller in self.controllers:
            controller.update(t)

    @cached_property
    def _tag_infos(self) -> list[TagInfo]:
        infos: list[TagInfo] = []
        for sensor in self.sensors:
            infos.append(sensor._tag_info)
        for calculation in self.calculations:
            infos.extend(calculation._output_tag_info_dict.values())
        for controller in self.controllers:
            active = controller
            while active.cascade_controller is not None:
                infos.append(active._make_sp_tag_info(infos))
                active = active.cascade_controller
            infos.append(active._make_sp_tag_info(infos))
        return infos

    @cached_property
    def tag_infos(self) -> list[TagInfo]:
        return self._tag_infos

    @cached_property
    def tag_list(self) -> list[str]:
        return [tag_info.tag for tag_info in self._tag_infos]


__all__ = ["ModelInterface"]
