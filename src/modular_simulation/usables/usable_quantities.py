from typing import TYPE_CHECKING
from pydantic import  PrivateAttr, BaseModel, ConfigDict, Field, model_validator
from modular_simulation.validation.exceptions import (
    SensorConfigurationError, 
    CalculationConfigurationError, 
    ControllerConfigurationError
)
from modular_simulation.usables.tag_info import TagInfo
from functools import cached_property
import logging
if TYPE_CHECKING:
    from modular_simulation.usables.sensors.sensor_base import SensorBase
    from modular_simulation.usables.calculations.calculation_base import CalculationBase
    from modular_simulation.usables.controllers.controller_base import ControllerBase
    from modular_simulation.measurables.measurable_quantities import MeasurableQuantities

logger = logging.getLogger(__name__)


class UsableQuantities(BaseModel):
    """
    Represents a collection of sensors, calculations, and controllers for managing usable quantities in a simulation system
    
    :var sensors: List of sensors, which provide measured tags to the system
    :vartype sensors: list[SensorBase]
    :var calculations: List of calculations, uses measured tags and other calculated tags 
                        to provide calculated tags to the system
    :vartype calculations: list[CalculationBase]
    :var controllers: List of controllers, which uses measured tags and configured setpoints
                        to control ControlElements of the system.
    :vartype controllers: list[ControllerBase]
    :var _initialized: Indicates whether the usable quantities have been initialized
    :vartype _initialized: bool
    """
    sensors: list["SensorBase"] = Field(
        default_factory = list,
    )
    calculations: list["CalculationBase"] = Field(
        default_factory = list,
    )
    controllers: list["ControllerBase"] = Field(
        default_factory = list
    )

    _initialized: bool = PrivateAttr(default = False)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra = 'forbid')

    @model_validator(mode = 'after')
    def _validate(self):
        exception_group = []
        exception_group.extend(self._validate_duplicate_tag())
        exception_group.extend(self._validate_calculations_inputs_are_available())
        exception_group.extend(self._validate_controller_mv_cv_are_measured())
        
        if len(exception_group) > 0:
            raise ExceptionGroup(
                "errors encountered during usable quantity instantiation:", 
                exception_group
                )
        return self
    
    def _validate_duplicate_tag(self) -> list[SensorConfigurationError]:
        exception_group = []
        all_tags = [tag_info.tag for tag_info in self._tag_infos]
        seen_tags = []
        duplicate_tags = []
        for tag in all_tags:
            # ignore case where multiple controllers control the same element
            # this could happen when one controls when below SP
            # another another controls when above SP
            # e.g., air conditioner with both warm and cool air sources
            # controlling room temperature. 
            if tag in seen_tags and ".sp" not in tag: 
                duplicate_tags.append(tag)
            else:
                seen_tags.append(tag)
        if len(duplicate_tags) > 0:
            exception_group.append(
                SensorConfigurationError(
                    "The following duplicate tag(s) found: "
                    f"{', '.join(duplicate_tags)}."
                )
            )
        return exception_group
    
    def _validate_calculations_inputs_are_available(self) -> list[CalculationConfigurationError]:
        exception_group = []
        all_tags = [tag_info.tag for tag_info in self._tag_infos]
        for calculation in self.calculations:
            missing_input_tags = []
            for input_tag_info in calculation._input_tag_info_dict.values():
                if input_tag_info.tag not in all_tags:
                    missing_input_tags.append(input_tag_info.tag)
                if len(missing_input_tags) > 0:
                    exception_group.append(
                        CalculationConfigurationError(
                            f"The following input tag(s) required by '{calculation.__class__.__name__}' "
                            f"are not available: {', '.join(missing_input_tags)}. "
                        )
                    )
        return exception_group
    
    def _validate_controller_mv_cv_are_measured(self) -> list[ControllerConfigurationError]:
        exception_group = []
        all_tags = [tag_info.tag for tag_info in self._tag_infos]
        missing_cv_tags, missing_mv_tags = [], []
        for controller in self.controllers:
            if controller.cv_tag not in all_tags:
                missing_cv_tags.append(controller.cv_tag)
            if controller.mv_tag not in all_tags:
                missing_mv_tags.append(controller.mv_tag)
        error_template = (
            "The following {type} variables are not available "
            "as either measurements or calculations: {tags}"
        )
        if len(missing_cv_tags) > 0:
            exception_group.append(
                ControllerConfigurationError(
                    error_template.format(type = "controlled", tags = ', '.join(missing_cv_tags))
                )
            )
        if len(missing_mv_tags) > 0:
            exception_group.append(
                ControllerConfigurationError(
                    error_template.format(type = "manipulated", tags = ', '.join(missing_mv_tags))
                )
            )
        
        return exception_group
        
    def _initialize(self, measurable_quantities: "MeasurableQuantities"):
        for sensor in self.sensors:
            sensor._initialize(measurable_quantities)
        for calculation in self.calculations:
            calculation._initialize(self._tag_infos)
        for controller in self.controllers:
            controller._initialize(self._tag_infos, self, measurable_quantities.control_elements)
        self._initialized = True

    def update(self, t: float) -> None:
        """
        updates the measurements and performs calculations with the latest info.
        Results are automatically linked to the controllers that depend on these 
        but a dictionary of results is still returned for tracking. 
        """
        if not self._initialized:
            raise RuntimeError(
                "usable quantity is not initialized. Ensure this quantity instance "
                "belongs to a System, and the system has been constructed. "
            )
        # fyi, this looping method is faster than caching the sensor callable
        # and tag in a dictionary and bypassing attribute lookup.
        # It is also faster than callaing dict.update on a newly constructed
        # dictionary made using comprehension.
        for sensor in self.sensors: 
            sensor.measure(t)
        for calculation in self.calculations:
            calculation.calculate(t) 
        for controller in self.controllers:
            controller.update(t)

    @cached_property
    def _tag_infos(self) -> list[TagInfo]:
        infos = []
        for sensor in self.sensors:
            infos.append(sensor._tag_info)
        for calculation in self.calculations:
            for output_tag_info in calculation._output_tag_info_dict.values():
                infos.append(output_tag_info)
        # controller needs cv info so have to pass in tag_infos
        for controller in self.controllers:
            while controller.cascade_controller is not None:
                infos.append(controller._make_sp_tag_info(infos))
                controller = controller.cascade_controller
            infos.append(controller._make_sp_tag_info(infos))
        return infos

    @cached_property
    def tag_infos(self):
        return self._tag_infos
    
    @cached_property
    def tag_list(self) -> list[str]:
        return [tag_info.tag for tag_info in self._tag_infos]

