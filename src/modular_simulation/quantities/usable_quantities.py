from typing import Dict, List, TYPE_CHECKING, Iterable
from pydantic import  PrivateAttr, BaseModel, ConfigDict, Field, model_validator
from modular_simulation.validation.exceptions import SensorConfigurationError, CalculationConfigurationError, ControllerConfigurationError
from modular_simulation.usables.tag_info import TagInfo
from functools import cached_property
import warnings
import logging
from textwrap import dedent
if TYPE_CHECKING:
    from modular_simulation.usables.sensor import Sensor
    from modular_simulation.usables.calculation import Calculation
    from modular_simulation.control_system.controller import Controller
    from modular_simulation.quantities.measurable_quantities import MeasurableQuantities
    from modular_simulation.usables.tag_info import TagData

logger = logging.getLogger(__name__)


class UsableQuantities(BaseModel):
    """
    1. Defines how measurements and calculations are obtained through
        - measurement_definition: Dict[str, Sensor]
        - calculation_definition: Dict[str, Calculation]
    2. Saves the current snapshot of measurements and calculations
        - results: Dict[str, Any]
    """
    sensors: List["Sensor"] = Field(
        default_factory = list,
    )
    calculations: List["Calculation"] = Field(
        default_factory = list,
    )
    controllers: List["Controller"] = Field(
        default_factory = list
    )
    measurable_quantities: "MeasurableQuantities" = Field(
        ...
    )

    _usable_results: Dict[str, "TagData"] = PrivateAttr(default_factory=dict)
    _tag_infos: List[TagInfo] = PrivateAttr(default_factory=list)
    _initialized: bool = PrivateAttr(default = False)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra = 'forbid')

    def model_post_init(self, context):
        # construct the _tag_infos
        for sensor in self.sensors:
            self._tag_infos.append(sensor._tag_info)
            if sensor.unit is None:
                sensor.unit = self.measurable_quantities.tag_unit_info[sensor.measurement_tag]
                sensor._tag_info.unit = sensor.unit
        for calculation in self.calculations:
            for output_tag_info in calculation._output_tag_info_dict.values():
                self._tag_infos.append(output_tag_info)
        # controller needs cv info so have to pass in tag_infos
        for controller in self.controllers:
            while controller.cascade_controller is not None:
                self._tag_infos.append(controller._make_sp_tag_info_and_mv_range(self._tag_infos))
                controller = controller.cascade_controller
            self._tag_infos.append(controller._make_sp_tag_info_and_mv_range(self._tag_infos))

    @model_validator(mode = 'after')
    def _validate(self):
        exception_group = []
        exception_group.extend(self._validate_duplicate_tag())
        exception_group.extend(self._validate_sensors_resolvable())
        exception_group.extend(self._validate_calculations_resolvable())
        
        if len(exception_group) > 0:
            raise ExceptionGroup(
                "errors encountered during usable quantity instantiation:", 
                exception_group
                )
        return self
    
    def _initialize(self):
        for sensor in self.sensors:
            sensor._initialize(self.measurable_quantities)
        for calculation in self.calculations:
            calculation._initialize(self._tag_infos)
        for controller in self.controllers:
            controller._initialize(self._tag_infos, self, self.measurable_quantities.control_elements)
        self._initialized = True
    def _validate_sensors_resolvable(self):
        exception_group = []
        available_measurement_tags = self.measurable_quantities.tag_list
        unavailable_measurement_tags = []

        for sensor in self.sensors:
            measurement_tag = sensor.measurement_tag
            if measurement_tag not in available_measurement_tags:
                unavailable_measurement_tags.append(measurement_tag)

        if len(unavailable_measurement_tags) > 0:
            exception_group.append(
                SensorConfigurationError(
                    "The following measurement tag(s) are not defined in measurable quantities: "
                    f"{', '.join(unavailable_measurement_tags)}."
                )
            )
        return exception_group
    
    def _validate_duplicate_tag(self):
        exception_group = []
        all_tags = [tag_info.tag for tag_info in self._tag_infos]
        seen_tags = []
        duplicate_tags = []
        for tag in all_tags:
            if tag in seen_tags:
                duplicate_tags.append(tag)
            else:
                seen_tags.append(duplicate_tags)
        if len(duplicate_tags) > 0:
            exception_group.append(
                SensorConfigurationError(
                    "The following duplicate tag(s) found: "
                    f"{', '.join(duplicate_tags)}."
                )
            )
        return exception_group
    
    def _validate_calculations_resolvable(self):
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
    
    def _validate_controllers_resolvable(self):
        exception_group = []

        all_tags = [tag_info.tag for tag_info in self._tag_infos]
        missing_cv_tags = []
        missing_mv_tags = []
        for controller in self.controllers:
            if controller.cv_tag not in all_tags:
                missing_cv_tags.append(controller.cv_tag)
            if controller.mv_tag not in all_tags:
                missing_mv_tags.append(controller.mv_tag)
        if len(missing_cv_tags) > 0:
            exception_group.append(
                ControllerConfigurationError(
                    "The following controlled variables are not available as either measurements or calculations: "
                    f"{', '.join(missing_cv_tags)}"
                )
            )
        if len(missing_mv_tags) > 0:
            exception_group.append(
                ControllerConfigurationError(
                    "The following manipulated variables are not available as either measurements or calculations:"
                    f"{', '.join(missing_mv_tags)}"
                )
            )
        # 3. check that final control element designate controlled variables (i.e., most-inner loops)
        #       are defined as control elements in the system measurables. 
        improper_ce_tags = []
        available_ce_tags = self.measurable_quantities.control_elements.tag_list
        for tag in [c.mv_tag for c in self.controllers]: # ignoring the cascade controllers, these mvs must be control elements
            if tag not in available_ce_tags:
                improper_ce_tags.append(tag)
        
        if len(improper_ce_tags) > 0:
            exception_group.append(
                ControllerConfigurationError(
                    "The following controlled variables are not defined as system control elements:"
                    f"{', '.join(improper_ce_tags)}."
                )
            )
        return exception_group

    @cached_property
    def tag_list(self) -> Iterable[str]:
        return [tag_info.tag for tag_info in self._tag_infos]

    @property
    def tag_infos(self) -> List[TagInfo]:
        return self._tag_infos
    
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


