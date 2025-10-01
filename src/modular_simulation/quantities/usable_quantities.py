from typing import Dict, List, TYPE_CHECKING, Iterable
from pydantic import  PrivateAttr, BaseModel, ConfigDict, Field, model_validator
from modular_simulation.validation.exceptions import SensorConfigurationError, CalculationConfigurationError
from functools import cached_property
import warnings
import logging
from textwrap import dedent
if TYPE_CHECKING:
    from modular_simulation.usables.sensor import Sensor
    from modular_simulation.usables.calculation import Calculation
    from modular_simulation.quantities.measurable_quantities import MeasurableQuantities
    from modular_simulation.usables.time_value_quality_triplet import TimeValueQualityTriplet

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
    measurable_quantities: "MeasurableQuantities" = Field(
        ...
    )

    _usable_results: Dict[str, "TimeValueQualityTriplet"] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra = 'forbid')

    @model_validator(mode = 'after')
    def check_duplicate_tags_and_resolvable_sensors(self):

        exception_group = []

        # 1. check for duplicate tags in sensor definitions
        #    and that each tag exists in measurable quantities
        available_measurement_tags = self.measurable_quantities.tag_list
        defined_calculation_tags = []
        for c in self.calculations:
            defined_calculation_tags.extend(c.output_tags)
        seen_measurement_tags = []
        seen_alias_tags = []
        seen_calculation_tags = []
        unavailable_measurement_tags = []
        duplicate_measurement_tags = []
        duplicate_alias_tags = []
        duplicate_calculation_tags_in_sensors = []
        duplicate_calculation_tags_in_calculations = []
        
        for sensor in self.sensors:
            measurement_tag = sensor.measurement_tag
            alias_tag = sensor.alias_tag
            if measurement_tag in seen_measurement_tags:
                duplicate_measurement_tags.append(measurement_tag)
            if alias_tag in seen_alias_tags:
                duplicate_alias_tags.append(alias_tag)
            if measurement_tag not in available_measurement_tags:
                unavailable_measurement_tags.append(measurement_tag)
            seen_measurement_tags.append(measurement_tag)
            seen_alias_tags.append(alias_tag)

        if len(duplicate_measurement_tags) > 0:
            exception_group.append(
                SensorConfigurationError(
                    "The following duplicate measurement tag(s) found: "
                    f"{', '.join(duplicate_measurement_tags)}."
                )
            )
        if len(duplicate_alias_tags) > 0:
            exception_group.append(
                SensorConfigurationError(
                    "The following duplicate measurement alias tag(s) found: "
                    f"{', '.join(duplicate_alias_tags)}."
                )
            )
        if len(unavailable_measurement_tags) > 0:
            exception_group.append(
                SensorConfigurationError(
                    "The following measurement tag(s) are not defined in measurable quantities: "
                    f"{', '.join(unavailable_measurement_tags)}."
                )
            )

        # 2A. check for duplicate tags in calculation definitions
        # 2B. check that all required input tags in calculation is defined as
        #       either measurements (through sensors) or calculations
        defined_alias_tags = [s.alias_tag for s in self.sensors]
        for calculation in self.calculations:
            for tag in calculation.output_tags:
                if tag in defined_alias_tags:
                    duplicate_calculation_tags_in_sensors.append(tag)    
                if tag in seen_calculation_tags:
                    duplicate_calculation_tags_in_calculations.append(tag)
                meas_input_tags = calculation.measured_input_tags
                missing_tags = [meas_tag for meas_tag in meas_input_tags if meas_tag not in defined_alias_tags]
                if len(missing_tags) > 0:
                    exception_group.append(
                        CalculationConfigurationError(
                            f"The following measurement tag(s) required by '{calculation.__class__.__name__}' "
                            f"is not available: {', '.join(missing_tags)}. "
                        )
                    )
                calc_input_tags = calculation.calculated_input_tags
                missing_tags = [calc_tag for calc_tag in calc_input_tags if calc_tag not in defined_calculation_tags]
                if len(missing_tags) > 0:
                    exception_group.append(
                        CalculationConfigurationError(
                            f"The following calculation tag(s) required by '{calculation.__class__.__name__}' "
                            f"is not available: {', '.join(missing_tags)}. "
                        )
                    )
                seen_calculation_tags.append(tag)

        if len(duplicate_calculation_tags_in_calculations):
            exception_group.append(
                CalculationConfigurationError(
                    "The following duplicate calculation tag(s) found: "
                    f"{', '.join(duplicate_calculation_tags_in_calculations)}."
                )
            )
        if len(duplicate_calculation_tags_in_sensors) > 0:
            exception_group.append(
                CalculationConfigurationError(
                    "The following calculation tag(s) are also defined as sensor measurement tags: "
                    f"{', '.join(duplicate_calculation_tags_in_calculations)}."
                )
            )

        # 3. raise error if necessary
        
        if len(exception_group) > 0:
            measurement_and_alias_tags = []
            for sensor in self.sensors:
                if sensor.alias_tag == sensor.measurement_tag:
                    measurement_and_alias_tags.append(sensor.measurement_tag)
                else:
                    measurement_and_alias_tags.append(f"{sensor.measurement_tag} (alias {sensor.alias_tag})")
            msg = dedent("""Additional info for the sub-exceptions:
                Defined measurable tags are: {0}.
                Defined measured tags are  : {1}.
                Defined calculated tags are: {2}.
            """).format(
                ", ".join(available_measurement_tags),
                ", ".join(measurement_and_alias_tags),
                ", ".join(defined_calculation_tags),
            )
            raise ExceptionGroup(msg, exception_group)
        
        # 4. raise warning if necessary if no errors
        if len(seen_measurement_tags) == 0:
            warnings.warn(
                "No measurement configured. Assuming you want to observe state evolution from some condition. "
            )

        # 5. initialize sensor and calculations once validated
        for sensor in self.sensors:
            sensor._initialize(self.measurable_quantities)
        for calculation in self.calculations:
            calculation._initialize(self)
        self.update(t = 0)

        return self
    
    def update(self, t: float) -> Dict[str, "TimeValueQualityTriplet"]:
        """
        updates the measurements and performs calculations with the latest info.
        Results are automatically linked to the controllers that depend on these 
        but a dictionary of results is still returned for tracking. 
        """
        # fyi, this looping method is faster than caching the sensor callable
        # and tag in a dictionary and bypassing attribute lookup.
        # It is also faster than callaing dict.update on a newly constructed
        # dictionary made using comprehension.
        for sensor in self.sensors: 
            self._usable_results[sensor.alias_tag] = sensor.measure(t)
        for calculation in self.calculations:
            self._usable_results.update(calculation.calculate(t)) # this one returns a dictionary so use .update
            
        return self._usable_results

    @cached_property
    def tag_list(self) -> Iterable[str]:
        return_list = [s.alias_tag for s in self.sensors]
        for c in self.calculations:
            return_list.extend(c.output_tags)
        return return_list


