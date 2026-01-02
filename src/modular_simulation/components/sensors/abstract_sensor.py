from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, override
import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr
from modular_simulation.validation.exceptions import (
    SensorConfigurationError,
)
from modular_simulation.components.point import Point, DataValue
from modular_simulation.components.abstract_component import (
    AbstractComponent,
    ComponentUpdateResult,
)
from modular_simulation.utils.typing import Seconds
from modular_simulation.components.calculations.point_metadata import PointMetadata, TagType
from modular_simulation.utils.metadata_extraction import extract_unique_metadata

if TYPE_CHECKING:
    from modular_simulation.framework.system import System


class AbstractSensor(AbstractComponent):
    """Abstract base class for all sensors in the modular simulation framework.
    All sensors are assumed to have a single configuration field named "measurement_tag",
    representing the tag name of the measured process state.
    """

    measurement_tag: str = Field(..., description="Tag name of the measured process state.")
    alias_tag: str = Field(default="", description="Alias tag name for the sensor output.")
    unit: str = Field(default="", description="The unit of the sensor measurement.")
    description: str = Field(default="", description="Description of the sensor.")

    # Noise and error modeling
    coefficient_of_variance: float = Field(
        default=0.0, ge=0.0, description="Coefficient of variance for measurement noise."
    )
    faulty_probability: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Probability of the sensor being faulty."
    )
    faulty_aware: bool = Field(
        default=True, description="Whether the control system is aware of sensor faults."
    )
    instrument_range: tuple[float, float] = Field(
        default=(-float("inf"), float("inf")), description="Operating range of the sensor."
    )
    random_seed: int | None = Field(default=None, description="Random seed for reproducibility.")

    _point: Point = PrivateAttr()
    _measurement_getter: Any = PrivateAttr()
    _generator: np.random.Generator = PrivateAttr()
    _is_faulty: bool = PrivateAttr(default=False)

    # Timing logic
    _t: Seconds = PrivateAttr(default=-1.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @override
    def model_post_init(self, context: Any) -> None:
        if self.alias_tag == "":
            self.alias_tag = self.measurement_tag

        self._generator = np.random.default_rng(self.random_seed)

        # Initialize Point
        self._point = Point(
            tag=self.alias_tag,
            unit=self.unit,
            type="measured",
            description=self.description,
            _raw_tag=self.measurement_tag,
        )

    # -------- AbstractComponent Interface --------

    @override
    def _install(self, system: System) -> list[Exception]:
        """Bind the sensor to the system's point registry."""
        exceptions: list[Exception] = []
        try:
            # Check if the measurement tag corresponds to a process state first
            # This avoids circular dependencies if the sensor alias matches the state name
            if self.measurement_tag in system.process_model.state_metadata_dict:
                from astropy.units import Unit

                metadata = system.process_model.state_metadata_dict[self.measurement_tag]
                source_unit = Unit(metadata.unit)
                target_unit = Unit(self.unit)
                converter = source_unit.get_converter(target_unit)

                def getter() -> DataValue:
                    # Read directly from process model
                    val = getattr(system.process_model, self.measurement_tag)
                    # We assume process model state is always "ok" if it exists,
                    # or at least we treat it as the raw truth.
                    return DataValue(value=converter(val), time=system.time, ok=True)

                self._measurement_getter = getter
            else:
                self._measurement_getter = system.point_registry.make_converted_data_getter(
                    tag=self.measurement_tag, target_unit=self.unit
                )
        except KeyError as e:
            exceptions.append(SensorConfigurationError(str(e)))
        except AttributeError as e:
            # Conversion error or getattr error
            exceptions.append(SensorConfigurationError(f"Failed to access process state: {e}"))
        except Exception as e:  # Catch unit errors etc
            exceptions.append(SensorConfigurationError(f"Configuration error: {e}"))

        if not exceptions:
            # Perform initial measurement
            try:
                # We can't call update() here because it checks _initialized which is not set yet by the base class
                # So we manually call _update
                result = self._update(t=system.time)
                if result.exceptions:
                    exceptions.extend(result.exceptions)
                elif np.isnan(self.point.data.value):
                    exceptions.append(
                        SensorConfigurationError(
                            f"'{self.name}' sensor resulted in nan during commissioning."
                        )
                    )
            except Exception as e:
                exceptions.append(e)

        return exceptions

    @override
    def _should_update(self, t: Seconds) -> bool:
        # Default behavior: update every step if not overridden
        return True

    @override
    def _update(self, t: Seconds) -> ComponentUpdateResult:
        """Execute the measurement logic."""
        self._t = t
        raw_data = self._measurement_getter()

        # Allow subclass to process the measurement (add noise, delay, etc.)
        measurement = raw_data

        # Layer on fault logic
        if self.faulty_probability > 0:
            self._is_faulty = self._generator.random() < self.faulty_probability

        if self._is_faulty:
            measurement.ok = False

        self._point.data = measurement
        return ComponentUpdateResult(data_value=measurement, exceptions=[])

    @override
    def _get_configuration_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @override
    def _get_runtime_state_dict(self) -> dict[str, Any]:
        return {"t": self._t, "is_faulty": self._is_faulty}

    @classmethod
    @override
    def _load_configuration(cls, data: dict[str, Any]) -> "AbstractSensor":
        return cls(**data)

    @override
    def _load_runtime_state(self, state: dict[str, Any]) -> None:
        self._t = state.get("t", -1.0)
        self._is_faulty = state.get("is_faulty", False)

    # -------- Properties --------

    @property
    def point(self) -> Point:
        return self._point

    @property
    def point_metadata_dict(self) -> dict[str, PointMetadata]:
        return {
            self.measurement_tag: PointMetadata(
                tag=self.measurement_tag,
                unit=self.unit,
                type=TagType.INPUT,
                description=self.description,
            ),
            self.alias_tag: PointMetadata(
                tag=self.alias_tag,
                unit=self.unit,
                type=TagType.OUTPUT,
                description=self.description,
            ),
        }
