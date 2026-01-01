from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, override
from collections.abc import Callable
import importlib
import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr
from modular_simulation.validation.exceptions import (
    CalculationDefinitionError,
    CalculationConfigurationError,
)
from modular_simulation.components.point import Point, DataValue
from modular_simulation.components.abstract_component import AbstractComponent, ComponentUpdateResult
from modular_simulation.utils.typing import StateValue, Seconds
from modular_simulation.utils.metadata_extraction import extract_unique_metadata
from modular_simulation.components.calculations.point_metadata import PointMetadata, TagType

if TYPE_CHECKING:
    from modular_simulation.framework.system import System


class AbstractCalculation(AbstractComponent):
    """Abstract base class for calculations in the modular simulation framework.

    Inputs and outputs tag names are expected to be annotated with the following info:
    1. unit
    2. description - optional
    3. input/output/constant type
    e.g., input_one_tag: Annotated[str, TagAnnotation(TagType.INPUT), Unit('m'), 'this is input one']
    """

    # -----construction time defined-----
    _field_metadata_dict: dict[str, PointMetadata] = PrivateAttr()
    _output_point_dict: dict[str, Point] = PrivateAttr()

    # -----initialization (wiring) time defined-----
    _input_data_getters: dict[str, Callable[[], DataValue]] = PrivateAttr(default_factory=dict)
    _input_data_dict: dict[str, DataValue] = PrivateAttr(default_factory=dict)
    _input_value_dict: dict[str, StateValue] = PrivateAttr(default_factory=dict)
    _t: Seconds = PrivateAttr(default=0)
    _system: "System" = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def model_post_init(self, context: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
        self._field_metadata_dict = {
            field_name: extract_unique_metadata(
                field, PointMetadata, field_name, CalculationDefinitionError
            )
            for field_name, field in self.__class__.model_fields.items()
            if field_name != "name"
        }
        output_tag_metadata_pairs: dict[str, PointMetadata] = {
            getattr(self, field_name): metadata
            for field_name, metadata in self._field_metadata_dict.items()
            if metadata.type == TagType.OUTPUT
        }
        self._output_point_dict = {
            output_tag: Point(
                tag=output_tag,
                unit=metadata.unit,
                type="calculated",
                description=metadata.description,
            )
            for output_tag, metadata in output_tag_metadata_pairs.items()
        }

    # -------- AbstractComponent Interface --------

    @override
    def _initialize(self, system: System) -> list[Exception]:
        """Link calculation inputs to tag info instances and create callables.

        Returns a list of exceptions (empty if successful).
        """
        exceptions: list[Exception] = []

        # Call pre-commissioning hook
        pre_error, successful = self._pre_commissioning_hook(system)
        if not successful and pre_error is not None:
            exceptions.append(pre_error)
            return exceptions

        input_tag_metadata_pairs: dict[str, PointMetadata] = {
            getattr(self, field): metadata
            for field, metadata in self._field_metadata_dict.items()
            if metadata.type == TagType.INPUT
        }

        for input_tag, input_metadata in input_tag_metadata_pairs.items():
            try:
                self._input_data_getters[input_tag] = system.tag_store.make_converted_data_getter(
                    tag=input_tag, target_unit=input_metadata.unit
                )
            except KeyError as e:
                exceptions.append(CalculationConfigurationError(str(e)))
                return exceptions

        self._system = system

        if not exceptions:
            # Perform initial calculation
            try:
                result = self._update(t=system.time)
                if result.exceptions:
                    exceptions.extend(result.exceptions)

                # if somehow the calculation returned NAN, something went wrong
                for point in self._output_point_dict.values():
                    if np.isnan(point.data.value):
                        exceptions.append(
                            CalculationConfigurationError(
                                f"'{self.name}' calculation resulted in nan during commissioning."
                            )
                        )
            except Exception as e:
                exceptions.append(e)

        return exceptions

    @override
    def _should_update(self, t: Seconds) -> bool:
        # Calculations generally should update on every step or when inputs change.
        # For now, we assume every step for simplicity, similar to original behavior.
        # A more complex check could verify input timestamps.
        return True

    @override
    def _update(self, t: Seconds) -> ComponentUpdateResult:
        """Execute the calculation and return result with all output DataValues."""
        # Original _calculate logic moved here? Or keep _calculate as internal?
        # The user said refactor components to implement public methods directly.
        # But keeping _calculate separates the "algorithm" from the "component plumbing".
        # I'll keep _calculate invocation here for cleanliness, but remove the check inside _calculate if possible,
        # or just call it.

        try:
            output_dict = self._calculate(t)
            return ComponentUpdateResult(data_value=output_dict, exceptions=[])
        except Exception as e:
            return ComponentUpdateResult(data_value=self.outputs, exceptions=[e])

    @override
    def _get_configuration_dict(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Convert the calculation's configuration to a dictionary."""
        return self.model_dump()

    @override
    def _get_runtime_state_dict(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Convert the calculation's runtime state to a dictionary."""
        return self._save_runtime_state()

    def _save_runtime_state(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Hook for subclasses to extend saved runtime state."""
        return {}

    @classmethod
    @override
    def _load_configuration(cls, data: dict[str, Any]) -> "AbstractCalculation":  # pyright: ignore[reportExplicitAny]
        """Create a calculation from configuration dictionary."""
        return cls(**data)

    @override
    def _load_runtime_state(self, state: dict[str, Any]) -> None:  # pyright: ignore[reportExplicitAny, reportUnusedParameter]
        """Subclass hook to restore any persisted runtime state."""
        pass

    # -------- hooks --------

    def _pre_commissioning_hook(
        self,
        system: System,  # pyright: ignore[reportUnusedParameter]
    ) -> tuple[CalculationConfigurationError | None, bool]:
        """Hook for any pre-commissioning steps."""
        return None, True

    # -------- Calculation Logic --------

    def retrieve_specific_input(self, tag_name: str) -> DataValue:
        """Get the DataValue for a given input tag name."""
        if tag_name not in self._input_data_dict:
            raise CalculationConfigurationError(
                f"'{self.name}' calculation's input tag '{tag_name}' was not found amongst the wired inputs. "
                + "Make sure the calculation has been commissioned to a system and the tag name is correct."
            )
        return self._input_data_dict[tag_name]

    def retrieve_specific_output(self, tag_name: str) -> DataValue:
        """Get the DataValue for a given output tag name."""
        if tag_name not in self._output_point_dict:
            raise CalculationConfigurationError(
                f"'{self.name}' calculation's output tag '{tag_name}' was not found amongst the defined outputs. "
                + "Make sure the tag name is correct."
            )
        return self._output_point_dict[tag_name].data

    def _update_input_triplets(self) -> None:
        tag_data_dict = self._input_data_dict
        for tag_name, tag_data_getter in self._input_data_getters.items():
            tag_data_dict[tag_name] = tag_data_getter()

    def _update_input_values(self) -> dict[str, StateValue]:
        value_dict = self._input_value_dict
        for tag_name, tag_data in self._input_data_dict.items():
            value_dict[tag_name] = tag_data.value
        return value_dict

    @property
    def ok(self) -> bool:
        """Whether or not the calculation quality is ok.

        If any of the inputs are not ok, the calculation is also not ok.
        """
        possible_faulty_inputs_oks = [
            input_value.ok for input_value in self._input_data_dict.values()
        ]
        return any(possible_faulty_inputs_oks) if possible_faulty_inputs_oks else True

    @abstractmethod
    def _calculation_algorithm(
        self, t: Seconds, inputs_dict: dict[str, StateValue]
    ) -> dict[str, StateValue]:
        pass

    def _calculate(self, t: Seconds) -> dict[str, DataValue]:
        """Internal calculation method."""
        if not self._initialized:
            raise RuntimeError(
                "Tried to call 'calculate' before the system orchestrated the various quantities. "
                + "Make sure this calculation is part of a system and the system has been commissioned."
            )
        self._update_input_triplets()
        value_dict = self._update_input_values()
        outputs_dict = self._calculation_algorithm(
            t=t,
            inputs_dict=value_dict,
        )
        output_data_dict: dict[str, DataValue] = {}
        for output_tag, output_value in outputs_dict.items():
            tag_data = DataValue(time=t, value=output_value, ok=self.ok)
            self._output_point_dict[output_tag].data = tag_data
            output_data_dict[output_tag] = tag_data
        return output_data_dict

    # -------- Properties --------

    @property
    def point_metadata_dict(self) -> dict[str, PointMetadata]:
        return {
            getattr(self, field): metadata
            for field, metadata in self._field_metadata_dict.items()
            if metadata.type != TagType.CONSTANT
        }

    @property
    def t(self) -> dict[str, Point]:
        return self._output_point_dict

    @property
    def outputs(self) -> dict[str, DataValue]:
        return {tag: point.data for tag, point in self._output_point_dict.items()}

    @property
    def output_point_dict(self) -> dict[str, Point]:
        """Public getter for the private _output_point_dict attribute."""
        return self._output_point_dict
