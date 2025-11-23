from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, override
from collections.abc import Callable
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from modular_simulation.validation.exceptions import (
    CalculationDefinitionError,
    CalculationConfigurationError,
)
from modular_simulation.usables.tag_info import TagInfo, TagData
from modular_simulation.utils.typing import StateValue, Seconds
from modular_simulation.utils.metadata_extraction import extract_unique_metadata
from .tag_metadata import TagMetadata, TagType

if TYPE_CHECKING:
    from modular_simulation.framework.system import System


class CalculationBase(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """
    inputs and outputs tag names are expected to be annotated with the following info
    1. unit
    2. description - optional
    3. input/output/constant type
    e.g., input_one_tag: Annotated[str, TagAnnotation(TagType.INPUT), Unit('m'), 'this is input one']
    """

    name: str | None = Field(default=None, description="Name of the calculation - optional.")

    # -----construction time defined-----
    _field_metadata_dict: dict[str, TagMetadata] = PrivateAttr()
    _output_tag_info_dict: dict[str, TagInfo] = PrivateAttr()

    # -----initialization (wiring) time defined-----
    _input_data_getters: dict[str, Callable[[], TagData]] = PrivateAttr(default_factory=dict)
    _input_data_dict: dict[str, TagData] = PrivateAttr(default_factory=dict)
    _input_value_dict: dict[str, StateValue] = PrivateAttr(default_factory=dict)
    _initialized: bool = PrivateAttr(default=False)
    _t: Seconds = PrivateAttr(default=0)
    model_config = ConfigDict(arbitrary_types_allowed=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def model_post_init(self, context: Any):  # pyright: ignore[reportExplicitAny, reportAny]
        if self.name is None:
            self.name = self.__class__.__name__

        self._field_metadata_dict = {
            field_name: extract_unique_metadata(
                field, TagMetadata, field_name, CalculationDefinitionError
            )
            for field_name, field in self.__class__.model_fields.items()
            if field_name != "name"
        }
        output_tag_metadata_pairs: dict[str, TagMetadata] = {
            getattr(self, field_name): metadata
            for field_name, metadata in self._field_metadata_dict.items()
            if metadata.type == TagType.OUTPUT
        }
        self._output_tag_info_dict = {
            output_tag: TagInfo(
                tag=output_tag,
                unit=metadata.unit,
                type="calculated",
                description=metadata.description,
            )
            for output_tag, metadata in output_tag_metadata_pairs.items()
        }

    def retrieve_specific_input(self, tag_name: str) -> TagData:
        """
        Get the TagData for a given input tag name
        """
        if tag_name not in self._input_data_dict:
            raise CalculationConfigurationError(
                f"'{self.name}' calculation's input tag '{tag_name}' was not found amongst the wired inputs. "
                + "Make sure the calculation has been wired to a system and the tag name is correct."
            )
        return self._input_data_dict[tag_name]

    def retrieve_specific_output(self, tag_name: str) -> TagData:
        """
        Get the TagData for a given output tag name
        """
        if tag_name not in self._output_tag_info_dict:
            raise CalculationConfigurationError(
                f"'{self.name}' calculation's output tag '{tag_name}' was not found amongst the defined outputs. "
                + "Make sure the tag name is correct."
            )
        return self._output_tag_info_dict[tag_name].data

    def _pre_wire_inputs(
        self,
        system: System,  # pyright: ignore[reportUnusedParameter]
    ) -> tuple[CalculationConfigurationError | None, bool]:
        """
        Hook for any pre-wiring steps that need to be done before inputs are wired.
        """
        return None, True

    def wire_inputs(self, system: System) -> tuple[CalculationConfigurationError | None, bool]:
        """
        Links calculation inputs to tag info instances
        and creates a simple callable for it.
        Also calls the .calculate method once to initialize the results
        so they are not NANs.
        Validation is already done so no error handling is placed here.
        """
        error, successful = self._pre_wire_inputs(system)
        if not successful:
            return error, successful
        input_tag_metadata_pairs: dict[str, TagMetadata] = {
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
                return CalculationConfigurationError(e), not successful
        self._initialized = True
        _ = self.calculate(t=system.time)
        # if somehow the calculation returned NAN, something went wrong and the
        # commissioning failed.
        successful = True
        for tag_info in self._output_tag_info_dict.values():
            if np.isnan(tag_info.data.value):
                error = CalculationConfigurationError(
                    f"'{self.name}' calculation resulted in nan during initialization."
                )
                self._initialized = False
                return error, not successful
        return None, successful

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
        """
        whether or not the calculation quality is ok. If any of the inputs are not ok,
            the calculationis also not ok.
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

    def calculate(self, t: Seconds) -> dict[str, TagData]:
        """public facing method to get the calculation result"""
        if not self._initialized:
            raise RuntimeError(
                "Tried to call 'calculate' before the system orchestrated the various quantities. "
                + "Make sure this calculation is part of a system and the system has been constructed."
            )
        self._update_input_triplets()
        value_dict = self._update_input_values()
        outputs_dict = self._calculation_algorithm(
            t=t,
            inputs_dict=value_dict,
        )
        output_data_dict: dict[str, TagData] = {}
        for output_tag, output_value in outputs_dict.items():
            tag_data = TagData(time=t, value=output_value, ok=self.ok)
            self._output_tag_info_dict[output_tag].data = tag_data
            output_data_dict[output_tag] = tag_data
        return output_data_dict

    @property
    def tag_metadata_dict(self) -> dict[str, TagMetadata]:
        return {
            getattr(self, field): metadata
            for field, metadata in self._field_metadata_dict.items()
            if metadata.type != TagType.CONSTANT
        }

    @property
    def t(self) -> dict[str, TagInfo]:
        return self._output_tag_info_dict

    @property
    def outputs(self) -> dict[str, TagData]:
        return {tag: tag_info.data for tag, tag_info in self._output_tag_info_dict.items()}

    @property
    def output_tag_info_dict(self) -> dict[str, TagInfo]:
        """Public getter for the private _output_tag_info_dict attribute."""
        return self._output_tag_info_dict
