import numpy as np
from typing import Annotated
from astropy.units import Unit
from pydantic import PrivateAttr, Field, BeforeValidator, PlainSerializer
from modular_simulation.usables import CalculationBase, TagMetadata, TagType
from modular_simulation.usables.tag_info import TagInfo
from modular_simulation.utils.typing import TimeValue, StateValue
from modular_simulation.utils.wrappers import second
from modular_simulation.validation.exceptions import CalculationConfigurationError
SMALL = 1e-12


class FirstOrderFilter(CalculationBase):
    """
    only constant "time_constant" is forced to be in units of seconds, but
    you are allowed to pass in Quantity of other time units. 
    """
    filtered_signal_tag: Annotated[str, TagMetadata(TagType.OUTPUT, Unit())]
    raw_signal_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit())]

    time_constant: Annotated[
        float | TimeValue, 
        TagMetadata(TagType.CONSTANT, Unit("second")),
        BeforeValidator(second),
        PlainSerializer(lambda tc: tc.to_value("second"))
    ] = Field(
        default = 0.0,
        description = "time constant of this first order filter in seconds. "
    )

    #----- wiring time definition -----
    _filtered_signal: StateValue = PrivateAttr()
    _t: TimeValue = PrivateAttr()

    def wire_inputs(self, t: TimeValue, available_tag_info_dict: dict[str, TagInfo]) -> None:
        """
        overwrite the unit info in the annotation
        with the tag info of the raw measurement itself. 
        """
        raw_signal_tag_info = available_tag_info_dict.get(self.raw_signal_tag)
        if raw_signal_tag_info is not None:
            self._tag_metadata_dict[self.raw_signal_tag].unit = raw_signal_tag_info.unit
            self._output_tag_info_dict[self.filtered_signal_tag].unit = raw_signal_tag_info.unit
        else:
            raise CalculationConfigurationError(
                f"FirstOrderFilter calculation for signal '{self.raw_signal_tag}' failed to initialize. "
                "No corresponding signal found amongst available tags. "
            )
        super().wire_inputs(t, available_tag_info_dict)
        self._filtered_signal = self._outputs[self.raw_signal_tag]
        self._t = t
        
    def _calculation_algorithm(self, t:TimeValue, inputs_dict:dict[str, StateValue]):
        raw = inputs_dict[self.raw_signal_tag]
        dt = t - self._t
        alpha = 1 - np.exp(-dt/max(1e-12, self.time_constant))
        self._filtered_signal = alpha * raw + (1-alpha) * self._filtered_signal
        self._t = t
        return {self.filtered_signal_tag: self._filtered_signal}
        
