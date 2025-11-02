from __future__ import annotations
import numpy as np
from typing import Annotated, TYPE_CHECKING
from pydantic import PrivateAttr, Field
from modular_simulation.usables import CalculationBase, TagMetadata, TagType
from modular_simulation.utils.typing import Seconds, StateValue
from modular_simulation.validation.exceptions import CalculationConfigurationError
if TYPE_CHECKING:
    from modular_simulation.framework.system import System
SMALL = 1e-12


class FirstOrderFilter(CalculationBase):
    """
    only constant "time_constant" is forced to be in units of seconds, but
    you are allowed to pass in Quantity of other time units. 
    """
    filtered_signal_tag: Annotated[str, TagMetadata(type=TagType.OUTPUT, unit="", description="")]
    raw_signal_tag: Annotated[str, TagMetadata(type=TagType.INPUT, unit="")]

    time_constant: Annotated[
        Seconds,
        TagMetadata(type=TagType.CONSTANT, unit="s"),
    ] = Field(
        default = 0.0,
        description = "time constant of this first order filter in seconds. "
    )

    #----- wiring time definition -----
    _filtered_signal: StateValue = PrivateAttr()
    _t: Seconds = PrivateAttr()

    def _pre_wire_inputs(self, system: System) -> tuple[Exception|None, bool]:
        """
        overwrite the unit info in the annotation
        with the tag info of the raw measurement itself. 
        """
        available_tag_info_dict = system.tag_info_dict
        raw_signal_tag_info = available_tag_info_dict.get(self.raw_signal_tag)
        if raw_signal_tag_info is not None:
            self._field_metadata_dict["raw_signal_tag"].unit = raw_signal_tag_info.unit
            self._field_metadata_dict["filtered_signal_tag"].unit = raw_signal_tag_info.unit
            self._output_tag_info_dict[self.filtered_signal_tag].unit = raw_signal_tag_info.unit
            self._filtered_signal = raw_signal_tag_info.data.value
        else:
            error = CalculationConfigurationError(
                f"FirstOrderFilter calculation for signal '{self.raw_signal_tag}' failed to initialize. "
                "No corresponding signal found amongst available tags. "
            )
            return error, False
        # initialize filtered signal to raw signal at start
        self._t = system.time
        return None, True
        
    def _calculation_algorithm(self, t:Seconds, inputs_dict:dict[str, StateValue]):
        raw = inputs_dict[self.raw_signal_tag]
        dt = t - self._t
        alpha = 1 - np.exp(-dt/max(1e-12, self.time_constant))
        self._filtered_signal = alpha * raw + (1-alpha) * self._filtered_signal
        self._t = t
        return {self.filtered_signal_tag: self._filtered_signal}
        
