from modular_simulation.usables import (
    Calculation, 
    TagMetadata,
    TagType, 
)
from modular_simulation.usables.tag_info import TagInfo
import numpy as np
from typing import Annotated, List
from astropy.units import Unit, UnitBase
from pydantic import PrivateAttr, Field
SMALL = 1e-12
class FirstOrderFilter(Calculation):
    
    filtered_signal_tag: Annotated[str, TagMetadata(TagType.OUTPUT, Unit())]
    raw_signal_tag: Annotated[str, TagMetadata(TagType.INPUT, Unit())]

    time_constant: Annotated[float, TagMetadata(TagType.CONSTANT, Unit("second"))] = Field(
        default = 0.0,
        description = "time constant of this first order filter. "
    )

    _filtered_signal: float|None = PrivateAttr(default = None)
    _last_t: float| None = PrivateAttr(default=None)

    def _initialize(
            self,
            tag_infos: List[TagInfo],
            ) -> None:
        """
        overwrite the unit info in the annotation
        with the tag info of the raw measurement itself. 
        """
        for tag_info in tag_infos:
            if tag_info.tag == self.raw_signal_tag: 
                self._input_tag_info_dict[self.raw_signal_tag].unit = tag_info.unit
                self._output_tag_info_dict[self.filtered_signal_tag].unit = tag_info.unit
                break
        super()._initialize(tag_infos)
        
    def _calculation_algorithm(self, t, inputs_dict):
        raw = inputs_dict[self.raw_signal_tag]
        if self._filtered_signal is None or self._last_t is None:
            self._filtered_signal = raw
            self._last_t = t
        else:
            dt = t - self._last_t
            alpha = 1 - np.exp(-dt/max(1e-12, self.time_constant))
            self._filtered_signal = alpha * raw + (1-alpha) * self._filtered_signal
            self._last_t = t
        return {self.filtered_signal_tag: self._filtered_signal}
        
