from dataclasses import dataclass, field
from astropy.units import UnitBase, UnitsError #type: ignore
import numpy as np
from modular_simulation.utils.typing import StateValue, TimeValue



@dataclass(slots=True)
class TagData:
    """
    Stores the timestamp, value, and validity status for a tag

    :var time: Timestamp when the tag value was recorded
    :vartype time: float
    :var ok: Indicates if the tag value is valid (not faulty)
    :vartype ok: bool
    """

    time: TimeValue = np.nan
    value: StateValue = np.nan
    ok: bool = False  # whether or not the value is ok (not faulty)


@dataclass(slots=True)
class TagInfo:
    """
    Defines the tag name, unit, and description of a single usable tag. 
    Also contains the current data and history of the tag are public
    read only properties. 
    """
    tag: str
    unit: UnitBase
    description: str = "no description provided"
    _raw_tag: str|None = None # in case 'tag' is an alias tag from a sensor, _raw_tag holds the raw tag name

    _data: TagData = field(init = False, default_factory = TagData)
    _history: list[TagData] = field(init = False, default_factory = list)

    def __init__(self, tag, unit, description, *, _raw_tag = None):
        self.tag = tag
        self.unit = unit
        self.description = description
        self._raw_tag = self.tag if _raw_tag is None else _raw_tag
        self._history = []
        self._data = TagData()

    def make_converted_data_getter(self, target_unit: UnitBase):
        if self.unit == target_unit:
            def getter() -> TagData:
                return self.data
            return getter
        
        converter = self.unit.get_converter(target_unit)

        def converted_getter() -> TagData:
            return TagData(
                self.data.time,
                converter(self.data.value),
                self.data.ok,
            )
        return converted_getter
        
    @property
    def data(self) -> TagData:
        """public access to the private data"""
        return self._data
   
    @data.setter
    def data(self, new_data: TagData) -> None:
        """setter for the private data & historizes new data"""
        self._data = new_data
        self._history.append(new_data)
    
    @property
    def history(self) -> list[TagData]:
        """public access to the tag's history"""
        return self._history
    


