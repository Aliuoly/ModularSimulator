from dataclasses import dataclass, field
from typing import List, Optional, Callable
from astropy.units import Unit, UnitBase
import numpy as np
from numpy.typing import NDArray



@dataclass(slots=True)
class TagData:
    """Simple container class for a single tag."""

    time: float = 0.0
    value: float | NDArray = np.nan
    ok: bool = False  # whether or not the value is ok (not faulty)

@dataclass(slots=True)
class TagInfo:
    """
    Defines the tag name, unit, and description of a single usable tag. 
    Also contains the current data and history of the tag are public
    read only properties. 
    """
    tag: str
    unit: Optional[Unit] = None
    description: Optional[str] = "no description provided"
    # data is private attr such that user can't set its value directly
    _data: TagData = field(
        init = False,
        default_factory = TagData
    )
    _history: List[TagData] = field(
        init = False,
        default_factory = list
    )

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
    def history(self) -> List[TagData]:
        """public access to the tag's history"""
        return self._history
    


