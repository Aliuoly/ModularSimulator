from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, override
import numpy as np

if TYPE_CHECKING:
    from astropy.units import UnitBase
    from modular_simulation.utils.typing import StateValue, Seconds


@dataclass(slots=True)
class DataValue:
    """
    Stores the timestamp, value, and validity status for a point

    :var time: Timestamp when the value was recorded
    :vartype time: float
    :var ok: Indicates if the value is valid (not faulty)
    :vartype ok: bool
    """

    time: Seconds = np.nan
    value: StateValue = np.nan
    ok: bool = False  # whether or not the value is ok (not faulty)


@dataclass(slots=True)
class Point:
    """
    Defines the tag name, unit, and description of a single usable point.
    Also contains the current data and history of the point as public
    read only properties.
    """

    tag: str
    unit: UnitBase
    type: Literal["measured", "calculated", "setpoint", "raw"]
    description: str
    _raw_tag: str

    _data: DataValue = field(init=False, default_factory=DataValue)
    _history: list[DataValue] = field(init=False, default_factory=list)

    def __init__(
        self,
        tag: str,
        type: Literal["measured", "calculated", "setpoint", "raw"],
        unit: UnitBase,
        description: str = "no description provided",
        *,
        _raw_tag: str | None = None,
    ):
        self.tag = tag
        self.unit = unit
        self.type = type
        self.description = description
        self._raw_tag = self.tag if _raw_tag is None else _raw_tag
        self._history = []
        self._data = DataValue()

    @override
    def __repr__(self) -> str:
        return f"Point(tag={self.tag}, type={self.type}, unit={self.unit}, description={self.description}, raw_tag={self._raw_tag}, data={self._data}, history length: {len(self._history)})"

    def make_converted_data_getter(self, target_unit: UnitBase | None = None):
        if target_unit is None:
            target_unit = self.unit
        if self.unit == target_unit:

            def getter() -> DataValue:
                return self.data

            return getter

        converter = self.unit.get_converter(target_unit)

        def converted_getter() -> DataValue:
            return DataValue(
                self.data.time,
                converter(self.data.value),
                self.data.ok,
            )

        return converted_getter

    def make_converted_data_setter(self, source_unit: UnitBase | None = None):
        if source_unit is None:
            source_unit = self.unit
        if self.unit == source_unit:

            def setter(new_data: DataValue) -> None:
                self.data = new_data

            return setter

        converter = source_unit.get_converter(self.unit)

        def converted_setter(new_data: DataValue) -> None:
            converted_value = converter(new_data.value)
            self.data = DataValue(
                new_data.time,
                converted_value,
                new_data.ok,
            )

        return converted_setter

    @property
    def data(self) -> DataValue:
        """public access to the private data"""
        return self._data

    @data.setter
    def data(self, new_data: DataValue) -> None:
        """setter for the private data & historizes new data"""
        self._data = new_data
        self._history.append(new_data)

    @property
    def raw_tag(self) -> str:
        """public access to the private raw tag"""
        return self._raw_tag

    @property
    def history(self) -> list[DataValue]:
        """public access to the point's history"""
        return self._history
