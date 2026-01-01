from collections.abc import Mapping, Callable, Iterator
from typing import override
from .point import Point, DataValue
from astropy.units import UnitBase, UnitsError


class PointRegistry(Mapping[str, Point]):
    """
    Provides centralized storage and convenient functions for accessing & updating points.
    Behaves like a dictionary mapping tag names to Point objects.
    """

    def __init__(self, points: dict[str, Point] | None = None):
        self._points: dict[str, Point] = points if points else {}

        # Internal caches for filtered views
        self._measured_points: dict[str, Point] | None = None
        self._calculated_points: dict[str, Point] | None = None
        self._setpoint_points: dict[str, Point] | None = None

    @override
    def __getitem__(self, key: str) -> Point:
        return self._points[key]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._points)

    @override
    def __len__(self) -> int:
        return len(self._points)

    @override
    def __repr__(self) -> str:
        return f"PointRegistry({self._points})"

    @property
    def measured_points(self) -> dict[str, Point]:
        """Returns a dictionary of all measured points."""
        if self._measured_points is None:
            self._measured_points = {k: v for k, v in self._points.items() if v.type == "measured"}
        return self._measured_points

    @property
    def calculated_points(self) -> dict[str, Point]:
        """Returns a dictionary of all calculated points."""
        if self._calculated_points is None:
            self._calculated_points = {
                k: v for k, v in self._points.items() if v.type == "calculated"
            }
        return self._calculated_points

    @property
    def setpoint_points(self) -> dict[str, Point]:
        """Returns a dictionary of all setpoint points."""
        if self._setpoint_points is None:
            self._setpoint_points = {k: v for k, v in self._points.items() if v.type == "setpoint"}
        return self._setpoint_points

    def add(self, point: Point | dict[str, Point]) -> None:
        """
        Adds a Point object to the registry.
        Updates cached views if they have been accessed.
        """
        if isinstance(point, dict):
            for p in point.values():
                self.add(p)
            return

        if point.tag in self._points:
            raise ValueError(f"Point '{point.tag}' already exists in registry.")

        self._points[point.tag] = point

        # Update caches if they exist (incremental update)
        if self._measured_points is not None and point.type == "measured":
            self._measured_points[point.tag] = point
        elif self._calculated_points is not None and point.type == "calculated":
            self._calculated_points[point.tag] = point
        elif self._setpoint_points is not None and point.type == "setpoint":
            self._setpoint_points[point.tag] = point

    def make_converted_data_getter(
        self, tag: str, target_unit: UnitBase | None = None
    ) -> Callable[[], DataValue]:
        """
        Convenient method equivalent to Point.make_converted_data_getter.
        Creates a getter for a point that converts data to the target unit.
        """
        if tag not in self._points:
            raise KeyError(f"Point '{tag}' not found in registry.")
        try:
            if target_unit is None:
                target_unit = self._points[tag].unit
            return self._points[tag].make_converted_data_getter(target_unit)
        except UnitsError as e:
            raise UnitsError(
                f"Failed to create converter for point '{tag}' from {self._points[tag].unit} to {target_unit}: {e}"
            )

    def make_converted_data_setter(
        self, tag: str, source_unit: UnitBase | None = None
    ) -> Callable[[DataValue], None]:
        """
        Convenient method equivalent to Point.make_converted_data_setter.
        Creates a setter for a point that converts data from the source unit.
        """
        if tag not in self._points:
            raise KeyError(f"Point '{tag}' not found in registry.")
        try:
            if source_unit is None:
                source_unit = self._points[tag].unit
            return self._points[tag].make_converted_data_setter(source_unit)
        except UnitsError as e:
            raise UnitsError(
                f"Failed to create converter for point '{tag}' from {source_unit} to {self._points[tag].unit}: {e}"
            )

    def make_data_updater(self, tag: str) -> Callable[[DataValue], None]:
        """Creates a callback to update the data of a specific point."""
        if tag not in self._points:
            raise KeyError(f"Point '{tag}' not found in registry.")

        point = self._points[tag]

        def data_updater(data_value: DataValue) -> None:
            point.data = data_value

        return data_updater

    @property
    def history(self) -> dict[str, list[DataValue]]:
        return {k: v.history for k, v in self._points.items()}
