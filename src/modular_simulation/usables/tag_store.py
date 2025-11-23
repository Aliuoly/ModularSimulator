from collections.abc import Mapping, Callable, Iterator
from typing import override
from .tag_info import TagInfo, TagData
from astropy.units import UnitBase, UnitsError


class TagStore(Mapping[str, TagInfo]):
    """
    Provides centralized storage and convenient functions for accessing & updating tags.
    Behaves like a dictionary mapping tag names to TagInfo objects.
    """

    def __init__(self, tags: dict[str, TagInfo] | None = None):
        self._tags: dict[str, TagInfo] = tags if tags else {}

        # Internal caches for filtered views
        self._measured_tags: dict[str, TagInfo] | None = None
        self._calculated_tags: dict[str, TagInfo] | None = None
        self._setpoint_tags: dict[str, TagInfo] | None = None

    @override
    def __getitem__(self, key: str) -> TagInfo:
        return self._tags[key]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._tags)

    @override
    def __len__(self) -> int:
        return len(self._tags)

    @property
    def measured_tags(self) -> dict[str, TagInfo]:
        """Returns a dictionary of all measured tags."""
        if self._measured_tags is None:
            self._measured_tags = {k: v for k, v in self._tags.items() if v.type == "measured"}
        return self._measured_tags

    @property
    def calculated_tags(self) -> dict[str, TagInfo]:
        """Returns a dictionary of all calculated tags."""
        if self._calculated_tags is None:
            self._calculated_tags = {k: v for k, v in self._tags.items() if v.type == "calculated"}
        return self._calculated_tags

    @property
    def setpoint_tags(self) -> dict[str, TagInfo]:
        """Returns a dictionary of all setpoint tags."""
        if self._setpoint_tags is None:
            self._setpoint_tags = {k: v for k, v in self._tags.items() if v.type == "setpoint"}
        return self._setpoint_tags

    def add(self, tag_info: TagInfo | dict[str, TagInfo]) -> None:
        """
        Adds a TagInfo object to the store.
        Updates cached views if they have been accessed.
        """
        if isinstance(tag_info, dict):
            for tag_info in tag_info.values():
                self.add(tag_info)
            return

        if tag_info.tag in self._tags:
            raise ValueError(f"Tag '{tag_info.tag}' already exists in store.")

        self._tags[tag_info.tag] = tag_info

        # Update caches if they exist (incremental update)
        if self._measured_tags is not None and tag_info.type == "measured":
            self._measured_tags[tag_info.tag] = tag_info
        elif self._calculated_tags is not None and tag_info.type == "calculated":
            self._calculated_tags[tag_info.tag] = tag_info
        elif self._setpoint_tags is not None and tag_info.type == "setpoint":
            self._setpoint_tags[tag_info.tag] = tag_info

    def make_converted_data_getter(
        self, tag: str, target_unit: UnitBase | None = None
    ) -> Callable[[], TagData]:
        """
        Convenient method equivalent to TagInfo.make_converted_data_getter.
        Creates a getter for a tag that converts data to the target unit.
        """
        if tag not in self._tags:
            raise KeyError(f"Tag '{tag}' not found in store.")
        try:
            if target_unit is None:
                target_unit = self._tags[tag].unit
            return self._tags[tag].make_converted_data_getter(target_unit)
        except UnitsError as e:
            raise UnitsError(
                f"Failed to create converter for tag '{tag}' from {self._tags[tag].unit} to {target_unit}: {e}"
            )

    def make_data_updater(self, tag: str) -> Callable[[TagData], None]:
        """Creates a callback to update the data of a specific tag."""
        if tag not in self._tags:
            raise KeyError(f"Tag '{tag}' not found in store.")

        tag_info = self._tags[tag]

        def data_updater(tag_data: TagData) -> None:
            tag_info.data = tag_data

        return data_updater

    @property
    def history(self) -> dict[str, list[TagData]]:
        return {k: v.history for k, v in self._tags.items()}
