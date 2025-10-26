"""Core dynamic model definitions for the runtime orchestrator."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
from astropy.units import Unit, UnitBase

from modular_simulation.validation.exceptions import MeasurableConfigurationError


class MeasurableType(IntEnum):
    """Enumeration describing the role of a measurable quantity."""

    DIFFERENTIAL_STATE = 0
    ALGEBRAIC_STATE = 1
    CONTROL_ELEMENT = 2
    CONSTANT = 3


@dataclass(slots=True)
class MeasurableMetadata:
    """Metadata describing how a field participates in the dynamic model."""

    type: MeasurableType
    unit: str | UnitBase


class _CategoryView:
    """Lightweight view over a subset of a :class:`DynamicModel`."""

    __slots__ = ("_model", "_category")

    def __init__(self, model: "DynamicModel", category: MeasurableType) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_category", category)

    def to_array(self) -> NDArray:
        """Return the category values as a contiguous ``numpy`` array."""

        return self._model._to_array(self._category)

    def update_from_array(self, array: NDArray) -> None:
        """Update the underlying model in-place from ``array`` values."""

        self._model._update_from_array(self._category, array)

    @property
    def _index_map(self) -> Mapping[str, slice]:
        return self._model._category_maps[self._category]

    @property
    def _array_size(self) -> int:
        return self._model._category_sizes[self._category]

    @property
    def tag_list(self) -> list[str]:
        return list(self._index_map.keys())

    def model_dump(self) -> dict[str, Any]:
        return {tag: getattr(self._model, tag) for tag in self._index_map}

    @property
    def tag_unit_info(self) -> dict[str, UnitBase]:
        return {tag: self._model.tag_unit_info[tag] for tag in self._index_map}

    def __bool__(self) -> bool:  # pragma: no cover - thin wrapper
        return self._array_size > 0

    def __getattr__(self, item: str) -> Any:
        index_map = self._index_map
        if item in index_map:
            return getattr(self._model, item)
        raise AttributeError(item) from None

    def __setattr__(self, key: str, value: Any) -> None:
        index_map = self._index_map
        if key in index_map:
            setattr(self._model, key, value)
            return
        raise AttributeError(key)


class DynamicModel(BaseModel, ABC):
    """Base container for a system's measurable state and governing equations."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _category_maps: dict[MeasurableType, dict[str, slice]] = PrivateAttr()
    _category_sizes: dict[MeasurableType, int] = PrivateAttr()
    _tag_unit_info: dict[str, UnitBase] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # pragma: no cover - validated via unit tests
        self._category_maps = {category: {} for category in MeasurableType}
        self._category_sizes = {category: 0 for category in MeasurableType}

        for field_name, field_info in self.__class__.model_fields.items():
            metadata = field_info.metadata
            if not metadata:
                raise MeasurableConfigurationError(
                    f"Field '{field_name}' is missing measurable metadata."
                )

            measurable_metadata: MeasurableMetadata | None = None
            for metadatum in metadata:
                if isinstance(metadatum, MeasurableMetadata):
                    measurable_metadata = metadatum
                    break

            if measurable_metadata is None:
                raise MeasurableConfigurationError(
                    f"Field '{field_name}' is missing measurable metadata."
                )

            array_value = np.asarray(getattr(self, field_name), dtype=float)
            category = measurable_metadata.type
            start = self._category_sizes[category]
            stop = start + array_value.size
            self._category_maps[category][field_name] = slice(start, stop)
            self._category_sizes[category] = stop

            unit = (
                measurable_metadata.unit
                if isinstance(measurable_metadata.unit, UnitBase)
                else Unit(measurable_metadata.unit)
            )
            self._tag_unit_info[field_name] = unit

    @model_validator(mode="after")
    def _ensure_metadata_unique(self) -> "DynamicModel":  # pragma: no cover - defensive
        seen = set()
        duplicates: list[str] = []
        for category_map in self._category_maps.values():
            for tag in category_map:
                if tag in seen:
                    duplicates.append(tag)
                seen.add(tag)
        if duplicates:
            raise MeasurableConfigurationError(
                "Duplicate tags detected in dynamic model definition: "
                + ", ".join(sorted(set(duplicates)))
            )
        if not seen:
            raise MeasurableConfigurationError(
                "No measurable quantities defined for the dynamic model."
            )
        return self

    def _to_array(self, category: MeasurableType) -> NDArray:
        size = self._category_sizes[category]
        if size == 0:
            return np.zeros(0, dtype=float)
        array = np.zeros(size, dtype=float)
        for tag, indices in self._category_maps[category].items():
            value = np.asarray(getattr(self, tag), dtype=float)
            array[indices] = value
        return array

    def _update_from_array(self, category: MeasurableType, array: NDArray) -> None:
        for tag, indices in self._category_maps[category].items():
            values = array[indices]
            if values.size == 1:
                setattr(self, tag, float(values[0]))
            else:
                setattr(self, tag, values.copy())

    @cached_property
    def states(self) -> _CategoryView:
        return _CategoryView(self, MeasurableType.DIFFERENTIAL_STATE)

    @cached_property
    def algebraic_states(self) -> _CategoryView:
        return _CategoryView(self, MeasurableType.ALGEBRAIC_STATE)

    @cached_property
    def control_elements(self) -> _CategoryView:
        return _CategoryView(self, MeasurableType.CONTROL_ELEMENT)

    @cached_property
    def constants(self) -> _CategoryView:
        return _CategoryView(self, MeasurableType.CONSTANT)

    @cached_property
    def tag_list(self) -> list[str]:
        tags: list[str] = []
        for category_map in self._category_maps.values():
            tags.extend(category_map.keys())
        return tags

    @cached_property
    def tag_unit_info(self) -> dict[str, UnitBase]:
        return dict(self._tag_unit_info)

    @staticmethod
    @abstractmethod
    def calculate_algebraic_values(
        y: NDArray,
        u: NDArray,
        k: NDArray,
        y_map: Mapping[str, slice],
        u_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
        algebraic_size: int,
    ) -> NDArray:
        """Compute algebraic quantities for the supplied state."""

    @staticmethod
    @abstractmethod
    def rhs(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        algebraic: NDArray,
        u_map: Mapping[str, slice],
        y_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
    ) -> NDArray:
        """Evaluate the system right-hand-side for the provided state."""


__all__ = [
    "DynamicModel",
    "MeasurableMetadata",
    "MeasurableType",
]
