from functools import cached_property
from operator import attrgetter
from typing import Any, Callable, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from modular_simulation.measurables.base_classes import (
    AlgebraicStates,
    BaseIndexedModel,
    ControlElements,
    Constants,
    States,
)

class MeasurableQuantities(BaseModel):
    constants: Constants = Field(
        default_factory = Constants,
        description = (
            "Constants of the system to which this MeasurableQuantities definition applies. "
            "e.g., geometric properties of vessels, rate constants for reactions, ..."
        )
    )
    states: States = Field(
        default_factory = States,
        description = (
            "Differential states of the system to which this MeasurableQuantities definition applies. "
            "e.g., concentration of a species or temperature in the reaction medium."
        )
    )
    control_elements: ControlElements = Field(
        default_factory = ControlElements,
        description = (
            "Final control elements of the system to which this MeasurableQuantities definition applies."
            "e.g., the feed rate of a stream to the system."
        )
    )
    algebraic_states: AlgebraicStates = Field(
        default_factory = AlgebraicStates,
        description = (
            "Algebraic states of the system to which this MeasurableQuantities definition applies. "
            "e.g., the outlet flow of a pressure driven stream dictated by the valve equation, "
                "the consumption rate of a species due to multiple reactions taking place."
        )
    )

    _history_data: Dict[str, Dict[str, Dict[str, List[Any]]]] = PrivateAttr(default_factory=dict)
    _history_slots: List[Tuple[str, str, BaseIndexedModel, Callable[[BaseIndexedModel], Any]]] = PrivateAttr(
        default_factory=list
    )

    model_config = ConfigDict(extra = 'forbid')

    @cached_property
    def available_tags(self) -> list[str]:
        tags = []
        for category in [self.states, self.control_elements, self.algebraic_states, self.constants]:
            tags.extend(category.__class__.model_fields.keys())
        return tags

    def initialize_history(self) -> None:
        self._history_data = {}
        self._history_slots = []
        categories = {
            "states": self.states,
            "control_elements": self.control_elements,
            "algebraic_states": self.algebraic_states,
            "constants": self.constants,
        }

        for name, category in categories.items():
            if category is None:
                continue
            fields = getattr(category.__class__, "model_fields", {})
            if not fields:
                continue
            cat_history: Dict[str, Dict[str, List[Any]]] = {}
            for tag in fields.keys():
                cat_history[tag] = {"time": [], "value": []}
                getter = attrgetter(tag)
                self._history_slots.append((name, tag, category, getter))
            if cat_history:
                self._history_data[name] = cat_history

    def record_history(self, t: float) -> None:
        for category_name, tag, owner, getter in self._history_slots:
            entry = self._history_data[category_name][tag]
            entry["time"].append(t)
            entry["value"].append(getter(owner))

    @property
    def history(self) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:
        return {
            category: {
                tag: {"time": data["time"].copy(), "value": data["value"].copy()}
                for tag, data in entries.items()
            }
            for category, entries in self._history_data.items()
        }
