from __future__ import annotations
import sys
from abc import ABC, abstractmethod
from typing import Any, TypedDict, TYPE_CHECKING
from pydantic import BaseModel, field_validator, PrivateAttr
import importlib

from modular_simulation.components.point import DataValue

if TYPE_CHECKING:
    from modular_simulation.framework import System
    from modular_simulation.utils.typing import Seconds


def get_component_class(module: str, name: str) -> type[AbstractComponent]:
    cls = getattr(sys.modules[module], name)  # pyright: ignore[reportAny]
    if not issubclass(cls, AbstractComponent):
        raise ValueError(f"Class {name} is not a subclass of AbstractComponent.")
    return cls


class ComponentDataDict(TypedDict):
    name: str
    type: str
    module: str
    config: dict[str, Any]  # pyright: ignore[reportExplicitAny]
    state: dict[str, Any]  # pyright: ignore[reportExplicitAny]


class ComponentUpdateResult:
    """Result of a component update.

    data_value can be either:
    - A single DataValue (for sensors, controllers)
    - A dict[str, DataValue] (for calculations with multiple outputs)
    """

    data_value: DataValue | dict[str, DataValue]
    exceptions: list[Exception]

    def __init__(
        self,
        data_value: DataValue | dict[str, DataValue],
        exceptions: list[Exception] | None = None,
    ):
        self.data_value = data_value
        self.exceptions = exceptions if exceptions else []

    @property
    def ok(self) -> bool:
        """Whether the result is ok based on its data value(s)."""
        if isinstance(self.data_value, dict):
            return all(dv.ok for dv in self.data_value.values())
        return self.data_value.ok


class AbstractComponent(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """
    Base class for all components. All components must implement the following methods:
    1. commission (as in, binding to an instance of System)
    2. update (updating the component's states)
    1. commission (as in, binding to an instance of System)
    2. update (updating the component's states)
    3. save (converting the component's configuration and history to a dictionary)
    4. load (creating a component from a dictionary holding its configuration and history)

    All components must also have a name attribute. If no name is provided, the name will be set to the class name.
    The orchestrating `System` will ensure that all components have unique names.
    """

    name: str = ""

    _initialized: bool = PrivateAttr(default=False)

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, name: Any) -> str:  # pyright: ignore[reportAny, reportExplicitAny]
        return name if name else cls.__name__

    def initialize(self, system: System) -> list[Exception]:
        """Commission the component into a system."""
        if self._initialized:
            return []

        exceptions = self._initialize(system)
        if not exceptions:
            self._initialized = True
        return exceptions

    def update(self, t: Seconds) -> ComponentUpdateResult:
        """Update the component's state if necessary."""
        if not self._initialized:
            return ComponentUpdateResult(
                data_value=DataValue(),
                exceptions=[RuntimeError(f"Component '{self.name}' is not initialized.")],
            )

        return self._update(t)

    def should_update(self, t: Seconds) -> bool:
        """Check if the component should update at time t."""
        if not self._initialized:
            return False
        return self._should_update(t)

    @abstractmethod
    def _initialize(self, system: System) -> list[Exception]:
        """Internal hook to commission the component into a system."""
        ...

    @abstractmethod
    def _update(self, t: Seconds) -> ComponentUpdateResult:
        """Internal hook to update the component's state."""
        ...

    @abstractmethod
    def _should_update(self, t: Seconds) -> bool:
        """Internal hook to check if the component should update."""
        ...

    def save(self) -> ComponentDataDict:
        """Serialize the component to a dictionary."""
        return ComponentDataDict(
            name=self.name,
            type=self.__class__.__name__,
            module=self.__class__.__module__,
            config=self._get_configuration_dict(),
            state=self._get_runtime_state_dict(),
        )

    @classmethod
    def load(cls, payload: dict[str, Any]) -> AbstractComponent:
        """Create a component from a dictionary."""
        module_name = payload["module"]
        type_name = payload["type"]
        module = importlib.import_module(module_name)
        component_cls = getattr(module, type_name)
        if not issubclass(component_cls, AbstractComponent):
            raise TypeError(f"{component_cls} is not a subclass of AbstractComponent")

        component = component_cls._load_configuration(payload["config"])
        component.name = payload["name"]
        component._load_runtime_state(payload["state"])
        return component

    @abstractmethod
    def _get_configuration_dict(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Convert the component's configuration to a dictionary."""
        ...

    @abstractmethod
    def _get_runtime_state_dict(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Convert the component's runtime state to a dictionary."""
        ...

    @classmethod
    @abstractmethod
    def _load_configuration(cls, data: dict[str, Any]) -> AbstractComponent:  # pyright: ignore[reportExplicitAny]
        """Create a component from a dictionary holding its configuration and history."""
        ...

    @abstractmethod
    def _load_runtime_state(self, state: dict[str, Any]) -> None:  # pyright: ignore[reportExplicitAny]
        """Subclass hook to restore any persisted runtime state."""
        ...
