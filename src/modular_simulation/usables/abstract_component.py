from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TypedDict, TYPE_CHECKING
from pydantic import BaseModel, field_validator, PrivateAttr

if TYPE_CHECKING:
    from modular_simulation.framework import System
    from modular_simulation.utils.typing import Seconds


class ComponentData(TypedDict):
    name: str
    type: str
    module: str
    config: dict[str, Any]  # pyright: ignore[reportExplicitAny]
    state: dict[str, Any]  # pyright: ignore[reportExplicitAny]


class AbstractComponent(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """
    Base class for all components. All components must implement the following methods:
    1. initialize (as in, binding to an instance of System)
    2. update (updating the component's states)
    3. to_dict (converting the component's configuration and history to a dictionary)
    4. from_dict (creating a component from a dictionary holding its configuration and history)

    All components must also have a name attribute. If no name is provided, the name will be set to the class name.
    The orchestrating `System` will ensure that all components have unique names.
    """

    name: str = ""

    _initialized: bool = PrivateAttr(default=False)

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, name: Any) -> str:  # pyright: ignore[reportAny, reportExplicitAny]
        return name if name else cls.__name__

    def initialize(self, system: System) -> list[BaseException]:
        """Initialize the component."""
        exceptions = self._initialize(system)
        if not exceptions:
            self._initialized = True
        return exceptions

    @abstractmethod
    def _initialize(self, system: System) -> list[BaseException]:
        """Initialize the component."""
        ...

    def update(self, t: Seconds) -> list[BaseException]:
        """Update the component's states."""
        if self._initialized:
            exceptions = self._update(t)
            return exceptions
        return [RuntimeError(f"Component '{self.name}' has not been initialized.")]

    @abstractmethod
    def _update(self, t: Seconds) -> list[BaseException]:
        """Update the component's states."""
        ...

    def to_dict(self) -> ComponentData:
        """Convert the component's configuration and history to a dictionary."""
        return ComponentData(
            name=self.name,
            type=self.__class__.__name__,
            module=self.__class__.__module__,
            config=self._get_configuration_dict(),
            state=self._get_runtime_state_dict(),
        )

    @abstractmethod
    def _get_configuration_dict(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Convert the component's configuration to a dictionary."""
        ...

    @abstractmethod
    def _get_runtime_state_dict(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Convert the component's runtime state to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: ComponentData) -> AbstractComponent:
        """Create a component from a dictionary holding its configuration and history."""
        component = cls._load_configuration(data["config"])
        component.name = data["name"]
        component._load_runtime_state(data["state"])
        return component

    @classmethod
    @abstractmethod
    def _load_configuration(cls, data: dict[str, Any]) -> AbstractComponent:  # pyright: ignore[reportExplicitAny]
        """Create a component from a dictionary holding its configuration and history."""
        ...

    @abstractmethod
    def _load_runtime_state(self, state: dict[str, Any]) -> None:  # pyright: ignore[reportExplicitAny]
        """Subclass hook to restore any persisted runtime state."""
        ...
