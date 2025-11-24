from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from dataclasses import field, dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING, NamedTuple, Protocol, override, cast
import importlib
from scipy.integrate import solve_ivp
from functools import cached_property
from astropy.units import UnitBase, Unit
from collections.abc import Callable
from abc import ABC, abstractmethod
from enum import IntEnum
from modular_simulation.validation.exceptions import MeasurableConfigurationError
from modular_simulation.utils.metadata_extraction import extract_unique_metadata
from modular_simulation.utils.typing import StateValue, ArrayIndex, Seconds

if TYPE_CHECKING:
    from modular_simulation.framework.system import System


class AlgebraicCallable(Protocol):
    def __call__(
        self,
        y: NDArray[np.float64],
        u: NDArray[np.float64],
        k: NDArray[np.float64],
        y_map: dict[str, ArrayIndex],
        u_map: dict[str, ArrayIndex],
        k_map: dict[str, ArrayIndex],
        algebraic_map: dict[str, ArrayIndex],
        algebraic_size: int,
    ) -> NDArray[np.float64]: ...


class RHSCallable(Protocol):
    def __call__(
        self,
        t: float,
        y: NDArray[np.float64],
        u: NDArray[np.float64],
        k: NDArray[np.float64],
        algebraic: NDArray[np.float64],
        y_map: dict[str, ArrayIndex],
        u_map: dict[str, ArrayIndex],
        k_map: dict[str, ArrayIndex],
        algebraic_map: dict[str, ArrayIndex],
    ) -> NDArray[np.float64]: ...


class ProcessModelParams(NamedTuple):
    y_map: dict[str, ArrayIndex]
    u_map: dict[str, ArrayIndex]
    k_map: dict[str, ArrayIndex]
    algebraic_map: dict[str, ArrayIndex]
    k: NDArray[np.float64]
    algebraic_values_function: AlgebraicCallable
    rhs_function: RHSCallable
    algebraic_size: int


class StateType(IntEnum):
    DIFFERENTIAL = 0
    ALGEBRAIC = 1
    CONTROLLED = 2
    CONSTANT = 3


class StateMetadata:
    """
    Represents information about a model state, including its type, unit, and description

    :var type: The type of the state (e.g., differential, algebraic, controlled, constant)
    :vartype type: StateType
    :var unit: The unit associated with the state value. Defaults to unitless ("").
    :vartype unit: UnitBase
    :var description: A brief description of the state. Use this rather than inline comment where applicable.
                        Defaults to empty string ("").
    :vartype description: str
    """

    type: StateType
    unit: UnitBase
    description: str = ""

    def __init__(self, type: StateType, unit: UnitBase | str, description: str = ""):
        if not isinstance(type, StateType):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"type must be a StateType, not {type}")  # pyright: ignore[reportUnreachable]
        if not isinstance(unit, UnitBase):
            unit = Unit(unit)
        self.type = type
        self.unit = unit
        self.description = description


@dataclass
class CategorizedStateView:
    model: ProcessModel
    state_type: StateType

    _index_map: dict[str, ArrayIndex] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        start = 0
        for name, info in self.model.state_metadata_dict.items():
            if info.type != self.state_type:
                continue
            value = self.model.state_getter(name)
            if np.ndim(value) == 0:  # scalar
                self._index_map[name] = start
                start += 1
            else:
                length = np.size(value)
                self._index_map[name] = slice(start, start + length)
                start += length

    def to_array(self) -> NDArray[np.float64]:
        # the following combination, from testing, gave the best times
        # use np.zeros(...) to remake array each time
        #   instead of np.empty(...) and alike
        #   using a preallocated array and updating it
        #   was no faster and slowed things down due to
        #   requirement of additional checking logics.
        # use dictionary for array indexing instead of
        #   an enum.
        array = np.zeros(self.array_size, dtype=np.float64)
        for attr_name, slice_or_index in self.index_map.items():
            array[slice_or_index] = self.model.state_getter(attr_name)
        return array

    def update_from_array(self, array: NDArray[np.float64]) -> None:
        """updates the class in place using the provided array."""
        model = self.model
        for attr_name, slice_or_index in self._index_map.items():
            setattr(model, attr_name, array[slice_or_index])

    @property
    def index_map(self) -> dict[str, ArrayIndex]:
        return self._index_map

    @cached_property
    def state_list(self) -> list[str]:
        return list(self._index_map.keys())

    @cached_property
    def array_size(self) -> int:
        max_index = 0
        for item in self._index_map.values():
            if isinstance(item, int):
                max_index = max(max_index, item + 1)
            elif isinstance(item, slice):  # pyright: ignore[reportUnnecessaryIsInstance]
                max_index = max(max_index, cast(int, item.stop) + 1)
            else:
                raise ValueError(f"Invalid index type: {type(item)}")
        return max_index


class ProcessModel(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """
    Define all states of the system here. Also define the ODE RHS for the differential states,
    and the algebraic equations for the algebraic states.
    """

    t: Seconds = Field(
        default=0.0,
        description="Current 'ground truth' time of the dynamic system. ALWAYS in units of seconds. ",
    )

    _state_metadata_dict: dict[str, StateMetadata] = PrivateAttr()
    _params: ProcessModelParams = PrivateAttr()
    _solver_options: dict[str, Any] = PrivateAttr()  # pyright: ignore[reportExplicitAny]
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def model_post_init(self, context: Any) -> None:  # pyright: ignore[reportAny, reportExplicitAny]
        """Validate that each field is annotated with exactly one StateMetadata."""
        self._state_metadata_dict = {
            name: extract_unique_metadata(field, StateMetadata, name, MeasurableConfigurationError)
            for name, field in self.__class__.model_fields.items()
            if name != "t"
        }

    def state_getter(self, state_name: str) -> StateValue:
        return cast(StateValue, getattr(self, state_name))

    def attach_system(self, system: System) -> None:
        """Called by the system when the process model is added to it."""

        self._solver_options = system.solver_options
        self._construct_params()

        # pre-calculate the algebraic values once to refresh them with current states
        # and control elements and constants
        initial_algebraic_array = self._params.algebraic_values_function(
            y=self.differential_view.to_array(),
            u=self.controlled_view.to_array(),
            k=self._params.k,
            y_map=self._params.y_map,
            u_map=self._params.u_map,
            k_map=self._params.k_map,
            algebraic_map=self._params.algebraic_map,
            algebraic_size=self._params.algebraic_size,
        )
        self.algebraic_view.update_from_array(initial_algebraic_array)

    def save(self) -> dict[str, Any]:
        """Return minimal configuration and state needed to reconstruct the model."""

        def _serialize_state(value: StateValue) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "t": self.t,
            "states": {
                name: _serialize_state(self.state_getter(name))
                for name in self.state_metadata_dict
            },
        }

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "ProcessModel":
        """Recreate a process model instance from serialized state values."""

        module = importlib.import_module(payload["module"])
        process_model_cls = getattr(module, payload["type"])
        if not issubclass(process_model_cls, cls):
            raise TypeError(f"{process_model_cls} is not a subclass of {cls}")

        state_values = payload.get("states", {})
        return process_model_cls(t=payload.get("t", 0.0), **state_values)

    # def _construct_fast_params(self, numba_options) -> None:
    #     """
    #     overwrites System's method of the same name to support numba njit decoration
    #     """
    #     model = self
    #     y_map = NDict.empty(key_type=types.unicode_type, value_type=types.slice2_type)
    #     index_map = model.differential_view.index_map
    #     for name, member in index_map.items():
    #         if isinstance(member, int):
    #             member = slice(member, member + 1)
    #         y_map[name] = member
    #     u_map = NDict.empty(key_type=types.unicode_type, value_type=types.slice2_type)
    #     index_map = model.controlled_view.index_map
    #     for name, member in index_map.items():
    #         if isinstance(member, int):
    #             member = slice(member, member + 1)
    #         u_map[name] = member
    #     k_map = NDict.empty(key_type=types.unicode_type, value_type=types.slice2_type)
    #     index_map = model.constant_view.index_map
    #     for name, member in index_map.items():
    #         if isinstance(member, int):
    #             member = slice(member, member + 1)
    #         k_map[name] = member
    #     algebraic_map = NDict.empty(key_type=types.unicode_type, value_type=types.slice2_type)
    #     index_map = model.algebraic_view.index_map
    #     for name, member in index_map.items():
    #         if isinstance(member, int):
    #             member = slice(member, member + 1)
    #         algebraic_map[name] = member
    #     algebraic_size = model.algebraic_view.array_size

    #     self._params = NumbaProcessModelParams(
    #         y_map=y_map,
    #         u_map=u_map,
    #         k_map=k_map,
    #         algebraic_map=algebraic_map,
    #         algebraic_size=algebraic_size,
    #         k=model.constant_view.to_array(),
    #         algebraic_values_function=jit(**numba_options)(model.calculate_algebraic_values),
    #         rhs_function=jit(**numba_options)(model.differential_rhs),
    #     )

    def _construct_params(self) -> None:
        model = self
        algebraic_size = model.algebraic_view.array_size
        self._params = ProcessModelParams(
            y_map=model.differential_view.index_map,
            u_map=model.controlled_view.index_map,
            k_map=model.constant_view.index_map,
            algebraic_size=algebraic_size,
            algebraic_map=model.algebraic_view.index_map,
            k=model.constant_view.to_array(),
            algebraic_values_function=model.calculate_algebraic_values,
            rhs_function=model.differential_rhs,
        )

    # ------ public abstract methods ------
    @staticmethod
    @abstractmethod
    def calculate_algebraic_values(
        y: NDArray[np.float64],
        u: NDArray[np.float64],
        k: NDArray[np.float64],
        y_map: dict[str, ArrayIndex],
        u_map: dict[str, ArrayIndex],
        k_map: dict[str, ArrayIndex],
        algebraic_map: dict[str, ArrayIndex],
        algebraic_size: int,
    ) -> NDArray[np.float64]:
        """
        Calculates derived quantities based on the provided state array, controls, and constants.

        This method MUST be stateless and depend only on its inputs. It will be called
        repeatedly by the ODE solver's internal steps.

        Args:
            y: The current values of the differential state variables as a flat NumPy array.
            y_map: A mapping from state names to their slices within ``y``.
            u_map: A mapping from control element names to their slices within ``u``.
            k_map: A mapping from constant names to their slices within ``k``.
            algebraic_map: A mapping from algebraic state names to their slices within the resulting array.

        Returns:
            A NumPy array containing the calculated algebraic values.
        """
        pass

    @staticmethod
    @abstractmethod
    def differential_rhs(
        t: float,
        y: NDArray[np.float64],
        u: NDArray[np.float64],
        k: NDArray[np.float64],
        algebraic: NDArray[np.float64],
        y_map: dict[str, ArrayIndex],
        u_map: dict[str, ArrayIndex],
        k_map: dict[str, ArrayIndex],
        algebraic_map: dict[str, ArrayIndex],
    ) -> NDArray[np.float64]:
        """
        Defines the right-hand side of the system's ordinary differential equations.

        This method should ONLY contain the derivative calculations and must be stateless.

        Args:
            t: The current simulation time.
            y: The current values of the differential state variables as a flat NumPy array.
            y_map: A mapping from state names to their slices within ``y``.
            algebraic: A NumPy array of freshly calculated algebraic values for this ``y``.
            u_map: A mapping from control element names to their slices within ``u``.
            k_map: A mapping from constant names to their slices within ``k``.
            algebraic_map: A mapping from algebraic state names to their slices within ``algebraic``.

        Returns:
            A NumPy array of the calculated derivatives (dy/dt).
        """
        pass

    @staticmethod
    def _rhs_wrapper(
        t: float,
        y: NDArray[np.float64],
        *args: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    ) -> NDArray[np.float64]:
        # the cast is guaranteed by the .step method
        # but below is done to make pyright happy.
        u = cast(NDArray[np.float64], args[0])
        params = cast(ProcessModelParams, args[1])
        algebraic = params.algebraic_values_function(
            y,
            u=u,
            k=params.k,
            y_map=params.y_map,
            u_map=params.u_map,
            k_map=params.k_map,
            algebraic_map=params.algebraic_map,
            algebraic_size=params.algebraic_size,
        )

        return params.rhs_function(
            t,
            y,
            u,
            params.k,
            algebraic,
            params.y_map,
            params.u_map,
            params.k_map,
            params.algebraic_map,
        )

    # ------ public methods ------
    def step(self, dt: Seconds) -> None:
        y0 = self.differential_view.to_array()
        u = self.controlled_view.to_array()
        end_t = self.t + dt
        final_y = y0
        result = solve_ivp(
            fun=self._rhs_wrapper,
            t_span=(self.t, end_t),
            y0=y0,
            args=(u, self._params),
            **self._solver_options,  # pyright: ignore[reportAny]
        )
        final_y = result.y[:, -1]
        self.differential_view.update_from_array(final_y)

        # After the final SUCCESSFUL step, update the actual algebraic_states object.
        final_algebraic_values = self._params.algebraic_values_function(
            final_y,
            u,
            self._params.k,
            self._params.y_map,
            self._params.u_map,
            self._params.k_map,
            self._params.algebraic_map,
            self._params.algebraic_size,
        )
        self.algebraic_view.update_from_array(final_algebraic_values)
        self.t = end_t

    def categorized_state_metadata_dict(self, type: StateType):
        return {name: info for name, info in self.state_metadata_dict.items() if info.type == type}

    def make_converted_getter(
        self, field_name: str, target_unit: UnitBase | str | None = None
    ) -> Callable[[], StateValue]:
        current_unit = self.state_metadata_dict[field_name].unit

        if target_unit is None:
            target_unit = current_unit
        if not isinstance(target_unit, UnitBase):
            target_unit = Unit(target_unit)

        state_getter = self.state_getter
        if current_unit == target_unit:

            def getter() -> StateValue:
                return state_getter(field_name)

            return getter

        converter = current_unit.get_converter(target_unit)

        def converted_getter() -> StateValue:
            return converter(state_getter(field_name))

        return converted_getter

    def make_converted_setter(
        self, field_name: str, source_unit: UnitBase | str | None = None
    ) -> Callable[[StateValue], None]:
        current_unit = self.state_metadata_dict[field_name].unit

        if source_unit is None:
            source_unit = current_unit
        if not isinstance(source_unit, UnitBase):
            source_unit = Unit(source_unit)

        if current_unit == source_unit:

            def setter(value: StateValue) -> None:
                return setattr(self, field_name, value)

            return setter

        # convert from the SOURCE that is calling this setter
        # to the STATE(CURRENT) unit to be used in the rhs and algebraic calculations.
        converter = source_unit.get_converter(current_unit)

        def converted_setter(value: StateValue) -> None:
            return setattr(self, field_name, converter(value))

        return converted_setter

    # ------ properties -------
    @cached_property
    def differential_view(self) -> CategorizedStateView:
        return CategorizedStateView(self, StateType.DIFFERENTIAL)

    @cached_property
    def algebraic_view(self) -> CategorizedStateView:
        return CategorizedStateView(self, StateType.ALGEBRAIC)

    @cached_property
    def controlled_view(self) -> CategorizedStateView:
        return CategorizedStateView(self, StateType.CONTROLLED)

    @cached_property
    def constant_view(self) -> CategorizedStateView:
        return CategorizedStateView(self, StateType.CONSTANT)

    @cached_property
    def state_metadata_dict(self) -> dict[str, StateMetadata]:
        """dictionary of info for all available states"""
        return self._state_metadata_dict

    @cached_property
    def state_list(self) -> list[str]:
        """list of measurable states' names ('tag list')"""
        return list(self.state_metadata_dict.keys())
