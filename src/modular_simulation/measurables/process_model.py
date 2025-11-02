from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, PlainSerializer, BeforeValidator
from dataclasses import field, dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING
from scipy.integrate import solve_ivp #type: ignore
from functools import cached_property
from astropy.units import UnitBase, Unit
from collections.abc import Callable
from abc import ABC, abstractmethod
from enum import IntEnum
from numba.typed.typeddict import Dict as NDict #type: ignore
from numba import types, jit #type: ignore
from modular_simulation.validation.exceptions import MeasurableConfigurationError
from modular_simulation.utils import extract_unique_metadata
from modular_simulation.utils.typing import StateValue, ArrayIndex, SerializableUnit, Seconds
if TYPE_CHECKING:
    from modular_simulation.framework.system import System
class StateType(IntEnum):
    DIFFERENTIAL = 0
    ALGEBRAIC    = 1
    CONTROLLED   = 2
    CONSTANT     = 3

class StateMetadata(BaseModel):
    """
    Represents information about a model state, including its type, unit, and description

    :var type: The type of the state (e.g., differential, algebraic, controlled, constant)
    :vartype type: StateType
    :var unit: The unit associated with the state value. Defaults to unitless ("").
    :vartype unit: SerializableUnit
    :var description: A brief description of the state. Use this rather than inline comment where applicable.
                        Defaults to empty string ("").
    :vartype description: str = ""
    """
    type: StateType
    unit: SerializableUnit = ""
    description: str = ""
    model_config = ConfigDict(extra='allow',arbitrary_types_allowed=True)


@dataclass
class CategorizedStateView:
    model: ProcessModel
    state_type: StateType
    
    _index_map: dict[str, ArrayIndex] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        start = 0
        self._index_map: dict[str, ArrayIndex] = {}
        for name, info in self.model.state_metadata_dict.items():
            if info.type != self.state_type:
                continue
            value = getattr(self.model, name)
            if np.ndim(value) == 0:  # scalar
                self._index_map[name] = start
                start += 1
            else:
                length = np.size(value)
                self._index_map[name] = slice(start, start + length)
                start += length

    def to_array(self) -> NDArray:
        # the following combination, from testing, gave the best times
        # use np.zeros(...) to remake array each time
        #   instead of np.empty(...) and alike
        #   using a preallocated array and updating it
        #   was no faster and slowed things down due to
        #   requirement of additional checking logics. 
        # use dictionary for array indexing instead of 
        #   an enum. 
        array = np.zeros(self.array_size, dtype=np.float64)
        model = self.model
        for attr_name, slice_or_index in self.index_map.items(): 
            array[slice_or_index] = getattr(model, attr_name)
        return array

    def update_from_array(self, array: NDArray) -> None:
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
                max_index = max(max_index, item+1)
            elif isinstance(item, slice):
                max_index = max(max_index, item.stop+1)
        return max_index
        

class ProcessModel(BaseModel, ABC):
    """
    Define all states of the system here. Also define the ODE RHS for the differential states,
    and the algebraic equations for the algebraic states. 
    """
    t: Seconds = Field(
        default = 0.0, 
        description = "Current 'ground truth' time of the dynamic system. ALWAYS in units of seconds. "
    )

    _state_metadata_dict: dict[str, StateMetadata] = PrivateAttr()
    _params: dict[str, Any] = PrivateAttr()
    _solver_options: dict[str, Any] = PrivateAttr()
    model_config = ConfigDict(extra = 'forbid')
    
    def model_post_init(self, context):
        """Validate that each field is annotated with exactly one StateMetadata."""
        self._state_metadata_dict = {
            name: extract_unique_metadata(field, StateMetadata, name, MeasurableConfigurationError)
            for name, field in self.__class__.model_fields.items()
            if name != "t"
        }

    def _attach_system(self, system: System) -> None:
        """Called by the system when the process model is added to it."""
        
        self._solver_options = system.solver_options
        if system.use_numba:
            self._construct_fast_params(system.numba_options)
        else:
            self._construct_params()
        
        # pre-calculate the algebraic values once to refresh them with current states
        # and control elements and constants
        initial_algebraic_array = self._params["algebraic_values_function"](
            y = self.differential_view.to_array(),
            u = self.controlled_view.to_array(),
            k = self._params["k"],
            y_map = self._params["y_map"],
            u_map = self._params["u_map"],
            k_map = self._params["k_map"],
            algebraic_map = self._params["algebraic_map"],
            algebraic_size = self._params["algebraic_size"]
        )
        self.algebraic_view.update_from_array(initial_algebraic_array)

    def _construct_fast_params(self, numba_options) -> None:
        """
        overwrites System's method of the same name to support numba njit decoration
        """
        model = self
        y_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = model.differential_view.index_map
        for name, member in index_map.items():
            if isinstance(member, int):
                member = slice(member, member+1)
            y_map[name] = member
        u_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = model.controlled_view.index_map
        for name, member in index_map.items():
            if isinstance(member, int):
                member = slice(member, member+1)
            u_map[name] = member
        k_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = model.constant_view.index_map
        for name, member in index_map.items():
            if isinstance(member, int):
                member = slice(member, member+1)
            k_map[name] = member
        algebraic_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = model.algebraic_view.index_map
        for name, member in index_map.items():
            if isinstance(member, int):
                member = slice(member, member+1)
            algebraic_map[name] = member
        algebraic_size = model.algebraic_view.array_size

        self._params = {
            'y_map': y_map,
            'u_map': u_map,
            'k_map': k_map,
            'algebraic_map': algebraic_map,
            'algebraic_size': algebraic_size,
            'k': model.constant_view.to_array(),
            'algebraic_values_function': jit(**numba_options)(model.calculate_algebraic_values),
            'rhs_function': jit(**numba_options)(model.differential_rhs),
        }

    def _construct_params(self) -> None:
        model = self
        algebraic_size = model.algebraic_view.array_size
        self._params = {
            'y_map': model.differential_view._index_map,
            'u_map': model.controlled_view._index_map,
            'k_map': model.constant_view._index_map,
            'algebraic_size': algebraic_size,
            'algebraic_map': model.algebraic_view._index_map,
            'k': model.constant_view.to_array(),
            'algebraic_values_function': model.calculate_algebraic_values,
            'rhs_function': model.differential_rhs,
        }
    #------ public abstract methods ------
    @staticmethod
    @abstractmethod
    def calculate_algebraic_values(
            y: NDArray,
            u: NDArray,
            k: NDArray,
            y_map: dict[str, ArrayIndex],
            u_map: dict[str, ArrayIndex],
            k_map: dict[str, ArrayIndex],
            algebraic_map: dict[str, ArrayIndex],
            algebraic_size: int,
            ) -> NDArray:
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
            y: NDArray,
            u: NDArray,
            k: NDArray,
            algebraic: NDArray,
            u_map: dict[str, ArrayIndex],
            y_map: dict[str, ArrayIndex],
            k_map: dict[str, ArrayIndex],
            algebraic_map: dict[str, ArrayIndex],
            ) -> NDArray:
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
            y: NDArray, 
            u: NDArray,
            k: NDArray,
            y_map: dict[str, ArrayIndex],
            u_map: dict[str, ArrayIndex],
            k_map: dict[str, ArrayIndex],
            algebraic_map: dict[str, ArrayIndex],
            algebraic_values_function: Callable,
            rhs_function: Callable,
            algebraic_size: int,
            ) -> NDArray:
        """
        A concrete wrapper called by the solver. It recalculates algebraic states
        before calling the user-defined `rhs` for the derivatives. This ensures
        correctness even when the solver rejects and retries steps.
        """
        algebraic_array = algebraic_values_function(
            y = y,
            u = u,
            k = k,
            y_map = y_map,
            u_map = u_map,
            k_map = k_map,
            algebraic_map = algebraic_map,
            algebraic_size = algebraic_size
        )
        return rhs_function(
            t, 
            y = y,
            u = u,
            k = k,
            algebraic = algebraic_array, 
            y_map = y_map,
            u_map = u_map,
            k_map = k_map,
            algebraic_map = algebraic_map
            )
    
    
    #------ public methods ------
    def step(self, dt: Seconds) -> None:
        y_map = self._params['y_map']
        u_map = self._params['u_map']
        k_map = self._params['k_map']
        algebraic_map = self._params['algebraic_map']
        k = self._params['k']
        algebraic_values_function = self._params["algebraic_values_function"]
        rhs_function = self._params["rhs_function"]
        algebraic_size = self._params["algebraic_size"]
        y0 = self.differential_view.to_array()
        u = self.controlled_view.to_array()
        end_t = self.t + dt
        final_y = y0
        result = solve_ivp(
            fun = self._rhs_wrapper,
            t_span = (self.t, end_t),
            y0 = y0,
            args = (u, k, y_map, u_map, k_map, algebraic_map, algebraic_values_function, rhs_function, algebraic_size),
            **self._solver_options
        )
        final_y = result.y[:, -1]
        self.differential_view.update_from_array(final_y)

        # After the final SUCCESSFUL step, update the actual algebraic_states object.
        final_algebraic_values = algebraic_values_function(
            final_y, u, k, y_map, u_map, k_map, algebraic_map, algebraic_size
        )
        self.algebraic_view.update_from_array(final_algebraic_values)
        self.t = end_t

    def categorized_state_metadata_dict(self, type: StateType):
        return {name: info for name, info in self.state_metadata_dict.items() if info.type == type}
    
    def make_converted_getter(self, field_name: str, target_unit: UnitBase|str|None = None) -> Callable[[], StateValue]:
        current_unit = self.state_metadata_dict[field_name].unit

        if target_unit is None:
            target_unit = current_unit
        if not isinstance(target_unit, UnitBase):
            target_unit = Unit(target_unit)

        if current_unit == target_unit:
            def getter() -> StateValue:
                return getattr(self, field_name)
            return getter
        
        converter = current_unit.get_converter(target_unit)
        def converted_getter() -> StateValue:
            return converter(getattr(self, field_name))
        return converted_getter

    def make_converted_setter(self, field_name: str, source_unit: UnitBase) -> Callable[[float], None]:
        current_unit = self.state_metadata_dict[field_name].unit

        if source_unit is None:
            source_unit = current_unit
        if not isinstance(source_unit, UnitBase):
            source_unit = Unit(source_unit)

        source_unit = source_unit if isinstance(source_unit, UnitBase) else Unit(source_unit)
        if current_unit == source_unit:
            def setter(value) -> StateValue:
                return setattr(self, field_name, value)
            return setter
        
        converter = current_unit.get_converter(source_unit)
        def converted_setter(value) -> StateValue:
            return setattr(self, field_name, converter(value))
        return converted_setter
        
    #------ properties -------
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
    