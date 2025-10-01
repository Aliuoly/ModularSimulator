from modular_simulation.quantities.measurable_quantities import MeasurableQuantities
from modular_simulation.quantities.usable_quantities import UsableQuantities
from modular_simulation.quantities.controllable_quantities import ControllableQuantities
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from numpy.typing import NDArray, ArrayLike
from typing import Any, Dict, List, Mapping, TYPE_CHECKING, Tuple, Callable
from scipy.integrate import solve_ivp #type: ignore
from functools import cached_property
from operator import attrgetter
from numba import jit
from numba.typed.typeddict import Dict as NDict
from numba import types
import warnings
from modular_simulation.control_system.controller import ControllerMode
from modular_simulation.measurables.base_classes import BaseIndexedModel
if TYPE_CHECKING:
    from modular_simulation.usables import TimeValueQualityTriplet
    from modular_simulation.control_system import Controller, Trajectory
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)

class System(BaseModel, ABC):
    """
    Abstract base class for a simulation system focused on readability and ease of use.

    This class provides the core simulation loop and a clear contract for defining
    the dynamics of a system. It is designed to handle systems of Differential-Algebraic
    Equations (DAEs) by recalculating algebraic states statelessly within each
    internal step of the ODE solver. This is NOT a robust/rigorous DAE system framework,
    as fundamentally, a ODE solver is used. It should work well enough for most systems though. 

    Subclasses must implement the `rhs` and `_calculate_algebraic_values` static methods,
    which operate on standard Python dictionaries and Pydantic objects, making them
    easy to write and debug.

    Attributes:
        measurable_quantities (MeasurableQuantities): A container for all state, control,
            and algebraic variables that can be measured or recorded.
        usable_quantities (UsableQuantities): Defines how sensors and calculations
            are used to generate measurements from the system's state.
        controllable_quantities (ControllableQuantities): Defines the controllers that
            manipulate the system's `ControlElements`.
        
        solver_options (Dict[str, Any]): A dictionary of options passed directly to
            SciPy's `solve_ivp` function.
        _t (float): The current simulation time.
        _history (List[Dict]): A log of the system's state at each time step.
    """
    dt: float = Field(
        default = ...,
        description = (
            "How often the system's sensors, calculations, and controllers update. "
            "The solver takes adaptive 'internal steps' regardless of this value to update the system's states. "
        )
    )
    measurable_quantities: "MeasurableQuantities" = Field(
        ...,
        description = (
            "A container for all constants, state, control, and algebraic variables that can be measured or recorded. "
        )
    )
    usable_quantities: "UsableQuantities" = Field(
        ...,
        description = (
            "Defines how sensors and calculations are used to generate measurements from the system's measurable_quantities."
        )
    )
    controllable_quantities: "ControllableQuantities" = Field(
        ...,
        description = (
            "Defines the controllers that manipulate the system's `ControlElements`."
        )
    )
    solver_options: Dict[str, Any] = Field(
        default_factory = lambda : {'method': 'LSODA'},
        description = (
            "arguments to scipy.integrate.solve_ivp call done by the system when taking steps."
        )
    )
    use_numba: bool = Field(
        default = True,
        description = (
            "Whether or not to attempt to compile the rhs and algebraic value function with numba. "
            "This should work out of the box and improve speed for large systems."
        )
    )
    numba_options: Dict[str, Any] = Field(
        default_factory = lambda : {'nopython': True, 'cache': True},
        description = (
            "arguments for numba JIT compilation of rhs and calculate_algebraic_values methods. "
            "Ignored if use_numba is False"
        )
    )
    record_history: bool = Field(
        default = False,
        description = (
            "Whether or not to historize the underlying measurable_quantities of the system. "
            "The usable_quantities, which are measured or calculation, are always historized regardless. "
        )
    )
    show_progress: bool = Field(
        default = True,
        description = (
            "Whether or not to show simulation progress when simulating multiple steps at once."
        )
    )

    _history: Dict[str, List[ArrayLike]] = PrivateAttr(default_factory = dict)
    _t: float = PrivateAttr(default = 0.)
    _params: Dict[str, Any] = PrivateAttr() # NDict for numba implementation, Dict[str, slice] otherwise

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    # add private attrs
    _history_slots: List[Tuple[Callable[[Any], Any], List]] = PrivateAttr(default_factory=list)

    def model_post_init(self, context: Any) -> None:

        if self.use_numba:
            self._construct_fast_params()
        else:
            self._construct_params()

        if self.record_history:
            # Build history dict and accessors exactly once
            mq = self.measurable_quantities
            for mq_name in mq.model_dump():
                mq_obj: BaseIndexedModel = getattr(mq, mq_name)
                for tag in mq_obj.model_dump():
                    lst = [] # type:ignore
                    self._history[tag] = lst
                    getter = attrgetter(tag)
                    # capture the object reference and append list ref
                    self._history_slots.append((lambda o=mq_obj, g=getter: g(o), lst)) #type:ignore[misc]
            self._history['time'] = []

        # pre-calculate the algebraic values once to refresh them with current states
        # and control elements and constants
        initial_algebraic_array = self._params["algebraic_values_function"](
            y = self.measurable_quantities.states.to_array(),
            u = self.measurable_quantities.control_elements.to_array(),
            k = self._params["k"],
            y_map = self._params["y_map"],
            u_map = self._params["u_map"],
            k_map = self._params["k_map"],
            algebraic_map = self._params["algebraic_map"],
            algebraic_size = self._params["algebraic_size"]
        )
        self.measurable_quantities.algebraic_states.update_from_array(initial_algebraic_array)
    
    def _construct_fast_params(self) -> None:
        """
        overwrites System's method of the same name to support numba njit decoration
        """
        # convert the Enum's to typed dictionaries for numba
        
        y_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = self.measurable_quantities.states._index_map
        for member in index_map:
            y_map[member] = index_map[member]

        u_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = self.measurable_quantities.control_elements._index_map
        for member in index_map:
            u_map[member] = index_map[member]
        
        k_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = self.measurable_quantities.constants._index_map
        for member in index_map:
            k_map[member] = index_map[member]

        algebraic_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = self.measurable_quantities.algebraic_states._index_map
        for member in index_map:
            algebraic_map[member] = index_map[member]

        algebraic_size = self.measurable_quantities.algebraic_states._array_size

        self._params = {
            'y_map': y_map,
            'u_map': u_map,
            'k_map': k_map,
            'algebraic_map': algebraic_map,
            'algebraic_size': algebraic_size,
            'k': self.measurable_quantities.constants.to_array(),
            'algebraic_values_function': jit(**self.numba_options)(self.calculate_algebraic_values),
            'rhs_function': jit(**self.numba_options)(self.rhs),
        }

    def _construct_params(self) -> None:
        algebraic_size = self.measurable_quantities.algebraic_states._array_size
        self._params = {
            'y_map': self.measurable_quantities.states._index_map,
            'u_map': self.measurable_quantities.control_elements._index_map,
            'k_map': self.measurable_quantities.constants._index_map,
            'algebraic_size': algebraic_size,
            'algebraic_map': self.measurable_quantities.algebraic_states._index_map,
            'k': self.measurable_quantities.constants.to_array(),
            'algebraic_values_function': self.calculate_algebraic_values,
            'rhs_function': self.rhs,
        }

    def _update_history(self) -> None:
        if not self.record_history:
            return
        # simple tight loop: call getter, append to pre-bound list
        for getter, lst in self._history_slots:
            lst.append(getter()) #type:ignore[call-arg]
        self._history['time'].append(self._t)


    def _pre_integration_step(self) -> Tuple[NDArray, NDArray]:
        """
        Handles the control loop logic before integration.

        This method updates all sensors and controllers based on the current state,
        sets the new values for the control elements, and returns the initial state
        array for the solver.
        """
        self.usable_quantities.update(self._t)
        self.controllable_quantities.update(self._t) # returns results too, but I don't need it here.
        # control elements is already updated here by reference.
        return (
            self.measurable_quantities.states.to_array(),
            self.measurable_quantities.control_elements.to_array(),
        )
    
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
            y_map: Mapping[str, slice],
            u_map: Mapping[str, slice],
            k_map: Mapping[str, slice],
            algebraic_map: Mapping[str, slice],
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
    
    def step(self, nsteps: int = 1) -> None:
        """
        The main public method to advance the simulation by one time step.

        Performs one integration step by calling the ODE solver.

        It configures and runs `solve_ivp`, then updates the system's algebraic
        states with the final, successful result.
        """
        show_progress = False
        if nsteps > 1:
            if logger.level == logging.NOTSET:
                # dont show progress bar if we are logging.
                show_progress = self.show_progress
            
        if not isinstance(nsteps, int):
            if isinstance(nsteps, float) and nsteps.is_integer():
                nsteps = int(nsteps)
                
            else:
                raise TypeError("nsteps must be an integer number of steps")

        y_map = self._params['y_map']
        u_map = self._params['u_map']
        k_map = self._params['k_map']
        algebraic_map = self._params['algebraic_map']
        k = self._params['k']
        algebraic_values_function = self._params["algebraic_values_function"]
        rhs_function = self._params["rhs_function"]
        algebraic_size = self._params["algebraic_size"]
        
        if show_progress:
            pbar = tqdm(total = nsteps*self.dt)
        for _ in range(nsteps):
            y0, u0 = self._pre_integration_step()
            final_y = y0
            if self.measurable_quantities.states:
                result = solve_ivp(
                    fun = self._rhs_wrapper,
                    t_span = (self._t, self._t + self.dt),
                    y0 = y0,
                    args = (u0, k, y_map, u_map, k_map, algebraic_map, algebraic_values_function, rhs_function, algebraic_size),
                    **self.solver_options
                )
                final_y = result.y[:, -1]
                self.measurable_quantities.states.update_from_array(final_y)

            # After the final SUCCESSFUL step, update the actual algebraic_states object.
            if self.measurable_quantities.algebraic_states:
                final_algebraic_values = algebraic_values_function(
                    final_y,u0, k, y_map, u_map, k_map, algebraic_map, algebraic_size
                )
                self.measurable_quantities.algebraic_states.update_from_array(final_algebraic_values)
            
            self._t += self.dt
            self._update_history()
            if show_progress:
                pbar.update(self.dt)
        return 
    
    def extend_controller_trajectory(self, cv_tag: str, value: float | None = None) -> "Trajectory":
        """
        used to 'extend' the setpoint trajectory of a controller from the current time onwards.
        If the trajectory has already been defined into the future, the trajectory is trimmed
        back to the current time. 

        cv_tag is used to specify which controller
        
        """
        if cv_tag not in self.controller_dictionary:
            raise ValueError(
                f"Specified cv_tag '{cv_tag}' to extend the trajectory for does not correspond"
                f" to any defined controllers. Available controller cv_tags are {self.cv_tag_list}."
            )

        controller = self.controller_dictionary[cv_tag]
        if controller.mode != ControllerMode.AUTO:
            warnings.warn(
                f"Tried to change trajectory of '{controller.cv_tag}' controller but failed - controller must be in AUTO mode, but is {controller.mode.name}. "
            )
        active_trajectory = controller.sp_trajectory
        if value is None:
            value = active_trajectory(self._t)

        old_value = active_trajectory(self._t)
        active_trajectory.set_now(self._t, value)
        new_value = active_trajectory(self._t)
        logger.info("Setpoint trajectory for '%(tag)s' controller changed at time %(time)0.0f " \
                        "from %(old)0.1e to %(new)0.1e",
                        {'tag': cv_tag, 'time': self._t, 'old': old_value, 'new': new_value})
        return active_trajectory
        
    @cached_property
    def controller_dictionary(self) -> Dict[str, "Controller"]:
        return_dict = {}
        for controller in self.controllable_quantities.controllers:
            return_dict.update({controller.cv_tag : controller})
            while controller.cascade_controller is not None:
                controller = controller.cascade_controller
                return_dict.update({controller.cv_tag : controller})
        return return_dict

    @cached_property
    def cv_tag_list(self) -> List[str]:
        return list(self.controller_dictionary.keys())

    @property
    def measured_history(self) -> Dict[str, Any]:
        """Returns historized measurements and calculations."""

        sensors_detail: Dict[str, List[TimeValueQualityTriplet]] = {}
        calculations_detail: Dict[str, List[TimeValueQualityTriplet]] = {}
        history: Dict[str, Any] = {
            "sensors": sensors_detail,
            "calculations": calculations_detail,
        }

        for sensor in self.usable_quantities.sensors:
            # use alias_tag for when it is different from measurement_tag
            sensors_detail[sensor.alias_tag] = sensor.measurement_history

        for calculation in self.usable_quantities.calculations:
            calculations_detail.update(calculation.history)

        return history

    @property
    def history(self) -> Dict[str, List]:
        """Returns a trimmed copy of the full history dictionary."""
        if not self.record_history:
            return {}
        return self._history

    @property
    def setpoint_history(self) -> Dict[str, Dict[str, List]]:
        """Returns historized controller setpoints keyed by ``cv_tag``."""
        history: Dict[str, Dict[str, List]] = {}
        for controller in self.controllable_quantities.controllers:
            history.update(controller.sp_history)
        return history



