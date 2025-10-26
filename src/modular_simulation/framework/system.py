from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict, model_validator
from numpy.typing import NDArray, ArrayLike
from typing import Any, TYPE_CHECKING
from collections.abc import Mapping, Callable
from scipy.integrate import solve_ivp #type: ignore
from functools import cached_property
from operator import attrgetter
from numba.typed.typeddict import Dict as NDict #type: ignore
from numba import types, jit #type: ignore
import warnings
from astropy.units import Quantity #type: ignore
from modular_simulation.usables.usable_quantities import UsableQuantities
from modular_simulation.usables.controllers.controller_base import ControllerMode
from modular_simulation.measurables.measurable_base import MeasurableBase
from modular_simulation.measurables.measurable_quantities import MeasurableQuantities
from modular_simulation.validation.exceptions import (
    SensorConfigurationError, 
    ControllerConfigurationError
)
if TYPE_CHECKING:
    from modular_simulation.usables import TagData
    from modular_simulation.usables import ControllerBase, Trajectory
import logging
from tqdm import tqdm #type: ignore
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
        
        solver_options (dict[str, Any]): A dictionary of options passed directly to
            SciPy's `solve_ivp` function.
        _t (float): The current simulation time.
        _history (list[dict]): A log of the system's state at each time step.
    """
    dt: Quantity = Field(
        default = ...,
        description = (
            "How often the system's sensors, calculations, and controllers update. "
            "The solver takes adaptive 'internal steps' regardless of this value to update the system's states. "
        )
    )
    measurable_quantities: MeasurableQuantities = Field(
        ...,
        description = (
            "A container for all constants, state, control, and algebraic variables that can be measured or recorded. "
        )
    )
    usable_quantities: UsableQuantities = Field(
        ...,
        description = (
            "Defines how sensors and calculations and controllers are used to generate measurements from the system's measurable_quantities."
        )
    )
    solver_options: dict[str, Any] = Field(
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
    numba_options: dict[str, Any] = Field(
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

    _history: dict[str, list[ArrayLike]] = PrivateAttr(default_factory = dict)
    _t: float = PrivateAttr(default = 0.)
    _params: dict[str, Any] = PrivateAttr() # NDict for numba implementation, dict[str, slice] otherwise

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    # add private attrs
    _history_slots: list[tuple[Callable[[Any], Any], list]] = PrivateAttr(default_factory=list)

    def get_state(self) -> dict:
        state = {}
        state["measurable_quantities"] = self.measurable_quantities.model_dump(serialize_as_any=True)
        state["sensors"] = []
        state["calculations"] = []
        state["controllers"] = []
        for sensor in self.usable_quantities.sensors:
            state["sensors"].append(sensor.get_state())
        for calculation in self.usable_quantities.calculations:
            state["calculations"].append(calculation.get_state())
        for controller in self.usable_quantities.controllers:
            state["controllers"].append(controller.get_state())
        return state
    
    def set_state(self, state: dict):
        self.measurable_quantities.model_validate_json(state["measurable_quantities"])
        
        for calculation in self.usable_quantities.calculations:
            state.update(calculation.get_state())
        for controller in self.usable_quantities.controllers:
            state.update(controller.get_state())

    @model_validator(mode = 'after')
    def _validate(self):
        exception_group = []
        exception_group.extend(self._validate_sensors_resolvable())
        exception_group.extend(self._validate_controllers_resolvable())
        
        if len(exception_group) > 0:
            raise ExceptionGroup(
                "errors encountered during usable quantity instantiation:", 
                exception_group
                )
        # after validation, initialize usable quantities
        self.usable_quantities._initialize(self.measurable_quantities)
        return self
    
    def _validate_sensors_resolvable(self):
        exception_group = []
        available_measurement_tags = self.measurable_quantities.tag_list
        unavailable_measurement_tags = []

        for sensor in self.usable_quantities.sensors:
            measurement_tag = sensor.measurement_tag
            if measurement_tag not in available_measurement_tags:
                unavailable_measurement_tags.append(measurement_tag)

        if len(unavailable_measurement_tags) > 0:
            exception_group.append(
                SensorConfigurationError(
                    "The following measurement tag(s) are not defined in measurable quantities: "
                    f"{', '.join(unavailable_measurement_tags)}."
                )
            )
        return exception_group
    
    def _validate_controllers_resolvable(self):
        exception_group = []
        available_ce_tags = self.measurable_quantities.control_elements.tag_list
        # this ignores the cascade controllers, these mvs must be control elements
        improper_ce_tags = [
            c.mv_tag 
            for c in self.usable_quantities.controllers 
            if c.mv_tag not in available_ce_tags
        ]
        
        if len(improper_ce_tags) > 0:
            exception_group.append(
                ControllerConfigurationError(
                    "The following controlled variables are not defined as system control elements:"
                    f"{', '.join(improper_ce_tags)}."
                )
            )
        return exception_group
    
    def model_post_init(self, contex: Any) -> None:
        if self.use_numba:
            self._construct_fast_params()
        else:
            self._construct_params()
        if self.record_history:
            # Build history dict and accessors exactly once
            mq = self.measurable_quantities
            for mq_name in mq.model_dump():
                mq_obj: MeasurableBase = getattr(mq, mq_name)
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


    def _pre_integration_step(self) -> tuple[NDArray, NDArray]:
        """
        Handles the control loop logic before integration.

        This method updates all sensors and controllers based on the current state,
        sets the new values for the control elements, and returns the initial state
        array for the solver.
        """
        self.usable_quantities.update(self._t)
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
    
    def step(self, duration: Quantity|None = None) -> None:
        """
        The main public method to advance the simulation by one time step.

        Performs one integration step by calling the ODE solver.

        It configures and runs `solve_ivp`, then updates the system's algebraic
        states with the final, successful result.
        """
        show_progress = False
        if duration is None:
            nsteps = 1
        else:
            nsteps = round(duration.to(self.dt.unit).value / self.dt.value)
        if nsteps > 1:
            if logger.level == logging.NOTSET:
                # dont show progress bar if we are logging.
                show_progress = self.show_progress
        
        if show_progress:
            pbar = tqdm(total = nsteps)
        for _ in range(int(nsteps)):
            self._single_step()
            if show_progress:
                pbar.update(1)
        return 
    def _single_step(self):
        y_map = self._params['y_map']
        u_map = self._params['u_map']
        k_map = self._params['k_map']
        algebraic_map = self._params['algebraic_map']
        k = self._params['k']
        algebraic_values_function = self._params["algebraic_values_function"]
        rhs_function = self._params["rhs_function"]
        algebraic_size = self._params["algebraic_size"]
        y0, u0 = self._pre_integration_step()
        final_y = y0
        if self.measurable_quantities.states:
            result = solve_ivp(
                fun = self._rhs_wrapper,
                t_span = (self._t, self._t + self.dt.value),
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
        
        self._t += self.dt.value
        self._update_history()
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
                f"Tried to change trajectory of '{controller.cv_tag}' controller but failed - "
                f"controller must be in AUTO mode, but is {controller.mode.name}. "
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

    def set_controller_mode(self, cv_tag: str, mode: ControllerMode | str) -> ControllerMode:
        """Change the mode of a controller identified by ``cv_tag``."""
        if cv_tag not in self.controller_dictionary:
            raise ValueError(
                f"Specified cv_tag '{cv_tag}' does not correspond to any defined controllers. "
                f"Available controller cv_tags are {self.cv_tag_list}."
            )
        controller = self.controller_dictionary[cv_tag]
        controller.change_control_mode(mode)
        return controller.mode

    @cached_property
    def controller_dictionary(self) -> dict[str, "ControllerBase"]:
        return_dict = {}
        for controller in self.usable_quantities.controllers:
            return_dict.update({controller.cv_tag : controller})
            while controller.cascade_controller is not None:
                controller = controller.cascade_controller
                return_dict.update({controller.cv_tag : controller})
        return return_dict

    @cached_property
    def cv_tag_list(self) -> list[str]:
        return list(self.controller_dictionary.keys())

    @property
    def time(self) -> float:
        """Return the current simulation time."""
        return self._t

    @property
    def measured_history(self) -> dict[str, Any]:
        """Returns historized measurements and calculations."""

        sensors_detail: dict[str, list[TagData]] = {}
        calculations_detail: dict[str, list[TagData]] = {}
        history: dict[str, Any] = {
            "sensors": sensors_detail,
            "calculations": calculations_detail,
        }

        for sensor in self.usable_quantities.sensors:
            # use alias_tag for when it is different from measurement_tag
            sensors_detail[sensor.alias_tag] = sensor.measurement_history

        for calculation in self.usable_quantities.calculations:
            for output_tag_name, output_tag_info in calculation._output_tag_info_dict.items():
                calculations_detail[output_tag_name] = (output_tag_info.history)

        return history

    @property
    def history(self) -> dict[str, list]:
        """Returns a trimmed copy of the full history dictionary."""
        if not self.record_history:
            return {}
        return self._history

    @property
    def setpoint_history(self) -> dict[str, dict[str, list]]:
        """Returns historized controller setpoints keyed by ``cv_tag``."""
        history: dict[str, dict[str, list]] = {}
        for controller in self.usable_quantities.controllers:
            history.update(controller.sp_history)
        return history



