from abc import ABC
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from typing import Any
from collections.abc import Callable
from scipy.integrate import solve_ivp #type: ignore
from functools import cached_property
from numba.typed.typeddict import Dict as NDict #type: ignore
from numba import types, jit #type: ignore
import warnings
from modular_simulation.usables import (
    SensorBase, 
    ControllerBase, 
    CalculationBase, 
    ControllerMode, 
    Trajectory,
    TagData,
    
)
from modular_simulation.usables.tag_info import TagInfo
from modular_simulation.measurables.process_model import ProcessModel
from modular_simulation.utils.wrappers import second
from modular_simulation.utils.typing import TimeValue, StateValue
from modular_simulation.validation.exceptions import (
    SensorConfigurationError, 
    ControllerConfigurationError
)
import logging
from tqdm import tqdm #type: ignore
logger = logging.getLogger(__name__)

class System(BaseModel):
    """
    """
    dt: TimeValue = Field(
        default = ...,
        description = (
            "How often the system's sensors, calculations, and controllers update. "
            "The solver takes adaptive 'internal steps' regardless of this value to update the system's states. "
        )
    )
    process_model: ProcessModel = Field(
        ...,
        description = (
            "A container for all constants, state, control, and algebraic variables that can be measured or recorded. "
        )
    )
    sensors: list[SensorBase] = Field(
        ...,
    )
    controllers: list[ControllerBase] = Field(
        ...,
    )
    calculations: list[CalculationBase] = Field(
        ...,
    )
    solver_options: dict[str, Any] = Field(
        default_factory = lambda : {'method': 'LSODA'},
        description = (
            "arguments to scipy.integrate.solve_ivp call done by the system when taking steps."
        )
    )
    use_numba: bool = Field(
        default = False,
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

    _history: dict[str, list[StateValue]] = PrivateAttr(default_factory = dict)
    _dt_in_seconds: float = PrivateAttr()
    _t_in_seconds: float = PrivateAttr() 
    _params: dict[str, Any] = PrivateAttr() # NDict for numba implementation, dict[str, slice] otherwise

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    # add private attrs
    _history_slots: list[tuple[Callable[[Any], Any], list]] = PrivateAttr(default_factory=list)

    def _validate(self):
        exception_group = []
        exception_group.extend(self._validate_sensors_resolvable())
        exception_group.extend(self._validate_controllers_resolvable())
        
        if len(exception_group) > 0:
            raise ExceptionGroup(
                "errors encountered during usable quantity instantiation:", 
                exception_group
                )
    
    def _validate_sensors_resolvable(self):
        exception_group = []
        available_measurement_tags = self.process_model.state_list
        unavailable_measurement_tags = []

        for sensor in self.sensors:
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
        available_ce_tags = self.process_model.controlled_view.state_list
        # this ignores the cascade controllers, these mvs must be control elements
        improper_ce_tags = [
            c.mv_tag 
            for c in self.controllers 
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
    
    def model_post_init(self, context: Any) -> None:
        
        self._validate()

        model = self.process_model
        self._t_in_seconds = model.t
        self._dt_in_seconds = self.dt.to_value("second")

        if self.use_numba:
            self._construct_fast_params()
        else:
            self._construct_params()
        if self.record_history:
            # Build history dict and accessors exactly once
            for state in model.model_dump():
                mq_obj: StateValue = getattr(model, state)
                lst = [] # type:ignore
                self._history[state] = lst
                getter = model.make_converted_getter(state, target_unit=None)
                # capture the object reference and append list ref
                self._history_slots.append((lambda o=mq_obj, g=getter: g(o), lst)) #type:ignore[misc]
            self._history['time'] = []

        # pre-calculate the algebraic values once to refresh them with current states
        # and control elements and constants
        initial_algebraic_array = self._params["algebraic_values_function"](
            y = model.differential_view.to_array(),
            u = model.controlled_view.to_array(),
            k = self._params["k"],
            y_map = self._params["y_map"],
            u_map = self._params["u_map"],
            k_map = self._params["k_map"],
            algebraic_map = self._params["algebraic_map"],
            algebraic_size = self._params["algebraic_size"]
        )
        model.algebraic_view.update_from_array(initial_algebraic_array)

        
        for sensor in self.sensors:
            sensor.commission(self.time, model)
        for controller in self.controllers:
            controller.commission(self)
        for calculation in self.calculations:
            calculation.wire_inputs(self.time, self.tag_info_dict)

    def _construct_fast_params(self) -> None:
        """
        overwrites System's method of the same name to support numba njit decoration
        """
        model = self.process_model
        y_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = model.differential_view.index_map
        for member in index_map:
            y_map[member] = index_map[member if isinstance(member, slice) else slice(member, member+1)]

        u_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = model.controlled_view.index_map
        for member in index_map:
            u_map[member] = index_map[member if isinstance(member, slice) else slice(member, member+1)]
        
        k_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = model.constant_view.index_map
        for member in index_map:
            k_map[member] = index_map[member if isinstance(member, slice) else slice(member, member+1)]

        algebraic_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = model.algebraic_view.index_map
        for member in index_map:
            algebraic_map[member] = index_map[member if isinstance(member, slice) else slice(member, member+1)]

        algebraic_size = model.algebraic_view.array_size

        self._params = {
            'y_map': y_map,
            'u_map': u_map,
            'k_map': k_map,
            'algebraic_map': algebraic_map,
            'algebraic_size': algebraic_size,
            'k': model.constant_view.to_array(),
            'algebraic_values_function': jit(**self.numba_options)(model.calculate_algebraic_values),
            'rhs_function': jit(**self.numba_options)(model.differential_rhs),
        }

    def _construct_params(self) -> None:
        model = self.process_model
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

    def _update_history(self) -> None:
        if not self.record_history:
            return
        # simple tight loop: call getter, append to pre-bound list
        for getter, lst in self._history_slots:
            lst.append(getter()) #type:ignore[call-arg]
        self._history['time'].append(self._t_in_seconds)

    def _update_components(self) -> None:
        """
        Handles the control loop logic before integration.

        This method updates all sensors and controllers based on the current state,
        sets the new values for the control elements, and returns the initial state
        array for the solver.
        """
        t = self.time
        for s in self.sensors:
            s.measure(t) 
        for c in self.calculations:
            c.calculate(t)
        for c in self.controllers:
            c.update(t)
    
    def step(self, duration: TimeValue|None = None) -> None:
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
        # update sensors, calculations, and controllers
        # and then update the states with single_step()
        for _ in range(int(nsteps)):
            self._update_components()
            self._update_process()
            if show_progress:
                pbar.update(1)

    def _update_process(self):
        y_map = self._params['y_map']
        u_map = self._params['u_map']
        k_map = self._params['k_map']
        algebraic_map = self._params['algebraic_map']
        k = self._params['k']
        algebraic_values_function = self._params["algebraic_values_function"]
        rhs_function = self._params["rhs_function"]
        algebraic_size = self._params["algebraic_size"]
        differential_view = self.process_model.differential_view
        y = differential_view.to_array()
        u = self.process_model.controlled_view.to_array()
        end_t = self._t_in_seconds + self._dt_in_seconds
        result = solve_ivp(
            fun = self.process_model._rhs_wrapper,
            t_span = (self._t_in_seconds, end_t),
            y0 = y,
            args = (u, k, y_map, u_map, k_map, algebraic_map, algebraic_values_function, rhs_function, algebraic_size),
            **self.solver_options
        )
        final_y = result.y[:, -1]
        differential_view.update_from_array(final_y)

        # After the final SUCCESSFUL step, update the actual algebraic_states object.
        final_algebraic_values = algebraic_values_function(
            final_y, u, k, y_map, u_map, k_map, algebraic_map, algebraic_size
        )
        self.process_model.algebraic_view.update_from_array(final_algebraic_values)
        self.process_model.t = end_t
        self._t_in_seconds = end_t
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
            value = active_trajectory(self._t_in_seconds)

        old_value = active_trajectory(self._t_in_seconds)
        active_trajectory.set(t = self._t_in_seconds, value = value)
        new_value = active_trajectory(self._t_in_seconds)
        logger.info("Setpoint trajectory for '%(tag)s' controller changed at time %(time)0.0f " \
                        "from %(old)0.1e to %(new)0.1e",
                        {'tag': cv_tag, 'time': self._t_in_seconds, 'old': old_value, 'new': new_value})
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
    def tag_info_dict(self) -> dict[str, TagInfo]:
        """
        does not include the controller sp tags due to order of definition.
        controller sp tag info is created AFTER this property is accessed. 
        Instead, controllers will call the update_tag_info_dict once
        they are ready with their sp tag info.
        """
        info_dict = {}
        for s in self.sensors:
            info_dict[s.alias_tag] = s._tag_info
        for c in self.calculations:
            info_dict.update(c._output_tag_info_dict)
        return info_dict
    
    def update_tag_info_dict(self, tag: str, tag_info: TagInfo) -> None:
        """updates the tag_info_dict after it has been constructed"""
        self.tag_info_dict[tag] =  tag_info

    @cached_property
    def controller_dictionary(self) -> dict[str, ControllerBase]:
        return_dict = {}
        for controller in self.controllers:
            return_dict.update({controller.cv_tag : controller})
            while controller.cascade_controller is not None:
                controller = controller.cascade_controller
                return_dict.update({controller.cv_tag : controller})
        return return_dict

    @cached_property
    def cv_tag_list(self) -> list[str]:
        return list(self.controller_dictionary.keys())
    
    @property
    def time(self) -> TimeValue:
        """Return the current simulation time."""
        return second(self._t_in_seconds)

    @property
    def measured_history(self) -> dict[str, Any]:
        """Returns historized measurements and calculations."""

        sensors_detail: dict[str, list[TagData]] = {}
        calculations_detail: dict[str, list[TagData]] = {}
        history: dict[str, Any] = {
            "sensors": sensors_detail,
            "calculations": calculations_detail,
        }

        for sensor in self.sensors:
            # use alias_tag for when it is different from measurement_tag
            sensors_detail[sensor.alias_tag] = sensor.measurement_history

        for calculation in self.calculations:
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
        for controller in self.controllers:
            history.update(controller.sp_history)
        return history



