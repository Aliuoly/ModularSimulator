from modular_simulation.quantities.measurable_quantities import MeasurableQuantities
from modular_simulation.quantities.usable_quantities import UsableQuantities
from modular_simulation.quantities.controllable_quantities import ControllableQuantities
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from numpy.typing import NDArray, ArrayLike
from typing import Any, Type, Dict, List, TYPE_CHECKING, Tuple, Callable
from enum import Enum, IntEnum
from scipy.integrate import solve_ivp #type: ignore
from modular_simulation.validation import validate_and_link
from functools import cached_property
from operator import attrgetter
import numpy as np
from modular_simulation.measurables.base_classes import BaseIndexedModel
if TYPE_CHECKING:
    from modular_simulation.measurables import States, AlgebraicStates, ControlElements, Constants
    from modular_simulation.usables import Sensor, Calculation, TimeValueQualityTriplet
    from modular_simulation.control_system import Controller, Trajectory
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)

class StaticParamEnum(IntEnum):
    STATE_MAP = 0
    CONTROL_MAP = 1
    CONSTANT_MAP = 2
    ALGEBRAIC_MAP = 3
    CONSTANTS = 4


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
        system_constants (Dict[str, Any]): A dictionary of constant parameters for the system.
        solver_options (Dict[str, Any]): A dictionary of options passed directly to
            SciPy's `solve_ivp` function.
        _t (float): The current simulation time.
        _history (List[Dict]): A log of the system's state at each time step.
    """
    dt: float = Field(
        ...,
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
    record_history: bool = Field(
        default = False,
        description = (
            "Whether or not to historize the underlying measurable_quantities of the system. "
            "The usable_quantities, which are measured or calculation, are always historized regardless. "
        )
    )

    _history: Dict[str, List[ArrayLike]] = PrivateAttr(default_factory = dict)
    _t: float = PrivateAttr(default = 0.)
    _rhs_static_params: Tuple[Type[Enum], Type[Enum], Type[Enum], Type[Enum], NDArray] = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    # add private attrs
    _history_slots: List[Tuple[Callable[[Any], Any], List]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        global StaticParamEnum
        super().model_post_init(__context)
        validate_and_link(self)

        params: List[Any] = [0] * 5
        params[StaticParamEnum.STATE_MAP]     = self.measurable_quantities.states._index_map
        params[StaticParamEnum.CONTROL_MAP]   = self.measurable_quantities.control_elements._index_map
        params[StaticParamEnum.CONSTANT_MAP]  = self.measurable_quantities.constants._index_map
        params[StaticParamEnum.ALGEBRAIC_MAP] = self.measurable_quantities.algebraic_states._index_map
        params[StaticParamEnum.CONSTANTS]     = self.measurable_quantities.constants.to_array()
        self._rhs_static_params = tuple(params)

        if self.record_history:
            # Build history dict and accessors exactly once
            mq = self.measurable_quantities
            for mq_name in mq.__class__.model_fields.keys():
                mq_obj: BaseIndexedModel = getattr(mq, mq_name)
                for tag in mq_obj.__class__.model_fields.keys():
                    lst = [] # type:ignore
                    self._history[tag] = lst
                    getter = attrgetter(tag)
                    # capture the object reference and append list ref
                    self._history_slots.append((lambda o=mq_obj, g=getter: g(o), lst)) #type:ignore[misc]
            self._history['time'] = []

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
    def _calculate_algebraic_values(
            y: NDArray, 
            y_map: Type[Enum], 
            u: NDArray,
            u_map: Type[Enum],
            k: NDArray,
            k_map: Type[Enum],
            ) -> NDArray:
        """
        Calculates derived quantities based on the provided state array, controls, and constants.

        This method MUST be stateless and depend only on its inputs. It will be called
        repeatedly by the ODE solver's internal steps.

        Args:
            y: The current values of the differential state variables as a flat NumPy array.
            StateMap: The Enum class used to map state names to indices in `y`.
            control_elements: The Pydantic object of control element values for the current step.
            system_constants: The dictionary of system constants.

        Returns:
            A dictionary of the calculated algebraic values.
        """
        pass

    @staticmethod
    @abstractmethod
    def rhs(
            t: float, 
            y: NDArray, 
            y_map: Type[Enum], 
            u: NDArray, 
            u_map: Type[Enum], 
            k: NDArray,
            k_map: Type[Enum], 
            algebraic: NDArray,
            algebraic_map: Type[Enum], 
            ) -> NDArray:
        """
        Defines the right-hand side of the system's ordinary differential equations.

        This method should ONLY contain the derivative calculations and must be stateless.

        Args:
            t: The current simulation time.
            y: The current values of the differential state variables as a flat NumPy array.
            StateMap: The Enum class used to map state names to indices in `y`.
            algebraic_values_dict: A dictionary of freshly calculated algebraic values for this `y`.
            control_elements: The Pydantic object of control element values for the current step.
            system_constants: The dictionary of system constants.

        Returns:
            A NumPy array of the calculated derivatives (dy/dt).
        """
        pass

    def _rhs_wrapper(
            self, 
            t: float, 
            y: NDArray, 
            u: NDArray,
            params: Tuple,
            ) -> NDArray:
        """
        A concrete wrapper called by the solver. It recalculates algebraic states
        before calling the user-defined `rhs` for the derivatives. This ensures
        correctness even when the solver rejects and retries steps.
        """
        global StaticParamEnum
        y_map = params[StaticParamEnum.STATE_MAP]
        u_map = params[StaticParamEnum.CONTROL_MAP]
        k = params[StaticParamEnum.CONSTANTS]
        k_map = params[StaticParamEnum.CONSTANT_MAP]
        algebraic_map = params[StaticParamEnum.ALGEBRAIC_MAP]

        algebraic_array = self._calculate_algebraic_values(
            y = y,
            y_map = y_map,
            u = u,
            u_map = u_map,
            k = k,
            k_map = k_map
        )
        return self.rhs(
            t, 
            y = y, y_map = y_map, 
            u = u, u_map = u_map,
            k = k, k_map = k_map,
            algebraic = algebraic_array, algebraic_map = algebraic_map
            )
    
    def step(self) -> None:
        """
        The main public method to advance the simulation by one time step.

        Performs one integration step by calling the ODE solver.

        It configures and runs `solve_ivp`, then updates the system's algebraic
        states with the final, successful result.
        """
        global StaticParamEnum

        y0, u0 = self._pre_integration_step()

        final_y = y0
        if self.measurable_quantities.states:
            result = solve_ivp(
                fun = self._rhs_wrapper,
                t_span = (self._t, self._t + self.dt),
                y0 = y0,
                args = (u0, self._rhs_static_params),
                **self.solver_options
            )
            final_y = result.y[:, -1]
            self.measurable_quantities.states.update_from_array(final_y)

        # After the final SUCCESSFUL step, update the actual algebraic_states object.
        if self.measurable_quantities.algebraic_states:
            final_algebraic_values = self._calculate_algebraic_values(
                y = final_y,
                y_map = self._rhs_static_params[StaticParamEnum.STATE_MAP], # type: ignore 
                u = u0,
                u_map = self._rhs_static_params[StaticParamEnum.CONTROL_MAP], # type: ignore 
                k = self._rhs_static_params[StaticParamEnum.CONSTANTS], # type: ignore 
                k_map = self._rhs_static_params[StaticParamEnum.CONSTANT_MAP], # type: ignore 
            )
            self.measurable_quantities.algebraic_states.update_from_array(final_algebraic_values)
        
        self._t += self.dt
        self._update_history()
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
        active_trajectory = controller.active_sp_trajectory()

        if value is None:
            value = active_trajectory.current_value(self._t)

        old_value = active_trajectory(self._t)
        controller.update_trajectory(self._t, value)
        new_value = controller.active_sp_trajectory()(self._t)
        logger.info("Setpoint trajectory for '%(tag)s' controller changed at time %(time)0.0f " \
                        "from %(old)0.1e to %(new)0.1e",
                        {'tag': cv_tag, 'time': self._t, 'old': old_value, 'new': new_value})
        return controller.active_sp_trajectory()
        
    @cached_property
    def controller_dictionary(self) -> Dict[str, "Controller"]:
        return {c.cv_tag: c for c in self.controllable_quantities.controllers}

    @cached_property
    def cv_tag_list(self) -> List[str]:
        return [c.cv_tag for c in self.controllable_quantities.controllers]

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
            sensors_detail[sensor.measurement_tag] = sensor.measurement_history()

        for calculation in self.usable_quantities.calculations:
            calculations_detail[calculation.output_tag] = calculation.history()

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
            traj_hist = controller.sp_trajectory.history()
            history[controller.cv_tag] = {
                "time": traj_hist["time"].copy(),
                "value": traj_hist["value"].copy(),
            }
        return history


class FastSystem(System):
    """
    An abstract base class for performance-optimized systems.

    Inherit from this class when simulation speed is critical. It requires implementing
    a "fast" contract where the core dynamics operate exclusively on NumPy arrays,
    making them compatible with JIT compilers like Numba.

    This class overrides the standard simulation loop to call the `_fast` methods.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the FastSystem, creating the key-to-index mappings."""
        super().__init__(*args, **kwargs)
        self._constants_map = self._get_constants_map()
        self._controls_map = self._get_controls_map()

    # Silences the abstract methods from the parent `System` class.
    # Users of FastSystem do not implement these readable methods.
    @staticmethod
    def _calculate_algebraic_values(*args): pass

    @staticmethod
    def rhs(*args) : pass

    # --- Performant Path Abstract Methods ---
    @staticmethod
    @abstractmethod
    def _get_constants_map() -> List[str]:
        """Return an ordered list of keys for the constants dictionary."""
        pass

    @staticmethod
    @abstractmethod
    def _get_controls_map() -> List[str]:
        """Return an ordered list of keys for the control elements."""
        pass
    
    @staticmethod
    @abstractmethod
    def _calculate_algebraic_values_fast(y: NDArray, control_elements_arr: NDArray, constants_arr: NDArray) -> NDArray:
        """(Fast Path) Calculates algebraic values using only NumPy arrays."""
        pass

    @staticmethod
    @abstractmethod
    def rhs_fast(t: float, y: NDArray, algebraic_states_arr: NDArray, control_elements_arr: NDArray, constants_arr: NDArray) -> NDArray:
        """(Fast Path) Calculates derivatives using only NumPy arrays."""
        pass

    def _rhs_wrapper( #type: ignore
            self, 
            t: float, 
            y: NDArray, 
            control_elements_arr: NDArray, 
            constants_arr: NDArray
            ) -> NDArray: 
        """
        Overrides the base wrapper to execute the performant path.

        This method converts Python objects (dicts, Pydantic models) into simple
        NumPy arrays before calling the user-defined `rhs_fast` function.
        """
        algebraic_states_arr = self.__class__._calculate_algebraic_values_fast(
            y,
            control_elements_arr,
            constants_arr,
        )
        return self.__class__.rhs_fast(
            t,
            y,
            algebraic_states_arr,
            control_elements_arr,
            constants_arr,
        )

    def _step(self, dt: float, y0: NDArray) -> NDArray:
        """Overrides the base `_step` to handle the performant path's data conversion."""

        constants_arr = np.asarray([self.system_constants[k] for k in self._constants_map], dtype=np.float64)
        control_elements_arr = np.asarray(
            [getattr(self.measurable_quantities.control_elements, k) for k in self._controls_map],
            dtype=np.float64,
        )
        static_params = (
            control_elements_arr,
            constants_arr, # order matters lol
        )
        result = solve_ivp(
            fun=self._rhs_wrapper, t_span=(0, dt), y0=y0, args=static_params, **self.solver_options
        )
        
        final_y = result.y[:, -1]

        if self.measurable_quantities.algebraic_states:
            constants_arr = np.asarray([self.system_constants[k] for k in self._constants_map], dtype=np.float64)
            controls_arr = np.asarray(
                [getattr(self.measurable_quantities.control_elements, k) for k in self._controls_map],
                dtype=np.float64,
            )
            final_alg_arr = self.__class__._calculate_algebraic_values_fast(
                final_y,
                controls_arr,
                constants_arr,
            )
            self.measurable_quantities.algebraic_states.update_from_array(final_alg_arr)
        return final_y




def create_system(
        system_class: Type[System],
        dt: float,
        initial_states: "States",
        initial_controls: "ControlElements",
        initial_algebraic: "AlgebraicStates",
        system_constants: "Constants",
        sensors: List["Sensor"],
        calculations: List["Calculation"],
        controllers: List["Controller"],
        *,
        solver_options: Dict[str, Any] = {'method': 'LSODA'},
        record_history: bool = True,
        ) -> System:
    """
    Factory to build a complete, internally consistent simulation system.
    Creates copies of the objects passed in so as to ensure no cross-contamination
    between multiple systems created with the same inputs.

    The optional ``record_history`` flags allow
    callers to disable internal historization of states to save memory when full logs are
    unnecessary. Measurements are still historized regardless.
    """
    # 1. Create the components for this specific system instance
    copied_states = deepcopy(initial_states)
    copied_controls = deepcopy(initial_controls)
    copied_algebraic = deepcopy(initial_algebraic)
    copied_sensors = deepcopy(sensors)
    copied_calculations = deepcopy(calculations)
    copied_controllers = deepcopy(controllers)
    copied_constants = deepcopy(system_constants)

    measurables = MeasurableQuantities(
        states=copied_states,
        control_elements=copied_controls,
        algebraic_states=copied_algebraic,
        constants = copied_constants,
    )
    
    usables = UsableQuantities(
        sensors=copied_sensors,
        calculations=copied_calculations,
    )
    
    # Re-create controllers to ensure their internal states are fresh
    

    # 2. Link them correctly during construction
    # The UsableQuantities must be created before the ControllableQuantities
    # because the controllers depend on the sensors being defined.
    # The instance of usable_quantities here HAS to be the same 
    # as the one defined above. 
    controllables = ControllableQuantities(
        controllers=copied_controllers,
    )

    # link measurables to usables
    
    # 3. Assemble the final system object
    system = system_class(
        dt = dt,
        measurable_quantities=measurables,
        usable_quantities=usables,
        controllable_quantities=controllables,
        solver_options=solver_options,
        record_history=record_history,
    )
    
    return system
