from modular_simulation.quantities import MeasurableQuantities, UsableQuantities, ControllableQuantities
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Any, Type, Dict, List, TYPE_CHECKING
from enum import Enum
from scipy.integrate import solve_ivp #type: ignore
from modular_simulation.measurables import ControlElements
from modular_simulation.validation import validate_and_link
import numpy as np
if TYPE_CHECKING:
    from modular_simulation.measurables import States, AlgebraicStates
    from modular_simulation.usables import Sensor, Calculation
    from modular_simulation.control_system import Controller
from copy import deepcopy

class System(ABC):
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
    measurable_quantities: MeasurableQuantities
    usable_quantities: UsableQuantities
    controllable_quantities: ControllableQuantities
    system_constants: Dict[str, Any]
    solver_options: Dict[str, Any]

    _history: Dict[str, NDArray]
    _measured_history: Dict[str, NDArray]

    def __init__(
            self, 
            measurable_quantities: MeasurableQuantities, 
            usable_quantities: UsableQuantities, 
            controllable_quantities: ControllableQuantities, 
            system_constants: Dict[str, Any],
            solver_options: Dict[str, Any],
            ):
        """
        Initializes the System object.

        Args:
            measurable_quantities: An object holding the system's state vectors.
            usable_quantities: An object defining sensors and calculations.
            controllable_quantities: An object defining the system's controllers.
            system_constants: A dictionary containing fixed parameters for the simulation.
            solver_options: A dictionary of keyword arguments to be passed to `solve_ivp`.
        """
        self.measurable_quantities = measurable_quantities
        self.usable_quantities = usable_quantities
        self.controllable_quantities = controllable_quantities
        
        self.system_constants = system_constants
        self.solver_options = solver_options

        validate_and_link(self)
        
        self._t = 0.
        self._history_size = 0
        self._measured_history_size = 0
        self._buffer_size = 10000
        self._history = {}
        self._measured_history = {}
        self._usable_snapshot = None
        self._update_history()

    def _update_history(self) -> None:
        """Creates a serializable dictionary snapshot of the current system state for logging."""
        if self._history_size % self._buffer_size == 0 and self._history_size != 0:
            for measurable_obj_name in self.measurable_quantities.__class__.model_fields.keys():
                measurable_obj = getattr(self.measurable_quantities, measurable_obj_name)
                for tag in measurable_obj.__class__.model_fields.keys():
                    self._history[tag].resize(len(self._history[tag]) + self._buffer_size)
            self._history['time'].resize(len(self._history[tag]) + self._buffer_size)
        if self._history == {}:
            for measurable_obj_name in self.measurable_quantities.__class__.model_fields.keys():
                measurable_obj = getattr(self.measurable_quantities, measurable_obj_name)
                for tag in measurable_obj.__class__.model_fields.keys():
                    self._history[tag] = np.zeros(self._buffer_size)
                    self._history[tag][self._history_size] = getattr(measurable_obj, tag)
            self._history['time'] = np.zeros(self._buffer_size)
            self._history['time'][self._history_size] = self._t
        else:
            for measurable_obj_name in self.measurable_quantities.__class__.model_fields.keys():
                measurable_obj = getattr(self.measurable_quantities, measurable_obj_name)
                for tag in measurable_obj.__class__.model_fields.keys():
                    self._history[tag][self._history_size] = getattr(measurable_obj, tag)
            self._history['time'][self._history_size] = self._t
        self._history_size += 1
            

    def _update_measured_history(self, snapshot) -> None:
        if self._history_size % self._buffer_size == 0 and self._measured_history_size != 0:
            for tag, s in snapshot.items():
                self._measured_history[tag].resize(len(self._measured_history[tag]) + self._buffer_size)
            self._measured_history['time'].resize(len(self._measured_history[tag]) + self._buffer_size)
        if self._measured_history == {}:
            self._measured_history = {tag: np.zeros(self._buffer_size) for tag, s in snapshot.items()}
            for tag, s in snapshot.items():
                self._measured_history[tag][self._measured_history_size] = s.value
            self._measured_history['time'] = np.zeros(self._buffer_size)
            self._measured_history['time'][self._measured_history_size] = self._t
        else:
            for tag, s in snapshot.items():
                self._measured_history[tag][self._measured_history_size]= s.value
            self._measured_history['time'][self._measured_history_size] = self._t
        self._measured_history_size += 1
            

    def _pre_integration_step(self) -> NDArray:
        """
        Handles the control loop logic before integration.
        
        This method updates all sensors and controllers based on the current state,
        sets the new values for the control elements, and returns the initial state
        array for the solver.
        """
        usable_results = self.usable_quantities.update(self.measurable_quantities, self._t)
        self._update_measured_history(usable_results)
        control_outputs = self.controllable_quantities.update(usable_results, self._t)
        self.measurable_quantities.control_elements.update(control_outputs)
        y0 = self.measurable_quantities.states.to_array()
        return y0

    @staticmethod
    @abstractmethod
    def _calculate_algebraic_values(
            y: NDArray, 
            StateMap: Type[Enum], 
            control_elements: ControlElements, 
            system_constants: Dict
            ) -> Dict[str, Any]:
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
            StateMap: Type[Enum], 
            algebraic_values_dict: Dict[str, Any],
            control_elements: ControlElements, 
            system_constants: Dict
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
            StateMap: Type[Enum],
            control_elements: ControlElements,
            system_constants: Dict[str, Any],
            ) -> NDArray:
        """
        A concrete wrapper called by the solver. It recalculates algebraic states
        before calling the user-defined `rhs` for the derivatives. This ensures
        correctness even when the solver rejects and retries steps.
        """
        algebraic_values_dict = self._calculate_algebraic_values(
            y=y,
            StateMap=StateMap,
            control_elements=control_elements,
            system_constants=system_constants
        )
        return self.rhs(t, y, StateMap, algebraic_values_dict, control_elements, system_constants)

    def _step(
            self, 
            dt: float, 
            y0: NDArray
            ) -> NDArray:
        """
        Performs one integration step by calling the ODE solver.

        It configures and runs `solve_ivp`, then updates the system's algebraic
        states with the final, successful result.
        """
        # This tuple contains everything that is constant for the duration of the step.
        static_params = (
            self.measurable_quantities.states.__class__.StateMap,
            self.measurable_quantities.control_elements,
            self.system_constants,
        )

        result = solve_ivp(
            fun=self._rhs_wrapper,
            t_span=(0, dt),
            y0=y0,
            args=static_params,
            **self.solver_options
        )
        
        final_y = result.y[:, -1]

        # After the final SUCCESSFUL step, update the actual algebraic_states object.
        if self.measurable_quantities.algebraic_states:
            final_algebraic_values = self._calculate_algebraic_values(
                y=final_y,
                StateMap=self.measurable_quantities.states.__class__.StateMap,
                control_elements=self.measurable_quantities.control_elements,
                system_constants=self.system_constants
            )
            self.measurable_quantities.algebraic_states = \
                self.measurable_quantities.algebraic_states.__class__(**final_algebraic_values)

        return final_y
    
    def step(self, dt: float) -> None:
        """
        The main public method to advance the simulation by one time step.
        
        Args:
            dt: The duration of the time step.
        """
        y0 = self._pre_integration_step()
        #printable = self.measurable_quantities.states.from_array(y0).model_dump()
        #printable.update(self.measurable_quantities.control_elements.model_dump())
        #printable = {key: f"{val:.3f}" for key, val in printable.items()}
        #print(printable)
        y = self._step(dt, y0)
        self.measurable_quantities.states = self.measurable_quantities.states.from_array(y)
        self._t += dt
        self._update_history()

    @property
    def measured_history(self) -> Dict[str, NDArray]:
        """Returns a trimmed copy of the measured history dictionary."""
        return {k: v[:self._measured_history_size].copy() for k, v in self._measured_history.items()}
    @property
    def history(self) -> Dict[str, NDArray]:
        """Returns a trimmed copy of the full history dictionary."""
        return {k: v[:self._history_size].copy() for k, v in self._history.items()}


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
            control_elements: ControlElements, 
            system_constants: Dict
            ) -> NDArray: 
        """
        Overrides the base wrapper to execute the performant path.

        This method converts Python objects (dicts, Pydantic models) into simple
        NumPy arrays before calling the user-defined `rhs_fast` function.
        """
        constants_arr = np.array([system_constants[k] for k in self._constants_map])
        control_elements_arr = np.array([getattr(control_elements, k) for k in self._controls_map])
        
        algebraic_states_arr = self.__class__._calculate_algebraic_values_fast(
            y=y, 
            control_elements_arr = control_elements_arr, 
            constants_arr = constants_arr
        )
        #print(f"Fout = {algebraic_states_arr[0]:.1f}")
        return self.__class__.rhs_fast(
            t = t, 
            y = y, 
            algebraic_states_arr = algebraic_states_arr, 
            control_elements_arr = control_elements_arr, 
            constants_arr = constants_arr
            )

    def _step(self, dt: float, y0: NDArray) -> NDArray:
        """Overrides the base `_step` to handle the performant path's data conversion."""
        static_params = (
            self.measurable_quantities.control_elements,
            self.system_constants,
        )

        result = solve_ivp(
            fun=self._rhs_wrapper, t_span=(0, dt), y0=y0, args=static_params, **self.solver_options
        )
        
        final_y = result.y[:, -1]

        if self.measurable_quantities.algebraic_states:
            constants_arr = np.array([self.system_constants[k] for k in self._constants_map])
            controls_arr = np.array([getattr(self.measurable_quantities.control_elements, k) for k in self._controls_map])
            final_alg_arr = self.__class__._calculate_algebraic_values_fast(
                y = final_y, 
                control_elements_arr = controls_arr, 
                constants_arr = constants_arr
            )
            # Convert the final array result back into a dictionary for Pydantic model creation.
            final_algebraic_values = dict(zip(self.measurable_quantities.algebraic_states.__class__.model_fields.keys(), final_alg_arr))
            self.measurable_quantities.algebraic_states = \
                self.measurable_quantities.algebraic_states.__class__(**final_algebraic_values)
        return final_y




def create_system(
        system_class: Type[System],
        initial_states: "States", 
        initial_controls: "ControlElements", 
        initial_algebraic: "AlgebraicStates", 
        measurement_definitions: Dict[str,"Sensor"],
        calculation_definitions: Dict[str, "Calculation"],
        control_definitions: Dict[str, "Controller"],
        system_constants: Dict[str, Any],
        solver_options: Dict[str, Any]
        ) -> System:
    """
    Factory to build a complete, internally consistent simulation system. 
    Creates copies of the objects passed in so as to ensure no cross-contamination
    between multiple systems created with the same inputs.
    """
    # 1. Create the components for this specific system instance
    copied_states = deepcopy(initial_states)
    copied_controls = deepcopy(initial_controls)
    copied_algebraic = deepcopy(initial_algebraic)
    copied_controllers = deepcopy(control_definitions)
    copied_measurements = deepcopy(measurement_definitions)
    copied_calculations = deepcopy(calculation_definitions)
    copied_constants = deepcopy(system_constants)

    measurables = MeasurableQuantities(
        states=copied_states,
        control_elements=copied_controls,
        algebraic_states=copied_algebraic
    )
    
    usables = UsableQuantities(
        measurement_definitions=copied_measurements,
        calculation_definitions=copied_calculations,
    )
    
    # Re-create controllers to ensure their internal states are fresh
    

    # 2. Link them correctly during construction
    # The UsableQuantities must be created before the ControllableQuantities
    # because the controllers depend on the sensors being defined.
    # The instance of usable_quantities here HAS to be the same 
    # as the one defined above. 
    controllables = ControllableQuantities(
        control_definitions=copied_controllers,
    )

    # 3. Assemble the final system object
    system = system_class(
        measurable_quantities=measurables,
        usable_quantities=usables,
        controllable_quantities=controllables,
        system_constants=copied_constants,
        solver_options=solver_options
    )
    
    return system
