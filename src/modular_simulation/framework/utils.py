from modular_simulation.quantities.measurable_quantities import MeasurableQuantities
from modular_simulation.quantities.usable_quantities import UsableQuantities
from modular_simulation.quantities.controllable_quantities import ControllableQuantities
from typing import Any, Dict, List, Mapping, TYPE_CHECKING, Callable, Type
from modular_simulation.framework.system import System
if TYPE_CHECKING:
    from modular_simulation.measurables import States, AlgebraicStates, ControlElements, Constants
    from modular_simulation.usables import Sensor, Calculation
    from modular_simulation.control_system import Controller
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)

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
