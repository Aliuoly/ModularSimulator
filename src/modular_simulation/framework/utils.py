from modular_simulation.measurables.measurable_quantities import MeasurableQuantities
from modular_simulation.usables.usable_quantities import UsableQuantities
from typing import Any,  TYPE_CHECKING
from modular_simulation.framework.system_old import System
if TYPE_CHECKING:
    from modular_simulation.measurables import States, AlgebraicStates, ControlElements, Constants
    from modular_simulation.usables import SensorBase, ControllerBase, CalculationBase
import logging
from astropy.units import Quantity #type: ignore
logger = logging.getLogger(__name__)

def create_system(
        system_class: type[System],
        dt: Quantity,
        initial_states: "States",
        initial_controls: "ControlElements",
        initial_algebraic: "AlgebraicStates",
        system_constants: "Constants",
        sensors: list["SensorBase"],
        calculations: list["CalculationBase"],
        controllers: list["ControllerBase"],
        *,
        use_numba: bool = False,
        numba_options: dict[str, Any] = {'nopython': True, 'cache': True},
        solver_options: dict[str, Any] = {'method': 'LSODA'},
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
    copied_states = initial_states.model_copy()
    copied_controls = initial_controls.model_copy()
    copied_algebraic = initial_algebraic.model_copy()
    copied_constants = system_constants.model_copy()
    copied_sensors = [s.model_copy() for s in sensors]
    copied_calculations = [c.model_copy() for c in calculations]
    copied_controllers = [c.model_copy() for c in controllers]

    measurables = MeasurableQuantities(
        states=copied_states,
        control_elements=copied_controls,
        algebraic_states=copied_algebraic,
        constants = copied_constants,
    )
    
    usables = UsableQuantities(
        sensors=copied_sensors,
        calculations=copied_calculations,
        controllers=copied_controllers,
    )

    # link measurables to usables
    
    # 3. Assemble the final system object
    system = system_class(
        dt = dt,
        measurable_quantities=measurables,
        usable_quantities=usables,
        solver_options=solver_options,
        record_history=record_history,
        use_numba=use_numba,
        numba_options = numba_options,
    )
    
    return system
