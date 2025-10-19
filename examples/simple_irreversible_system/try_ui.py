from modular_simulation.ui import launch_ui
from modular_simulation.measurables import MeasurableQuantities
from run_simulation import (
    initial_algebraic,
    initial_controls,
    initial_states,
    system_constants, 
    dt
)
from system_definitions import IrreversibleSystem
measurables = MeasurableQuantities(
    algebraic_states=initial_algebraic,
    constants = system_constants,
    control_elements=initial_controls,
    states = initial_states,
)
launch_ui(IrreversibleSystem, measurables, dt)