import numpy as np
import numba
from enum import Enum
from typing import List, Type, ClassVar
from numpy.typing import NDArray

from modular_simulation.system import FastSystem
from modular_simulation.measurables import States, ControlElements, AlgebraicStates
from modular_simulation.usables import Sensor
from modular_simulation.control_system import Controller, Trajectory

# --- Reusable Pydantic Models for Data Structure ---
# These are the same as the readable version, but F_out has been correctly
# moved from a differential state to an algebraic state.

class IrreversableStateMap(Enum):
    V = 0  # Reactor volume [L]
    A = 1  # Concentration of A [mol/L]
    B = 2  # Concentration of B [mol/L]

class IrreversableStates(States):
    StateMap: ClassVar[Type[Enum]] = IrreversableStateMap
    V: float
    A: float
    B: float

class IrreversableControlElements(ControlElements):
    F_in: float  # Inlet flow rate (of A) [L/s]

class IrreversableAlgebraicStates(AlgebraicStates):
    F_out: float # Outlet flow rate [L/s]


# --- Fast System Implementation ---

class IrreversableFastSystem(FastSystem):
    """
    A performance-optimized implementation of the irreversible reaction system.

    The core logic in `rhs_fast` is JIT-compiled with Numba for high speed.
    """

    # --- Contract for FastSystem: Define array mappings ---
    
    @staticmethod
    def _get_constants_map() -> List[str]:
        """Returns the ordered list of keys for the constants dictionary."""
        return ['k', 'Cv', 'CA_in']

    @staticmethod
    def _get_controls_map() -> List[str]:
        """Returns the ordered list of keys for the control elements model."""
        return ['F_in']

    # --- Core Dynamics for FastSystem ---

    @staticmethod
    @numba.jit(nopython=True)
    # This function is not JIT-compiled because it uses an Enum (StateMap),
    # but it's called outside the solver's hot loop, so performance is not critical.
    def _calculate_algebraic_values_fast(
        y: NDArray,
        control_elements_arr: NDArray, # not used, but is here to follow _rhs_wrapper's usage pattern. See FastSystem definition. 
        constants_arr: NDArray
    ) -> NDArray:
        """
        Calculates F_out based on the algebraic relationship F_out = Cv * sqrt(V).
        """
        # Unpack constants using the map defined above
        # Index 1 corresponds to 'Cv'
        Cv = constants_arr[1]
        
        volume = max(1e-6, y[0]) # NOTE: HAS TO MATCH StateMap Enum, but is hardcoded due to Numba limitations
        
        # Calculate and return the algebraic state(s) as a NumPy array
        F_out = Cv * (volume)**0.5
        return np.array([F_out])

    @staticmethod
    @numba.jit(nopython=True)
    def rhs_fast(
        t: float,
        y: NDArray,
        constants_arr: NDArray,
        control_elements_arr: NDArray,
        algebraic_states_arr: NDArray,
    ) -> NDArray:
        """
        Calculates derivatives using only NumPy arrays. This method is JIT-compiled.
        
        NOTE: Because this is a Numba-compiled function, we cannot use the Enum
        and must use hard-coded integer indices for maximum performance.
        This is the primary trade-off for using the FastSystem path.
        """
        # --- Unpack States (hard-coded indices for Numba) ---
        V_idx, A_idx, B_idx = 0, 1, 2 # NOTE: HAS TO MATCH THE StateMap, even though it had to be hardcoded here due to NUMBA limitations.
        volume = max(1e-6, y[V_idx])
        molarity_A = y[A_idx]
        molarity_B = y[B_idx]
        
        # --- Unpack other arrays (hard-coded indices) ---
        k, _, CA_in = constants_arr[0], constants_arr[1], constants_arr[2]
        F_in = control_elements_arr[0]
        F_out = algebraic_states_arr[0]
        
        # --- Calculate Reaction Rate & Derivatives ---
        r = molarity_A * volume * k
        
        dy = np.zeros_like(y)
        dV_dt = F_in - F_out
        
        dy[V_idx] = dV_dt
        dy[A_idx] = (1/volume) * (-r + F_in * CA_in - F_out * molarity_A - molarity_A * dV_dt)
        dy[B_idx] = (1/volume) * (2*r - F_out * molarity_B - molarity_B * dV_dt)
        
        return dy

# --- Sensor, controller, and Trajectory classes remain the same ---
# (They are independent of the System implementation)

class FlowOutSensor(Sensor):
    def measure(self, measurable_quantities):
        # Correctly measures from the algebraic_states model now
        return measurable_quantities.algebraic_states.F_out

class FlowInSensor(Sensor):
    def measure(self, measurable_quantities):
        return measurable_quantities.control_elements.F_in

class BConcentrationSensor(Sensor):
    def measure(self, measurable_quantities):
        return measurable_quantities.states.B

class VolumeSensor(Sensor):
    def measure(self, measurable_quantities):
        return measurable_quantities.states.V

class PIDController(Controller):
    """A simple Proportional-Integral controller."""
    def __init__(self, sp_trajectory, pv_tag, Kp, Ti):
        self.sp_trajectory = sp_trajectory
        self.pv_tag = pv_tag
        self.Ti = Ti
        self.Kp = Kp
        self.last_error = 0
        self.integral = 0
        self._t_previous = 0.0

    def update(self, usable_results, t):
        dt = t - self._t_previous
        self._t_previous = t
        
        measured_pv = usable_results[self.pv_tag]
        setpoint = self.sp_trajectory(t)
        
        error = setpoint - measured_pv
        self.integral += error * dt
        
        # PI control law
        correction = self.Kp * error + (self.Kp / self.Ti) * self.integral
        
        # Ensure output is non-negative (e.g., flow rate can't be negative)
        return max(0.0, correction)

class ConstantTrajectory(Trajectory):
    def __init__(self, value):
        self.value = value
    def __call__(self, t):
        return self.value
    def change(self, value):
        self.value = value

