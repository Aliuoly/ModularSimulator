import numpy as np
from pydantic import ConfigDict, Field
from enum import Enum
from typing import Type, ClassVar, List
import numba #type: ignore

from modular_simulation.measurables import States, ControlElements, AlgebraicStates, Constants
from modular_simulation.control_system import Trajectory
from modular_simulation.system import System, FastSystem
from numpy.typing import NDArray

# 1. Define the Data Structures for the System
# ============================================

class IrreversibleStateMap(Enum):
    """Maps the differential state names to their index in the NumPy array."""
    V = 0  # Reactor Volume
    A = 1  # Concentration of A
    B = 2  # Concentration of B

class IrreversibleStates(States):
    """Pydantic model for the differential states of the system."""
    model_config = ConfigDict(extra='forbid')
    StateMap: ClassVar = IrreversibleStateMap
    V: float = Field()
    A: float
    B: float

class IrreversibleControlElements(ControlElements):
    """Pydantic model for the externally controlled variables."""
    F_in: float  # Inlet flow rate

class IrreversibleAlgebraicStates(AlgebraicStates):
    """Pydantic model for states that are algebraic functions of other variables."""
    model_config = ConfigDict(extra='forbid')
    F_out: float # Outlet flow rate, an algebraic function of Volume

class IrreversibleConstants(Constants):
    """"""
    k: float
    Cv: float
    CA_in: float
    
# 2. Define the System Dynamics
# =============================

class IrreversibleSystem(System):
    """
    Implements the 'readable' contract from the `System` base class to define
    the dynamics of the irreversible reaction A -> 2B.
    """

    @staticmethod
    def _calculate_algebraic_values(
            y: NDArray, 
            y_map: Type[Enum], 
            u: NDArray,
            u_map: Type[Enum],
            k: NDArray,
            k_map: Type[Enum],
            ) -> NDArray:
        """
        Calculates the outlet flow (F_out) based on the current reactor volume.
        This is the algebraic part of the DAE system.
        """
        # Ensure volume doesn't go to zero to prevent division errors.
        volume = max(1e-6, y[y_map.V.value]) #type: ignore
        
        # F_out = Cv * sqrt(V)
        F_out = k[k_map.Cv.value]* (volume**0.5) #type: ignore
        
        return F_out

    @staticmethod
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
        Calculates the derivatives for the differential states (dV/dt, dA/dt, dB/dt).
        This is the differential part of the DAE system.
        """
        # Unpack values from the inputs
        F_out = algebraic[algebraic_map.F_out.value] #type: ignore
        F_in = u[u_map.F_in.value] #type: ignore
        kc = k[k_map.k.value] #type: ignore
        CA_in = k[k_map.CA_in.value] #type: ignore

        # Unpack current state values using the StateMap for clarity
        volume = max(1e-6, y[y_map.V.value]) #type: ignore
        molarity_A = y[y_map.A.value] #type: ignore
        molarity_B = y[y_map.B.value] #type: ignore
        
        # Calculate reaction rate: r = k * [A] * V
        reaction_rate = molarity_A * volume * kc
        
        # Initialize the derivative array
        dy = np.zeros_like(y)
        
        # Calculate the derivatives
        dV_dt = F_in - F_out
        dy[y_map.V.value] = dV_dt #type: ignore
        dy[y_map.A.value] = (1/volume) * (-reaction_rate + F_in * CA_in - F_out * molarity_A - molarity_A * dV_dt) #type: ignore 
        dy[y_map.B.value] = (1/volume) * (2*reaction_rate - F_out * molarity_B - molarity_B * dV_dt) #type: ignore

        return dy

# --- Fast System Implementation ---

class IrreversibleFastSystem(FastSystem):
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

class ConstantTrajectory(Trajectory):
    """Provides a constant setpoint value over time."""
    def __init__(self, value):
        self.value = value
    
    def __call__(self, t):
        return self.value
    
    def change(self, new_value):
        """Allows for changing the setpoint during a simulation."""
        self.value = new_value

