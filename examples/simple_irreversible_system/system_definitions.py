import numpy as np
from pydantic import ConfigDict, Field
from enum import Enum
from typing import Type, ClassVar
from numba import njit
from numba.typed.typeddict import Dict as NDict

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
    def calculate_algebraic_values(
        y: NDArray, 
        u: NDArray,
        k: NDArray,
        y_map: Type[Enum], 
        u_map: Type[Enum],
        k_map: Type[Enum],
        algebraic_map: Type[Enum]
        ) -> NDArray:
        """
        Calculates the outlet flow (F_out) based on the current reactor volume.
        This is the algebraic part of the DAE system.
        """
        result = np.zeros(len(algebraic_map))
        # Ensure volume doesn't go to zero to prevent division errors.
        volume = max(1e-6, y[y_map.V.value]) #type: ignore
        
        # F_out = Cv * sqrt(V)
        result[algebraic_map.F_out.value] = k[k_map.Cv.value] * (volume**0.5)  # type: ignore
        return result

    @staticmethod
    def rhs(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        algebraic: NDArray,
        y_map: Type[Enum],
        u_map: Type[Enum],
        k_map: Type[Enum],
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

    
    @staticmethod
    @njit
    # This function is not JIT-compiled because it uses an Enum (StateMap),
    # but it's called outside the solver's hot loop, so performance is not critical.
    def calculate_algebraic_values_fast(
        y: NDArray, 
        u: NDArray,
        k: NDArray,
        y_map: NDict, 
        u_map: NDict,
        k_map: NDict,
        algebraic_map: NDict
        ) -> NDArray:
        """
        Calculates the outlet flow (F_out) based on the current reactor volume.
        This is the algebraic part of the DAE system.
        """
        result = np.zeros(len(algebraic_map))
        # Ensure volume doesn't go to zero to prevent division errors.
        volume = max(1e-6, y[y_map['V']])
        
        # F_out = Cv * sqrt(V)
        result[algebraic_map["F_out"]] = k[k_map["Cv"]] * (volume**0.5) 
        return result

    @staticmethod
    @njit
    def rhs_fast(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        algebraic: NDArray,
        y_map: NDict,
        u_map: NDict,
        k_map: NDict,
        algebraic_map: NDict, 
    ) -> NDArray:
        # Unpack values from the inputs
        F_out = algebraic[algebraic_map["F_out"]] #type: ignore
        F_in = u[u_map["F_in"]] #type: ignore
        kc = k[k_map["k"]] #type: ignore
        CA_in = k[k_map["CA_in"]] #type: ignore

        # Unpack current state values using the StateMap for clarity
        volume = max(1e-6, y[y_map["V"]]) #type: ignore
        molarity_A = y[y_map["A"]] #type: ignore
        molarity_B = y[y_map["B"]] #type: ignore
        
        # Calculate reaction rate: r = k * [A] * V
        reaction_rate = molarity_A * volume * kc
        
        # Initialize the derivative array
        dy = np.zeros_like(y)
        
        # Calculate the derivatives
        dV_dt = F_in - F_out
        dy[y_map["V"]] = dV_dt #type: ignore
        dy[y_map["A"]] = (1/volume) * (-reaction_rate + F_in * CA_in - F_out * molarity_A - molarity_A * dV_dt) #type: ignore 
        dy[y_map["B"]] = (1/volume) * (2*reaction_rate - F_out * molarity_B - molarity_B * dV_dt) #type: ignore

        
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

