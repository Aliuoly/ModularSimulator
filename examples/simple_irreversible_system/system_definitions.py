import numpy as np
from typing import Mapping
from numpy.typing import NDArray
from modular_simulation.measurables import States, ControlElements, AlgebraicStates, Constants
from modular_simulation.framework import System

# 1. Define the Data Structures for the System
# ============================================

class IrreversibleStates(States):
    """Pydantic model for the differential states of the system."""
    V: float
    A: float
    B: float

class IrreversibleControlElements(ControlElements):
    F_in: float  # Inlet flow rate

class IrreversibleAlgebraicStates(AlgebraicStates):
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
        y_map: Mapping[str, slice],
        u_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
        algebraic_size: int,
        ) -> NDArray:
        """
        Calculates the outlet flow (F_out) based on the current reactor volume.
        This is the algebraic part of the DAE system.
        """
        result = np.zeros(algebraic_size)
        # Ensure volume doesn't go to zero to prevent division errors.
        volume = max(1e-6, y[y_map["V"]][0]) 
        Cv = k[k_map["Cv"]][0]
        # F_out = Cv * sqrt(V)
        result[algebraic_map["F_out"]][0] =  Cv * (volume**0.5)  # type: ignore[arg-type]
        return result

    @staticmethod
    def rhs(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        algebraic: NDArray,
        y_map: Mapping[str, slice],
        u_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
        ) -> NDArray:
        """
        Calculates the derivatives for the differential states (dV/dt, dA/dt, dB/dt).
        This is the differential part of the DAE system.
        """
        # Unpack values from the inputs
        F_out = algebraic[algebraic_map["F_out"]][0] 
        F_in = u[u_map["F_in"]][0] 
        kc = k[k_map["k"]][0] 
        CA_in = k[k_map["CA_in"]][0] 

        # Unpack current state values using the StateMap for clarity
        volume = max(1e-6, y[y_map["V"]][0]) 
        molarity_A = y[y_map["A"]][0] 
        molarity_B = y[y_map["B"]][0] 
        
        # Calculate reaction rate: r = k * [A] * V
        reaction_rate = molarity_A * volume * kc
        
        # Initialize the derivative array
        dy = np.zeros_like(y)
        
        # Calculate the derivatives
        dV_dt = F_in - F_out
        dy[y_map["V"]] = dV_dt 
        dy[y_map["A"]] = (1/volume) * (
            -reaction_rate + F_in * CA_in - F_out * molarity_A - molarity_A * dV_dt
        ) 
        dy[y_map["B"]] = (1/volume) * (
            2*reaction_rate - F_out * molarity_B - molarity_B * dV_dt
        ) 

        return dy