from modular_simulation.framework import System
from numpy.typing import NDArray
from typing import Mapping
import numpy as np
from modular_simulation.measurables import (
    States, 
    AlgebraicStates,
    Constants,
    ControlElements
)


# tank A -> tank B
# tank A and B outlets are pressure driven flow
# tank A outlet becomes tank B inlet

class TankAStates(States):
    V_A: float

class TankAControlElements(ControlElements):
    F_in_A: float

class TankAAlgebraicStates(AlgebraicStates):
    F_out_A: float

class TankAConstants(Constants):
    Cv_F_out_A: float



class TankBStates(States):
    V_B: float

class TankBControlElements(ControlElements):
    pass

class TankBAlgebraicStates(AlgebraicStates):
    F_in_B: float
    F_out_B: float
    
class TankBConstants(Constants):
    Cv_F_out_B: float

class TankASystem(System):
    
    @staticmethod
    def rhs(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        algebraic: NDArray,
        u_map: Mapping[str, slice],
        y_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
        ) -> NDArray:

        F_in = u[u_map["F_in_A"]][0]
        F_out = algebraic[algebraic_map["F_out_A"]][0]

        dy = np.zeros_like(y)
        dy[y_map["V_A"]] = F_in - F_out

        return dy
    
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
        algebraic_states = np.zeros(algebraic_size)
        Cv = k[k_map["Cv_F_out_A"]][0]
        V_A = y[y_map["V_A"]][0]
        V_B = y[y_map["V_B"]][0]
        diffV = (V_A - V_B)
        algebraic_states[algebraic_map["F_out_A"]] = Cv * abs(diffV)**0.5 * np.sign(diffV)
        algebraic_states[algebraic_map["F_in_B"]] = algebraic_states[algebraic_map["F_out_A"]]
        return algebraic_states
    
class TankBSystem(System):
    @staticmethod
    def rhs(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        algebraic: NDArray,
        u_map: Mapping[str, slice],
        y_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
        ) -> NDArray:

        F_in = algebraic[algebraic_map["F_in_B"]][0]
        F_out = algebraic[algebraic_map["F_out_B"]][0]

        dy = np.zeros_like(y)
        dy[y_map["V_B"]] = F_in - F_out

        return dy
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
        algebraic_states = np.zeros(algebraic_size)
        Cv = k[k_map["Cv_F_out_B"]][0]
        V_B = y[y_map["V_B"]][0]
        algebraic_states[algebraic_map["F_out_B"]] = Cv * V_B**0.5
        return algebraic_states

