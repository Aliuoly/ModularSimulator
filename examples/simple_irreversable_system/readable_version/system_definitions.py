import numpy as np
from pydantic import ConfigDict, Field
from enum import Enum
from typing import Dict, Any, Type, ClassVar

from modular_simulation.measurables import States, ControlElements, AlgebraicStates
from modular_simulation.usables import Sensor
from modular_simulation.control_system import Controller, Trajectory
from modular_simulation.system import System
from numpy.typing import NDArray

# 1. Define the Data Structures for the System
# ============================================

class IrreversableStateMap(Enum):
    """Maps the differential state names to their index in the NumPy array."""
    V = 0  # Reactor Volume
    A = 1  # Concentration of A
    B = 2  # Concentration of B

class IrreversableStates(States):
    """Pydantic model for the differential states of the system."""
    model_config = ConfigDict(extra='forbid')
    StateMap: ClassVar = IrreversableStateMap
    V: float = Field()
    A: float
    B: float

class IrreversableControlElements(ControlElements):
    """Pydantic model for the externally controlled variables."""
    F_in: float  # Inlet flow rate

class IrreversableAlgebraicStates(AlgebraicStates):
    """Pydantic model for states that are algebraic functions of other variables."""
    model_config = ConfigDict(extra='forbid')
    F_out: float # Outlet flow rate, an algebraic function of Volume

# 2. Define the System Dynamics
# =============================

class IrreversableSystem(System):
    """
    Implements the 'readable' contract from the `System` base class to define
    the dynamics of the irreversible reaction A -> 2B.
    """

    @staticmethod
    def _calculate_algebraic_values(
            y: NDArray,
            StateMap: Type[Enum],
            control_elements: ControlElements,
            system_constants: Dict
            ) -> Dict[str, Any]:
        """
        Calculates the outlet flow (F_out) based on the current reactor volume.
        This is the algebraic part of the DAE system.
        """
        # Ensure volume doesn't go to zero to prevent division errors.
        volume = max(1e-6, y[StateMap.V.value])
        
        # F_out = Cv * sqrt(V)
        F_out = system_constants['Cv'] * (volume**0.5)
        
        return {"F_out": F_out}

    @staticmethod
    def rhs(
            t: float,
            y: NDArray,
            StateMap: Type[Enum],
            algebraic_values_dict: Dict[str, Any],
            control_elements: ControlElements,
            system_constants: Dict
            ) -> NDArray:
        """
        Calculates the derivatives for the differential states (dV/dt, dA/dt, dB/dt).
        This is the differential part of the DAE system.
        """
        # Unpack values from the inputs
        F_out = algebraic_values_dict['F_out']
        F_in = control_elements.F_in
        k = system_constants['k']
        CA_in = system_constants['CA_in']

        # Unpack current state values using the StateMap for clarity
        volume = max(1e-6, y[StateMap.V.value])
        molarity_A = y[StateMap.A.value]
        molarity_B = y[StateMap.B.value]
        
        # Calculate reaction rate: r = k * [A] * V
        reaction_rate = molarity_A * volume * k
        
        # Initialize the derivative array
        dy = np.zeros_like(y)
        
        # Calculate the derivatives
        dV_dt = F_in - F_out
        dy[StateMap.V.value] = dV_dt
        dy[StateMap.A.value] = (1/volume) * (-reaction_rate + F_in * CA_in - F_out * molarity_A - molarity_A * dV_dt)
        dy[StateMap.B.value] = (1/volume) * (2*reaction_rate - F_out * molarity_B - molarity_B * dV_dt)

        return dy

# 3. Define Sensors, Controllers, and Trajectories
# =================================================

class FlowOutSensor(Sensor):
    """Measures the outlet flow rate from the algebraic states."""
    def measure(self, measurable_quantities):
        return measurable_quantities.algebraic_states.F_out

class FlowInSensor(Sensor):
    """Measures the inlet flow rate from the control elements."""
    def measure(self, measurable_quantities):
        return measurable_quantities.control_elements.F_in
    
class BConcentrationSensor(Sensor):
    """Measures the concentration of B from the differential states."""
    def measure(self, measurable_quantities):
        return measurable_quantities.states.B
    
class VolumeSensor(Sensor):
    """Measures the reactor volume from the differential states."""
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
    """Provides a constant setpoint value over time."""
    def __init__(self, value):
        self.value = value
    
    def __call__(self, t):
        return self.value
    
    def change(self, new_value):
        """Allows for changing the setpoint during a simulation."""
        self.value = new_value

