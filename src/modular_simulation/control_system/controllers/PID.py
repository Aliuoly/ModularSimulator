from pydantic import Field, PrivateAttr
from typing import Union, TYPE_CHECKING
from modular_simulation.control_system import Controller
if TYPE_CHECKING:
    from modular_simulation.usables import Sensor
    from modular_simulation.control_system import Trajectory
    from modular_simulation.quantities import UsableResults

class PIDController(Controller):
    
    pv_tag: str = Field(
        ..., 
        description="The measurement tag of the sensor providing the process variable (PV) for this controller."
        )
    sp_trajectory: "Trajectory" = Field(
        ..., 
        description="A Trajectory instance defining the setpoint (SP) over time."
        )
    Kp: float = Field(
        ..., 
        description = "Proportional gain"
        )
    Ti: float = Field(
        ..., 
        description = "Integral time constant"
        )

    _last_t: float = PrivateAttr(default=0.0)
    _last_error: float = PrivateAttr(default=0.0)
    _integral: float = PrivateAttr(default=0.0)
    _pv_sensor: Union["Sensor", None] = PrivateAttr(default=None)

    """A simple Proportional-Integral controller."""

    def _control_algorithm(self, pv, sp, usable_results: "UsableResults", t):
        dt = t - self._last_t
        self._last_t = t
        
        error = sp - pv
        self._integral += error * dt
        
        # PI control law
        correction = self.Kp * error + (self.Kp / self.Ti) * self._integral
        
        # Ensure output is non-negative (e.g., flow rate can't be negative)
        return max(0.0, correction)