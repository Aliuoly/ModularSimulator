from pydantic import Field, PrivateAttr
import numpy as np
from numpy.typing import NDArray
from modular_simulation.control_system import Controller
from modular_simulation.usables import TimeValueQualityTriplet
import logging
logger = logging.getLogger(__name__)

class PIDController(Controller):

    """
    A simple Proportional-Integral-Derivative controller.
    The control output is return according to the time-domain formulation

    u(t) = Kp * (e(t) + 1/Ti * ∫(e*dt) + Td/dt * de(t)/dt)

    an optional filter may be applied to the derivative term to smooth out
    taking the derivative of a noisy signal.
    """
    
    Kp: float = Field(
        ..., 
        gt = 0,
        description = "Proportional gain"
        )
    Ti: float = Field(
        default = np.inf,
        gt = 0,
        description = "Integral time constant"
        )
    Td: float = Field(
        default = 0.0,
        ge = 0,
        description = "Derivative time constant"
    )
    inverted: bool = Field(
        default = False,
        description = "If True, the controller assumes that higher control output -> lower pv."
    )
    derivative_filter_tc: float = Field(
        default = 0.0,
        ge = 0.0,
        description = "Time constant of the derivative filter used to smooth out derivative action"
    )

    _last_t: float = PrivateAttr(default=0.0)
    _last_error: float = PrivateAttr(default=0.0)
    _integral: float = PrivateAttr(default=0.0)
    _filtered_derivative: float = PrivateAttr(default = 0.0)

    def _control_algorithm(
            self,
            pv: TimeValueQualityTriplet,
            sp: TimeValueQualityTriplet | float | NDArray, 
            t: float
            ) -> TimeValueQualityTriplet:
        
        if isinstance(sp, TimeValueQualityTriplet):
            spok = sp.ok
            spval = sp.value
        else:
            spok = True
            spval = sp
        if not pv.ok or not spok:
            # skip if bad quality
            self._last_t = t
            self._last_value.ok = False
            return self._last_value
        
        
        dt = t - self._last_t
        if dt == 0:
            return self._last_value
        self._last_t = t
        
        error = spval - pv.value
        if self.inverted:
            error = -error
        self._integral += error * dt
        # first order approximation of the timec onstant -> filter factor
        # valid enough so whatever. 
        alpha = dt / (dt + self.derivative_filter_tc) 
        self._filtered_derivative = alpha * error + (1-alpha) * self._last_error
        
        # PI control law
        output = self.Kp * error \
                   + (self.Kp / self.Ti) * self._integral\
                   + (self.Kp * self.Td / dt) * self._filtered_derivative
        logger.debug("%s controller (PID): Time %0.0f, cv %0.1e, sp %0.1e, error %0.1e, integral %0.1e, derivative %0.1e, output %0.1e", 
                     self.cv_tag, t, pv.value, spval, error, self._integral, self._filtered_derivative, output)
        # Ensure output is non-negative (e.g., flow rate can't be negative)
        self._last_error = error
        return TimeValueQualityTriplet(t, output, ok = True)