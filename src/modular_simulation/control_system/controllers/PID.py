from pydantic import Field, PrivateAttr
import numpy as np
from modular_simulation.control_system.controller import Controller
from numpy.typing import NDArray
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

    # additional PID only private attributes
    _last_t: float = PrivateAttr(default=0.0)
    _last_error: float|NDArray = PrivateAttr(default=0.0)
    _integral: float|NDArray = PrivateAttr(default=0.0)
    _filtered_derivative: float|NDArray = PrivateAttr(default = 0.0)

    def _control_algorithm(
            self,
            t: float,
            cv: float | NDArray[np.float64],
            sp: float | NDArray[np.float64],
            ) -> float | NDArray[np.float64]:
        """
        PID control algorithm for SISO systems. As such, only handles scalar cv and sp."""
        dt = t - self._last_t
        self._last_t = t
        
        error = sp - cv
        if self.inverted:
            error = -error
        self._integral += error * dt
        # first order approximation of the timec onstant -> filter factor
        # valid enough so whatever. 
        alpha = dt / (dt + self.derivative_filter_tc) 
        self._filtered_derivative = alpha * (error - self._last_error) + (1-alpha) * self._filtered_derivative
        
        # PID control law
        p_term = self.Kp * error
        i_term = self.Kp / self.Ti * self._integral
        d_term = self.Kp * self.Td / dt * self._filtered_derivative
        output = p_term + i_term + d_term
        # account for initial 'zero integral' output
        overflow, underflow = output + self._u0 - self.mv_range[1], output + self._u0 - self.mv_range[0]
        saturated = "No"
        if overflow > 0 and self.Ti != np.inf:
            # we are overflowing the range, reduce integral and output to match upper range
            output -= overflow
            # out = p_term + d_term + i_term
            # limited_out = p_term + d_term + limited_i_term
            # out - limited_out = overflow = i_term - limited_i_term
            # limited_i_term = i_term - overflow
            # Kp/Ti*limited_integral = Kp/Ti*intergral - overflow
            # limited_integral = integral - overflow * Ti/Kp
            self._integral += -overflow * self.Ti / self.Kp # since overflow > 0, this decreases integral.
            saturated = "Overflow"
        if underflow < 0 and self.Ti != np.inf:
            # we are underflowing the range, increase integral and output to match lower range
            output -= underflow
            self._integral += -underflow * self.Ti / self.Kp #since underflow < 0, this increases integral. 
            saturated = "Underflow"
        # check if saturated - if so, limit the integral term
        logger.debug(
            # scientific notation takes up 4 spaces by itself, and due to sign of the number another 1 space is possible
            # and from the decimal point another space is taken. thus, need at least 6 + decimal place many spaces
            # so leave 5 spaces free. e.g., %6.1e is ok, since max space = 6, use 1 for decimal, 4 for scientific notation, 1 for sign
            "%-12.12s PID | sat=%-10.10s t=%8.1f cv=%8.2f sp=%8.2f err=%10.2e P=%10.2e I=%10.2e D=%10.2e out=%8.2f",
            self.cv_tag, saturated, t, cv, sp, error, p_term, self._integral, self._filtered_derivative, output + self._u0,
        )
        # Ensure output is non-negative (e.g., flow rate can't be negative)
        self._last_error = error
        return output