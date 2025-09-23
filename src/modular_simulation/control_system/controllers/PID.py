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
        candidate_integral = self._integral + error * dt
        # first order approximation of the timec onstant -> filter factor
        # valid enough so whatever. 
        alpha = dt / (dt + self.derivative_filter_tc) 
        self._filtered_derivative = alpha * error + (1-alpha) * self._last_error
        
        
        proportional_term = self.Kp * error
        integral_term = (self.Kp / self.Ti) * candidate_integral
        derivative_term = (self.Kp * self.Td / dt) * self._filtered_derivative

        unsaturated_output = proportional_term + integral_term + derivative_term
        mv_unsat = unsaturated_output + self._u0
        mv_sat = np.clip(mv_unsat, *self.mv_range)

        tol = 1e-9
        hit_upper_limit = (mv_unsat - mv_unsat) > tol
        hit_lower_limit = (mv_unsat - mv_unsat) < tol

        if hit_upper_limit or hit_lower_limit:
            candidate_integral = self._integral
        
        self._integral = candidate_integral

        logger.debug(
            "%s controller (PID): Time %0.0f, cv %0.1e, sp %0.1e, P term %0.1e, I term %0.1e, D term %0.1e, output %0.1e",
            self.cv_tag,
            t,
            pv.value,
            spval,
            error,
            integral_term,
            self._filtered_derivative,
            mv_sat,
        )
        self._last_error = error
        return TimeValueQualityTriplet(t, mv_sat, ok = True)