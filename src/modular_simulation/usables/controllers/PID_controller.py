import logging
from collections import deque
import numpy as np
from pydantic import Field, PrivateAttr
from modular_simulation.usables.controllers.controller_base import ControllerBase
from modular_simulation.utils.typing import Seconds, StateValue
logger = logging.getLogger(__name__)

class PIDController(ControllerBase):
    """
    A simple Proportional-Integral-Derivative controller.
    The control output is return according to the time-domain formulation

    position form : u(t) = Kp * (e(t) + 1/Ti * ∫(e*dt) + Td * de(t)/dt)
    velocity form : du(t)/dt = Kp * (de(t)/dt + 1/Ti * e + Td * d^2e(t)/dt^2)

    which is represented in the discrete form

    position form : u(k) = Kp * (e(k) + 1/Ti * sum(e*dt) + Td/dt * (e(k)-e(k-1)))
    velocity form : u(k) = u(k-1) + Kp * [(e(k)-e(k-1)) + dt/Ti * e + Td/dt * (e(k)-2e(k-1)+e(k-2))]

    an optional filter may be applied to the derivative term to smooth out
    taking the derivative of a noisy signal.
    """
    
    Kp: float = Field(
        ..., 
        gt = 0,
        description = "Proportional gain"
        )
    Ti: Seconds = Field(
        default = float('inf'),
        gt = 0,
        description = "Integral time constant"
        )
    Td: Seconds = Field(
        default = 0.0,
        ge = 0,
        description = "Derivative time constant"
    )
    derivative_filter_tc: Seconds = Field(
        default = 0.0,
        ge = 0.0,
        description = "Time constant of the derivative filter used to smooth out derivative action"
    )
    setpoint_weight: float = Field(
        default = 1.0,
        ge = 0.0,
        le = 1.0,
        description = (
            "Used to implement the 2-DOF PID I-PD form. "
            "The setpoint weight for the proportional and derivative terms. "
            "the error for the P and D term are modified to be "
            "weight * setpoint - pv. "
            "if 1, corresponds to classic PID. if 0, corresponds to I-PD."
        )
    )
    inverted: bool = Field(
        default = False,
        description = "If True, the controller assumes that higher control output -> lower pv."
    )
    velocity_form: bool = Field(
        default = False,
        description=(
            "Whether to use the velocity form of the PID formulation."
        )
    )
    # additional PID only private attributes
    _error_queue: deque[StateValue] = PrivateAttr(default_factory = lambda: deque([0.0, 0.0], maxlen = 2))
    _integral: StateValue = PrivateAttr(default=0.0)
    _filtered_derivative: StateValue = PrivateAttr(default = 0.0)

    def _post_commission(self, system):
        # if positional form, sets the integral such that the
        # control action will stay constant if the error is 0
        if not self.velocity_form:
            # u(k) = Kp * (e(k) + 1/Ti * sum(e*dt) + Td/dt * (e(k)-e(k-1)))
            # error and de 0, with sum(e*dt) = integral, gives
            # u(k) = Kp * 1/Ti * integral
            # integral = u * Ti/Kp
            if not np.isnan(self._control_action.value) and self.Ti != float('inf'):
                self._integral = self._control_action.value * self.Ti / self.Kp

    def _control_algorithm(
            self,
            t: Seconds,
            cv: StateValue,
            sp: StateValue,
            ) -> tuple[StateValue, bool]:
        """
        PID control algorithm for SISO systems. As such, only handles scalar cv and sp.
        """
        successful = True
        dt = t - self.t
        
        error = sp - cv
        last_error, last_last_error = self._error_queue
        PD_error = self.setpoint_weight * sp - cv
        if self.inverted:
            error = -error
            PD_error = -PD_error
        self._integral += error * dt
        # first order approximation of the timec onstant -> filter factor
        # valid enough so whatever. 
        alpha = dt / (dt + self.derivative_filter_tc) 
        self._filtered_derivative = alpha * (PD_error - self._error_queue[1]) + (1-alpha) * self._filtered_derivative

        # PID control law
        if self.velocity_form:
            p_term = self.Kp * (PD_error - last_error)
            i_term = self.Kp / self.Ti * dt * error
            d_term = self.Kp * self.Td / dt * (PD_error - 2*last_error + last_last_error)
            output = p_term + i_term + d_term + self._control_action.value
            logger.debug(
                # scientific notation takes up 4 spaces by itself, and due to sign of the number another 1 space is possible
                # and from the decimal point another space is taken. thus, need at least 6 + decimal place many spaces
                # so leave 5 spaces free. e.g., %6.1e is ok, since max space = 6, use 1 for decimal, 4 for scientific notation, 1 for sign
                "%-12.12s PID | sat=%-10.10s t=%8.1f cv=%8.2f sp=%8.2f err=%10.2e P=%10.2e I=%10.2e D=%10.2e out=%s=%8.2f%s",
                self.cv_tag, False, t, cv, sp, error, p_term, self._integral, self._filtered_derivative, self.mv_tag, output, str(self._mv_tag_info.unit),
            )
        else:
            p_term = self.Kp * PD_error
            i_term = self.Kp / self.Ti * self._integral
            d_term = self.Kp * self.Td / dt * self._filtered_derivative
            output = p_term + i_term + d_term 
            overflow, underflow = output - self.mv_range[1], output - self.mv_range[0]
            saturated = "No"
            if overflow > 0 and self.Ti != float('inf'):
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
            if underflow < 0 and self.Ti != float('inf'):
                # we are underflowing the range, increase integral and output to match lower range
                output -= underflow
                self._integral += -underflow * self.Ti / self.Kp #since underflow < 0, this increases integral. 
                saturated = "Underflow"
            # check if saturated - if so, limit the integral term
            logger.debug(
                # scientific notation takes up 4 spaces by itself, and due to sign of the number another 1 space is possible
                # and from the decimal point another space is taken. thus, need at least 6 + decimal place many spaces
                # so leave 5 spaces free. e.g., %6.1e is ok, since max space = 6, use 1 for decimal, 4 for scientific notation, 1 for sign
                "%-12.12s PID | sat=%-10.10s t=%8.1f cv=%8.2f sp=%8.2f err=%10.2e P=%10.2e I=%10.2e D=%10.2e out=%s=%8.2f%s",
                self.cv_tag, saturated, t, cv, sp, error, p_term, self._integral, self._filtered_derivative, self.mv_tag, output, str(self._mv_tag_info.unit),
            )
        self._error_queue.appendleft(PD_error)
        return output, successful