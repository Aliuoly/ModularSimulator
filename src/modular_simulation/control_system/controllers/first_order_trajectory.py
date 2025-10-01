from typing import Callable
from pydantic import Field, PrivateAttr
from modular_simulation.control_system.controller import Controller
from modular_simulation.validation.exceptions import ControllerConfigurationError
import logging
logger = logging.getLogger(__name__)

class FirstOrderTrajectoryController(Controller):
    
    closed_loop_time_constant_fraction: float = Field(
        default = 1.0,
        gt = 0.0,
        description = (
            "The closed loop time constant used by the controller is defined as "
            "closed_loop_time_constant_fraction * open_loop_time_constant, where "
            "open_loop_time_constant can be measured or a constant. "
        )
    )
    open_loop_time_constant: float | str = Field(
        ...,
        description = (
            "if a float is provided, this is the constant open_loop_time_constant used by "
            "the controller. If a string is provided, the string is assumed to be a calculation output tag "
            "defined as part of the usable_quantities of the system. "
        )
    )

    _get_open_loop_tc: Callable[[], float] = PrivateAttr()
    _t: float = PrivateAttr(default = 0.)

    # ------------------------------------------------------------------------
    def _initialize(self, usable_quantities, control_elements, is_final_control_element = True):
        # do whatever normal initialization first
        super()._initialize(usable_quantities, control_elements, is_final_control_element)
        # and then resolve the open_loop_time_constant thing
        if isinstance(self.open_loop_time_constant, str):
            for sensor in usable_quantities.sensors:
                if sensor.alias_tag == self.open_loop_time_constant:
                    self._get_open_loop_tc = lambda s = sensor: s._last_value.value
                    return

            for calculation in usable_quantities.calculations:
                for output_tag in calculation._output_tags:
                    if output_tag == self.open_loop_time_constant:
                        self._get_open_loop_tc = lambda c = calculation: c._last_results[output_tag].value
                        return
            # if not found, raise error
            raise ControllerConfigurationError(
                f"The configured open loop time constant tag '{self.open_loop_time_constant}' "
                "was not found in the defined measurements or calculations. "
            )

    def _control_algorithm(self,
        t: float,
        cv: float,
        sp: float,
        ) -> float:
        open_loop_tc = self._get_open_loop_tc()
        closed_loop_tc = open_loop_tc * self.closed_loop_time_constant_fraction
        dt = t - self._t
        if dt < 1e-12:
            return self._last_output.value
        alpha_desired = dt / (dt + closed_loop_tc) # approximation
        alpha = dt / (dt + open_loop_tc)
        desired_next_value = alpha_desired * sp + (1-alpha_desired) * cv
        # now use the open loop tc to see what the output needs to be
        # to reach the desired_next_value in the next dt
        # next_value = desired_next_value = alpha * output + (1-alpha) * cv
        # output = (desired_next_value - (1-alpha) * cv) / alpha

        # e.g. cv = 0, sp = 1, dt = 1, tc = 4, desired tc = 1 -> alpha = 0.2, desired_alpha = 0.5
        # desired_next_value = 0.5, output = (0.5 - 0.8 * 0) / 0.2 = 2.5 -> next value = 0.2*2.5 + 0.8 * 0 = 0.5
        return (desired_next_value - (1-alpha) * cv) / dt