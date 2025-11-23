from __future__ import annotations
from typing import TYPE_CHECKING, cast, override
from collections.abc import Callable
from pydantic import Field, PrivateAttr
from modular_simulation.usables.control_system.controller_base import ControllerBase
from modular_simulation.usables.tag_info import TagData
from modular_simulation.validation.exceptions import ControllerConfigurationError
from modular_simulation.utils.typing import Seconds, StateValue

if TYPE_CHECKING:
    from modular_simulation.framework.system import System
from astropy.units import Unit, UnitBase
import logging

logger = logging.getLogger(__name__)


class FirstOrderTrajectoryController(ControllerBase):
    """ControllerBase that targets user-specified first-order closed-loop dynamics.

    The controller approximates the process as a first-order lag and computes
    the manipulated variable that will drive the measured CV toward the
    setpoint with a desired closed-loop time constant.  The open-loop time
    constant may be provided directly or resolved from a measurement or
    calculation tag at runtime.
    """

    closed_loop_time_constant_fraction: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "The closed loop time constant used by the controller is defined as "
            "closed_loop_time_constant_fraction * open_loop_time_constant, where "
            "open_loop_time_constant can be measured or a constant. "
        ),
    )
    open_loop_time_constant: Seconds | str = Field(
        ...,
        description=(
            "if a Seconds is provided, this is the constant open_loop_time_constant used by "
            "the controller. If a string is provided, the string is assumed to be a calculation output tag "
            "defined as part of the usable_quantities of the system. "
        ),
    )

    _get_open_loop_tc: Callable[[], Seconds] = PrivateAttr()

    # ------------------------------------------------------------------------
    @override
    def _post_commission(
        self,
        system: System,
        mv_getter: Callable[[], TagData],
        mv_range: tuple[StateValue, StateValue],
        mv_tag: str,
        mv_unit: UnitBase,
    ) -> bool:
        # resolve the open_loop_time_constant thing
        if isinstance(self.open_loop_time_constant, str):
            found_tag_info = system.tag_store.get(self.open_loop_time_constant)
            if found_tag_info is None:
                raise ControllerConfigurationError(
                    f"The configured open loop time constant tag '{self.open_loop_time_constant}' "
                    + "was not found in the defined measurements or calculations. "
                )
            if found_tag_info.unit.is_equivalent("second") is False:
                raise ControllerConfigurationError(
                    f"The configured open loop time constant tag '{self.open_loop_time_constant}' "
                    + "was found to not have units of time. Got '{found_tag_info.unit}' instead. "
                )
            converter = found_tag_info.unit.get_converter(Unit("second"))

            def tc_getter() -> Seconds:
                return cast(float, converter(found_tag_info.data.value))

            self._get_open_loop_tc = tc_getter
            return True
        else:

            def tc_getter() -> Seconds:
                return cast(float, self.open_loop_time_constant)

            self._get_open_loop_tc = tc_getter
            return True

    @override
    def _control_algorithm(
        self,
        t: Seconds,
        cv: StateValue,
        sp: StateValue,
    ) -> tuple[StateValue, bool]:
        """Compute the MV required to hit the desired next CV sample.

        The algorithm derives the desired CV trajectory based on the requested
        closed-loop time constant, then uses the open-loop model to solve for
        the MV that would achieve that point over the sample interval ``dt``.
        """
        open_loop_tc = self._get_open_loop_tc()
        closed_loop_tc = open_loop_tc * self.closed_loop_time_constant_fraction
        dt = t - self.t
        alpha_desired = dt / (dt + closed_loop_tc)  # approximation
        alpha = dt / (dt + open_loop_tc)
        desired_next_value = alpha_desired * sp + (1 - alpha_desired) * cv
        # now use the open loop tc to see what the output needs to be
        # to reach the desired_next_value in the next dt
        # next_value = desired_next_value = alpha * output + (1-alpha) * cv
        # output = (desired_next_value - (1-alpha) * cv) / alpha

        # e.g. cv = 0, sp = 1, dt = 1, tc = 4, desired tc = 1 -> alpha = 0.2, desired_alpha = 0.5
        # desired_next_value = 0.5, output = (0.5 - 0.8 * 0) / 0.2 = 2.5 -> next value = 0.2*2.5 + 0.8 * 0 = 0.5
        output = (desired_next_value - (1 - alpha) * cv) / alpha
        expected_next_value = alpha * output + (1 - alpha) * cv
        logger.debug(
            "%-12.12s FOTC | t=%8.1f cv=%8.2f sp=%8.2f out=%8.2f, cv_pred_at_out=%8.2f",
            self.cv_tag,
            t,
            cv,
            sp,
            output,
            expected_next_value,
        )
        return output, True
