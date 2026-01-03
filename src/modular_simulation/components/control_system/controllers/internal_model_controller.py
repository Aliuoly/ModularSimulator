from __future__ import annotations
from collections.abc import Callable
from typing import Any, Annotated, TYPE_CHECKING, override, cast
from astropy.units import UnitBase
import numpy as np
from pydantic import Field, PrivateAttr, BaseModel, BeforeValidator
from modular_simulation.components.control_system.abstract_controller import AbstractController
from modular_simulation.components.calculations.abstract_calculation import AbstractCalculation
from modular_simulation.components.point import DataValue
from modular_simulation.validation.exceptions import ControllerConfigurationError
from modular_simulation.utils.typing import Seconds, StateValue
from modular_simulation.utils.bounded_minimize import bounded_minimize
import logging

if TYPE_CHECKING:
    from modular_simulation.framework.system import System
logger = logging.getLogger(__name__)


def convert_calculation_to_str(calculation_name: type[AbstractCalculation] | str) -> str:
    if isinstance(calculation_name, str):
        return calculation_name
    elif issubclass(calculation_name, AbstractCalculation):
        return calculation_name.__name__
    else:
        raise ValueError(
            "calculation name to a IMC controller must be either str or type[CalculationBase]"
        )


class CalculationModelPath(BaseModel):
    """Descriptor pointing an IMC to a ``CalculationBase`` method to use as its model.

    ``calculation_name`` can be the class/type itself or the custom name used
    when the calculation was registered with the system.  ``method_name`` is
    the attribute on that calculation that implements the forward model.
    """

    calculation_name: Annotated[str, BeforeValidator(convert_calculation_to_str)]
    method_name: str


class InternalModelController(AbstractController):
    """Internal Model Control (IMC) implementation for SISO loops.

    ``model`` may be a callable operating directly in system units or a
    :class:`CalculationBase` method referenced via :class:`CalculationModelPath`.
    During initialization the controller resolves the model, builds unit
    converters between the calculation space and the system's manipulated
    variable, and caches MV bounds in those units.  Each update solves a
    scalar optimization problem to minimize the predicted setpoint error while
    honouring MV constraints and optional setpoint filtering.
    """

    model: CalculationModelPath = Field(
        ...,
        description=(
            "A model that takes as input the manipulated variable's value "
            "and returns the predicted controlled variable's value. "
        ),
    )
    sp_filter_tc: Seconds = Field(
        default=0.0, ge=0.0, description="first order filter time constant on the setpoint. "
    )
    solver_options: dict[str, Any] = Field(
        default={"tol": 1e-3, "max_iter": 25},
        description="optional arguments for the minimize_scalar solver from scipy.",
    )

    _filtered_sp: StateValue = PrivateAttr()
    _mv_range_in_model_units: tuple[StateValue, StateValue] = PrivateAttr()
    _mv_converter_from_model_unit_to_controller_unit: Callable[[StateValue], StateValue] = (
        PrivateAttr()
    )
    _internal_model: Callable[[StateValue], StateValue] = PrivateAttr()

    @override
    def post_wire_to_element(
        self,
        system: System,
        mv_getter: Callable[[], DataValue],
        mv_range: tuple[StateValue, StateValue],
        mv_tag: str,
        mv_unit: UnitBase,
    ) -> bool:
        # initialize filtered sp to be current pv
        self._filtered_sp = system.point_registry[self.cv_tag].data.value
        mv_controller_unit = mv_unit
        found_calculation = [
            c for c in system.calculations if c.name == self.model.calculation_name
        ]
        if len(found_calculation) > 1:
            raise ControllerConfigurationError(
                f"'{self.cv_tag}' controller's internal model could not be resolved. "
                + "Multiple calculations have names matching the specified model name. "
                + "Make sure the calculation names are unique. "
            )
        elif len(found_calculation) == 0:
            raise ControllerConfigurationError(
                f"'{self.cv_tag}' controller's internal model could not be resolved. "
                + f"No calculation have name matching the specified model name '{self.model.calculation_name}'. "
                + f"Available calculations are: {', '.join([c.name for c in system.calculations])} "
            )
        calculation = found_calculation[0]

        self._internal_model = cast(
            Callable[[StateValue], StateValue], getattr(calculation, self.model.method_name)
        )

        self._mv_converter_from_model_unit_to_controller_unit = cast(
            UnitBase, calculation.point_metadata_dict[mv_tag].unit
        ).get_converter(mv_controller_unit)

        converter_from_controller_unit_to_model_unit = mv_controller_unit.get_converter(
            calculation.point_metadata_dict[mv_tag].unit,
        )

        self._mv_range_in_model_units = (
            converter_from_controller_unit_to_model_unit(mv_range[0]),
            converter_from_controller_unit_to_model_unit(mv_range[1]),
        )
        return True

    @override
    def _control_algorithm(
        self,
        t: Seconds,
        cv: StateValue,
        sp: StateValue,
    ) -> tuple[StateValue, bool]:
        """Solve for the MV that minimizes the squared prediction error.

        The requested setpoint is optionally low-pass filtered, then SciPy's
        :func:`minimize_scalar` searches the pre-converted MV bounds expressed
        in the internal-model units.  The optimizer works on the squared
        residual ``(model(u) - sp)**2`` and the final MV is converted back to
        system units before being returned.
        """
        dt = t - self.t
        ff = dt / (dt + self.sp_filter_tc)
        sp_effective = ff * sp + (1 - ff) * self._filtered_sp
        self._filtered_sp = sp_effective

        # --- 2  build the residual function f(u) = model(u) - y_set ----------
        internal_model = self._internal_model

        def residual(u: StateValue) -> StateValue:
            return (internal_model(u) - sp_effective) ** 2

        # --- 3  root‑solve starting from last control action -----------------
        output, offset = bounded_minimize(residual, *self._mv_range_in_model_units)
        logger.debug(
            "%-12.12s IMC | t=%8.1f cv=%8.2f sp=%8.2f out=%8.2f, cv_pred_at_out=%8.2f, offset=%8.2f",
            self.cv_tag,
            t,
            cv,
            sp,
            output,
            internal_model(output),
            offset,
        )
        successful = not np.isnan(offset)
        return self._mv_converter_from_model_unit_to_controller_unit(output), successful
