from typing import Callable, Any, Protocol, Annotated
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr, BaseModel, BeforeValidator
from astropy.units import Quantity #type: ignore
from .controller_base import ControllerBase
from modular_simulation.interfaces.calculations.calculation_base import CalculationBase
from modular_simulation.interfaces.tag_info import TagInfo
from modular_simulation.validation.exceptions import ControllerConfigurationError
import logging
from modular_simulation.utils import bounded_minimize
logger = logging.getLogger(__name__)

def convert_calculation_to_str(calculation_name: type[CalculationBase] | str) -> str:
    if isinstance(calculation_name, str):
        return calculation_name
    elif issubclass(calculation_name, CalculationBase):
        return calculation_name.__name__
    else:
        raise ValueError("calculation name to a IMC controller must be either str or type[CalculationBase]")
class CalculationModelPath(BaseModel):
    """Descriptor pointing an IMC to a ``CalculationBase`` method to use as its model.

    ``calculation_name`` can be the class/type itself or the custom name used
    when the calculation was registered with the system.  ``method_name`` is
    the attribute on that calculation that implements the forward model.
    """
    calculation_name: Annotated[str, BeforeValidator(convert_calculation_to_str)]
    method_name: str

class IMCModel(Protocol):
    """Protocol describing call signature for IMC internal models."""
    def __call__(self, input: Quantity) -> float | NDArray:
        ...

class InternalModelController(ControllerBase):
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
        description = (
            "A model that takes as input the manipulated variable's value "
            "and returns the predicted controlled variable's value. "
        )
    )
    sp_filter_factor: float = Field(
        default = 1.0,
        gt = 0.0,
        le = 1.0,
        description = "first order filter factor on setpoint changes. "
    )
    solver_options: dict[str, Any] = Field(
        default = {"tol": 1e-3, "max_iter": 25},
        description="optional arguments for the minimize_scalar solver from scipy."
    )

    _filtered_sp: float|None = PrivateAttr(default = None)
    _mv_range_in_model_units: tuple[float, float] = PrivateAttr()
    _mv_converter_from_model_units_to_system_units: Callable = PrivateAttr()

    def _initialize(
        self,
        tag_infos: list[TagInfo],
        sensors,
        calculations,
        control_elements,
        is_final_control_element = True,
    ):
        """override the controller _initialize method to do some additional work in setting model"""
        # first do the normal initialize though
        super()._initialize(
            tag_infos,
            sensors,
            calculations,
            control_elements,
            is_final_control_element,
        )
        # and now set the model
        # look through the available calculations and grab the right one
        # also grab the unit info so we can pass in the model input
        # with the units the calculation expects.
        for calculation in calculations:
            if calculation.calculation_name == self.model.calculation_name:
                self._internal_model = getattr(calculation, self.model.method_name)
                self._mv_converter_from_model_units_to_system_units = \
                    calculation._input_tag_info_dict[self.mv_tag].unit.get_converter(
                        self._mv_system_unit
                    )
                self._mv_range_in_model_units = (
                    self._mv_converter_from_model_units_to_system_units(self.mv_range[0]),
                    self._mv_converter_from_model_units_to_system_units(self.mv_range[1])
                )
                return 
        # if we didn't return, raise error
        raise ControllerConfigurationError(
            f"Could not find the model for '{self.cv_tag}' controller. "
        )


    # ------------------------------------------------------------------------
    def _control_algorithm(self,
        t: float,
        cv: float,
        sp: float,
        ) -> float:
        """Solve for the MV that minimizes the squared prediction error.

        The requested setpoint is optionally low-pass filtered, then SciPy's
        :func:`minimize_scalar` searches the pre-converted MV bounds expressed
        in the internal-model units.  The optimizer works on the squared
        residual ``(model(u) - sp)**2`` and the final MV is converted back to
        system units before being returned.
        """

        if self._filtered_sp is None:
            sp_effective = sp
        else:
            sp_effective = self.sp_filter_factor * sp + (1 - self.sp_filter_factor) * self._filtered_sp
        self._filtered_sp = sp_effective

        # --- 2  build the residual function f(u) = model(u) - y_set ----------
        internal_model = self._internal_model
        def residual(u: float) -> float:
            return (internal_model(u) - sp_effective)**2

        # --- 3  root‑solve starting from last control action -----------------
        output, offset = bounded_minimize(residual, *self._mv_range_in_model_units)
        logger.debug(
            "%-12.12s IMC | t=%8.1f cv=%8.2f sp=%8.2f out=%8.2f, cv_pred_at_out=%8.2f, offset=%8.2f",
            self.cv_tag, t, cv, sp, output, internal_model(output), offset
        )
        return self._mv_converter_from_model_units_to_system_units(output)
