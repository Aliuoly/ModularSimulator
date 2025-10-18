from typing import Callable, Dict, Any, Type, Protocol
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from pydantic import Field, PrivateAttr
from astropy.units import Quantity
from modular_simulation.control_system.controller import Controller
from modular_simulation.usables import Calculation
from modular_simulation.usables.tag_info import TagInfo
from modular_simulation.validation.exceptions import ControllerConfigurationError
import logging
logger = logging.getLogger(__name__)


class CalculationModelPath:
    """
    Routing between an IMC's internal model and a Calculation's method. 
    Arguments:
        1. calculation_name: a Calculation Subclass OR a string corresponding to the 
                custom name assigned to the calculation at definition time. 
        2. method_name: the method of the calculation to be used as the IMC's internal model. 
    """
    def __init__(self, calculation_name: Type[Calculation] | str, method_name: str):
        if isinstance(str, calculation_name):
            self.calculation_name = calculation_name
        else:
            self.calculation_name = calculation_name.__name__
        self.method_name = method_name

class IMCModel(Protocol):
    def __call__(self, input: Quantity) -> float | NDArray:
        ...



class InternalModelController(Controller):
    
    model: Callable[[float], float] | CalculationModelPath = Field(
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
    solver_options: Dict[str, Any] = Field(
        default = {"tol": 1e-3, "max_iter": 25},
        description="optional arguments for the minimize_scalar solver from scipy."
    )

    _filtered_sp: float|None = PrivateAttr(default = None)
    _mv_range_in_model_units: tuple[float, float] = PrivateAttr()
    _mv_converter_from_model_units_to_system_units: Callable = PrivateAttr()

    def _initialize(
            self, 
            tag_infos: list[TagInfo],
            usable_quantities, # kept for IMC intiialization. I hate it. 
            control_elements, 
            is_final_control_element = True):
        """override the controller _initialize method to do some additional work in setting model"""
        # first do the normal initialize though
        super()._initialize(tag_infos, usable_quantities, control_elements, is_final_control_element)
        # and now set the model
        if isinstance(self.model, CalculationModelPath):
            # look through the available calculations and grab the right one
            # also grab the unit info so we can pass in the model input
            # with the units the calculation expects. 
            for calculation in usable_quantities.calculations:
                if calculation.calculation_name == self.model.calculation_name:
                    self._internal_model = getattr(calculation, self.model.method_name)
                    self._mv_range_in_model_units = (
                        self.mv_range[0].to(calculation._input_tag_info_dict[self.mv_tag].unit).value,
                        self.mv_range[1].to(calculation._input_tag_info_dict[self.mv_tag].unit).value
                    )
                    self._mv_converter_from_model_units_to_system_units = \
                        calculation._input_tag_info_dict[self.mv_tag].unit.get_converter(
                            self._mv_system_unit
                        )
                    return 
            # if we didn't return, raise error
            raise ControllerConfigurationError(
                f"Could not find the model for '{self.cv_tag}' controller. "
            )
        else:
            self._internal_model = self.model


    # ------------------------------------------------------------------------
    def _control_algorithm(self,
        t: float,
        cv: float,
        sp: float,
        ) -> float:
        """Classic IMC: solve *model(u, meas) = y_set*  for u each cycle.

        *   `meas` supplies disturbances and the current plant state.
        *   `setpoint` is optionally filtered (first‑order).
        *   `fsolve` (or Newton, Brent…) finds the root.
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
        output = minimize_scalar(residual, bounds = self._mv_range_in_model_units).x
        logger.debug(
            "%-12.12s IMC | t=%8.1f cv=%8.2f sp=%8.2f out=%8.2f, cv_pred_at_out=%8.2f",
            self.cv_tag, t, cv, sp, output, internal_model(output)
        )
        return self._mv_converter_from_model_units_to_system_units(output)
