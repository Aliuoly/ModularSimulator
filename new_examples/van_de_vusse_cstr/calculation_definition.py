"""Helper calculations for the Van de Vusse CSTR example."""
from typing import Annotated

from modular_simulation.usables import CalculationBase, TagMetadata, TagType


class HeatDutyCalculation(CalculationBase):
    """Compute jacket heat duty from measured temperatures."""

    heat_duty_tag: Annotated[
        str,
        TagMetadata(
            type=TagType.OUTPUT,
            unit="kJ/hour",
            description="Estimated heat duty transferred to the jacket",
        ),
    ]
    Tk_tag: Annotated[
        str,
        TagMetadata(
            type=TagType.INPUT,
            unit="deg_C",
            description="Measured jacket temperature",
        ),
    ]
    T_tag: Annotated[
        str,
        TagMetadata(
            type=TagType.INPUT,
            unit="deg_C",
            description="Measured reactor temperature",
        ),
    ]

    kw: float
    area: float

    def _calculation_algorithm(self, t: float, inputs_dict: dict[str, float]) -> dict[str, float]:
        Tk = inputs_dict[self.Tk_tag]
        T = inputs_dict[self.T_tag]
        # kw is stored in kJ/(s*K*m^2); convert to kJ/hour for reporting
        heat_duty_kj_per_s = self.kw * self.area * (Tk - T)
        return {self.heat_duty_tag: heat_duty_kj_per_s * 3600.0}

