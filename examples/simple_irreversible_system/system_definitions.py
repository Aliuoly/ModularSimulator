"""Dynamic model definitions for the simple irreversible reaction example."""
from __future__ import annotations

from typing import Annotated, Mapping

import numpy as np
from astropy.units import Unit
from numpy.typing import NDArray

from modular_simulation.core import DynamicModel, MeasurableMetadata, MeasurableType

SQRT_L_PER_S = Unit("L") ** 0.5 / Unit("s")


class IrreversibleModel(DynamicModel):
    """Dynamic model describing the A -> 2B reaction with volume expansion."""

    V: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("L")),
    ] = 0.0
    A: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("mol/L")),
    ] = 0.0
    B: Annotated[
        float,
        MeasurableMetadata(MeasurableType.DIFFERENTIAL_STATE, Unit("mol/L")),
    ] = 0.0

    F_out: Annotated[
        float,
        MeasurableMetadata(MeasurableType.ALGEBRAIC_STATE, Unit("L/s")),
    ] = 0.0

    F_in: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONTROL_ELEMENT, Unit("L/s")),
    ] = 0.0

    k: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("1/s")),
    ] = 0.0
    Cv: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, SQRT_L_PER_S),
    ] = 0.0
    CA_in: Annotated[
        float,
        MeasurableMetadata(MeasurableType.CONSTANT, Unit("mol/L")),
    ] = 0.0

    @staticmethod
    def calculate_algebraic_values(
        y: NDArray,
        u: NDArray,
        k: NDArray,
        y_map: Mapping[str, slice],
        u_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
        algebraic_size: int,
    ) -> NDArray:
        """Compute the outlet flow from the current reactor volume."""

        result = np.zeros(algebraic_size)
        volume = max(1.0e-6, y[y_map["V"]][0])
        Cv = k[k_map["Cv"]][0]
        result[algebraic_map["F_out"]][0] = Cv * np.sqrt(volume)
        return result

    @staticmethod
    def rhs(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        algebraic: NDArray,
        u_map: Mapping[str, slice],
        y_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
    ) -> NDArray:
        """Evaluate the material balance derivatives."""

        dy = np.zeros_like(y)

        F_out = algebraic[algebraic_map["F_out"]][0]
        F_in = u[u_map["F_in"]][0]
        rate_constant = k[k_map["k"]][0]
        inlet_conc = k[k_map["CA_in"]][0]

        volume = max(1.0e-6, y[y_map["V"]][0])
        conc_a = y[y_map["A"]][0]
        conc_b = y[y_map["B"]][0]

        reaction_rate = conc_a * volume * rate_constant
        dV_dt = F_in - F_out

        dy[y_map["V"]] = dV_dt
        dy[y_map["A"]] = (1.0 / volume) * (
            -reaction_rate + F_in * inlet_conc - F_out * conc_a - conc_a * dV_dt
        )
        dy[y_map["B"]] = (1.0 / volume) * (
            2.0 * reaction_rate - F_out * conc_b - conc_b * dV_dt
        )

        return dy


__all__ = ["IrreversibleModel"]
