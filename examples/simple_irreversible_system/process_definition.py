from astropy.units import Unit
import numpy as np
from typing import Annotated, override
from collections.abc import Mapping
from numpy.typing import NDArray
from modular_simulation.measurables import ProcessModel, StateType as T, StateMetadata as M
from modular_simulation.utils.typing import ArrayIndex


# 1. Define the Data Structures for the System
# ============================================
class IrreversibleProcessModel(ProcessModel):
    # ----Differential states----
    V: Annotated[float, M(type=T.DIFFERENTIAL, unit="L", description="system volume")] = 10.0
    A: Annotated[float, M(type=T.DIFFERENTIAL, unit="mol/L", description="concentration of A")] = (
        1.0
    )
    B: Annotated[float, M(type=T.DIFFERENTIAL, unit="mol/L", description="concentration of B")] = (
        0.0
    )
    # ----Algebraic states----
    F_out: Annotated[float, M(type=T.ALGEBRAIC, unit="L/s", description="outflow flow")] = 1.0
    # ----Controlled states----
    F_in: Annotated[float, M(type=T.CONTROLLED, unit="L/s", description="inlet flow")] = 1.0
    # ----Constants----
    k: Annotated[float, M(type=T.CONSTANT, unit="1/s", description="rate constant")] = 1e-3
    Cv: Annotated[
        float,
        M(
            type=T.CONSTANT,
            unit=Unit("L") ** 0.5 / Unit("s"),
            description="outlet valve constant",
        ),
    ] = 1e-1
    CA_in: Annotated[
        float, M(type=T.CONSTANT, unit="mol/L", description="inlet A concentration")
    ] = 1.0

    @override
    @staticmethod
    def calculate_algebraic_values(
        y: NDArray[np.float64],
        u: NDArray[np.float64],
        k: NDArray[np.float64],
        y_map: Mapping[str, ArrayIndex],
        u_map: Mapping[str, ArrayIndex],
        k_map: Mapping[str, ArrayIndex],
        algebraic_map: Mapping[str, ArrayIndex],
        algebraic_size: int,
    ) -> NDArray[np.float64]:
        """
        Calculates the outlet flow (F_out) based on the current reactor volume.
        This is the algebraic part of the DAE system.
        """
        result = np.zeros(algebraic_size, dtype=np.float64)
        # Ensure volume doesn't go to zero to prevent division errors.
        volume = max(1e-6, y[y_map["V"]])
        Cv = k[k_map["Cv"]]
        # F_out = Cv * sqrt(V)
        result[algebraic_map["F_out"]] = Cv * (volume**0.5)  # type: ignore[arg-type]
        return result

    @override
    @staticmethod
    def differential_rhs(
        t: float,
        y: NDArray[np.float64],
        u: NDArray[np.float64],
        k: NDArray[np.float64],
        algebraic: NDArray[np.float64],
        y_map: Mapping[str, ArrayIndex],
        u_map: Mapping[str, ArrayIndex],
        k_map: Mapping[str, ArrayIndex],
        algebraic_map: Mapping[str, ArrayIndex],
    ) -> NDArray[np.float64]:
        """
        Calculates the derivatives for the differential states (dV/dt, dA/dt, dB/dt).
        This is the differential part of the DAE system.
        """
        # Unpack values from the inputs
        F_out = algebraic[algebraic_map["F_out"]]
        F_in = u[u_map["F_in"]]
        kc = k[k_map["k"]]
        CA_in = k[k_map["CA_in"]]

        # Unpack current state values using the StateMap for clarity
        volume = max(1e-6, y[y_map["V"]])
        molarity_A = y[y_map["A"]]
        molarity_B = y[y_map["B"]]

        # Calculate reaction rate: r = k * [A] * V
        reaction_rate = molarity_A * volume * kc

        # Initialize the derivative array
        dy = np.zeros_like(y)

        # Calculate the derivatives
        dV_dt = F_in - F_out
        dy[y_map["V"]] = dV_dt
        dy[y_map["A"]] = (1 / volume) * (
            -reaction_rate + F_in * CA_in - F_out * molarity_A - molarity_A * dV_dt
        )
        dy[y_map["B"]] = (1 / volume) * (
            2 * reaction_rate - F_out * molarity_B - molarity_B * dV_dt
        )

        return dy
