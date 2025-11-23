# pyright: reportExplicitAny=false
# pyright: reportAny=false
from typing import Callable, Any, Protocol
import numpy as np
from numpy.typing import NDArray

class OdeFunc(Protocol):
    def __call__(self, t: float, y: NDArray[np.float64], *args: Any) -> NDArray[np.float64]: ...

class OdeResult:
    # Time points
    t: NDArray[np.float64]
    # Solution array shape: (n_states, n_points)
    y: NDArray[np.float64]
    # Whether the solver finished successfully
    success: bool
    # Solver message
    message: str
    # Status code
    status: int

def solve_ivp(
    fun: OdeFunc,
    t_span: tuple[float, float],
    y0: NDArray[np.float64],
    args: tuple[Any, ...] = ...,
    method: str | Callable[..., Any] | None = ...,
    t_eval: NDArray[np.float64] | None = ...,
    dense_output: bool = ...,
    events: Any = ...,
    vectorized: bool = ...,
    **options: Any,
) -> OdeResult: ...
