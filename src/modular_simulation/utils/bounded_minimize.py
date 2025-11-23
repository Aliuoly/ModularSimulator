from modular_simulation.utils.typing import StateValue
from typing import Callable


def bounded_minimize(f: Callable[[StateValue], StateValue], a: StateValue, b: StateValue, tol: float = 1e-4, max_iter: int = 100) -> tuple[StateValue, StateValue]:
    """
    Lightweight bounded scalar minimizer using the Golden Section Search.

    Parameters
    ----------
    f : callable
        Function to minimize.
    a, b : float
        Search interval [a, b].
    tol : float, optional
        Tolerance for stopping (default 1e-8).
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    x_min : float
        Estimated location of the minimum.
    f_min : float
        Function value at x_min.
    """
    gr = (5**0.5 - 1) / 2  # golden ratio conjugate (~0.618)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)

    # choose the better endpoint
    if fc < fd:
        return c, fc
    else:
        return d, fd
