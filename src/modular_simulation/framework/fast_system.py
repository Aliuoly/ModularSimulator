from abc import abstractmethod
from numpy.typing import NDArray
from typing import Callable, ClassVar, Dict, Any
from scipy.integrate import solve_ivp #type: ignore
from numba.typed.typeddict import Dict as NDict
from numba import types, njit
from modular_simulation.framework.system import System

import logging
logger = logging.getLogger(__name__)

class FastSystem(System):
    """
    An abstract base class for performance-optimized systems.

    Inherit from this class when simulation speed is critical. It requires implementing
    a "fast" contract where the core dynamics operate exclusively on NumPy arrays,
    making them compatible with JIT compilers like Numba.

    This class overrides the standard simulation loop to call the `_fast` methods.
    """

    # ``auto_njit`` allows subclasses to opt out of the automatic compilation
    # behaviour introduced by ``FastSystem``. A subclass can set this to
    # ``False`` when hand-crafted ``*_fast`` implementations are desired.
    auto_njit: ClassVar[bool] = True
    njit_options: ClassVar[Dict[str, Any]] = {"cache": True}

    def __init_subclass__(cls, auto_njit: bool | None = None, **kwargs: Any) -> None:
        """Automatically compile readable implementations with ``numba``.

        ``FastSystem`` subclasses can inherit the readable versions of
        ``calculate_algebraic_values`` and ``rhs`` from an existing ``System``
        subclass (e.g. ``class FooFastSystem(FastSystem, FooSystem)``).  When
        ``auto_njit`` is enabled this hook compiles the inherited implementations
        with :func:`numba.njit` and wires the compiled functions into the fast
        execution path so users no longer have to duplicate logic manually.

        Subclasses can disable the behaviour globally by setting
        ``auto_njit = False`` or per-subclass via ``class FooFastSystem(...,
        auto_njit=False)``.  If the automatic compilation fails (because the
        readable implementation uses unsupported Python/NumPy features) a
        descriptive ``TypeError`` is raised instructing the user to provide an
        explicit ``*_fast`` override.
        """

        super().__init_subclass__(**kwargs)

        if cls is FastSystem:
            # The base class itself should not attempt to build the wrappers.
            return

        auto = cls.auto_njit if auto_njit is None else auto_njit
        if not auto:
            return

        if "calculate_algebraic_values_fast" not in cls.__dict__:
            readable = getattr(cls, "calculate_algebraic_values", None)
            if readable is not None and readable is not System.calculate_algebraic_values:
                cls.calculate_algebraic_values_fast = staticmethod(  # type: ignore[assignment]
                    cls._build_algebraic_wrapper(readable)
                )

        if "rhs_fast" not in cls.__dict__:
            readable_rhs = getattr(cls, "rhs", None)
            if readable_rhs is not None and readable_rhs is not System.rhs:
                cls.rhs_fast = staticmethod(  # type: ignore[assignment]
                    cls._build_rhs_wrapper(readable_rhs)
                )

    @classmethod
    def _build_algebraic_wrapper(cls, readable: Callable) -> Callable:
        compiled = njit(**cls.njit_options)(readable)

        def wrapper(
            y: NDArray,
            u: NDArray,
            k: NDArray,
            y_map: NDict,
            u_map: NDict,
            k_map: NDict,
            algebraic_map: NDict,
            algebraic_size: int,
        ) -> NDArray:
            try:
                return compiled(y, u, k, y_map, u_map, k_map, algebraic_map)
            except TypingError as exc:  # pragma: no cover - executed only on failure
                raise TypeError(
                    "Automatic numba compilation of 'calculate_algebraic_values' "
                    "failed. Define 'calculate_algebraic_values_fast' manually "
                    "or disable auto_njit for this FastSystem subclass."
                ) from exc

        return wrapper

    @classmethod
    def _build_rhs_wrapper(cls, readable: Callable) -> Callable:
        compiled = njit(**cls.njit_options)(readable)

        def wrapper(
            t: float,
            y: NDArray,
            u: NDArray,
            k: NDArray,
            algebraic: NDArray,
            u_map: NDict,
            y_map: NDict,
            k_map: NDict,
            algebraic_map: NDict,
        ) -> NDArray:
            try:
                return compiled(t, y, u, k, algebraic, y_map, u_map, k_map, algebraic_map)
            except TypingError as exc:  # pragma: no cover - executed only on failure
                raise TypeError(
                    "Automatic numba compilation of 'rhs' failed. Define 'rhs_fast' "
                    "manually or disable auto_njit for this FastSystem subclass."
                ) from exc

        return wrapper

    def _construct_params(self) -> None:
        """
        overwrites System's method of the same name to support numba njit decoration
        """
        # convert the Enum's to typed dictionaries for numba
        
        y_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = self.measurable_quantities.states._index_dict
        for member in index_map:
            y_map[member] = index_map[member]

        u_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = self.measurable_quantities.control_elements._index_dict
        for member in index_map:
            u_map[member] = index_map[member]
        
        k_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = self.measurable_quantities.constants._index_dict
        for member in index_map:
            k_map[member] = index_map[member]

        algebraic_map = NDict.empty(key_type = types.unicode_type, value_type = types.slice2_type)
        index_map = self.measurable_quantities.algebraic_states._index_dict
        for member in index_map:
            algebraic_map[member] = index_map[member]

        algebraic_size = self.measurable_quantities.algebraic_states.get_total_size()

        self._params = {
            'y_map': y_map,
            'u_map': u_map,
            'k_map': k_map,
            'algebraic_map': algebraic_map,
            'algebraic_size': algebraic_size,
            'k': self.measurable_quantities.constants.to_array(),
            'algebraic_values_function': self.calculate_algebraic_values_fast,
            'rhs_function': self.rhs_fast,
        }
        
    @staticmethod
    @abstractmethod
    def calculate_algebraic_values_fast(
            y: NDArray, 
            u: NDArray,
            k: NDArray,
            y_map: NDict, 
            u_map: NDict,
            k_map: NDict,
            algebraic_map: NDict,
            algebraic_size: int,
            ) -> NDArray:
        pass

    @staticmethod
    @abstractmethod
    def rhs_fast(
            t: float,
            y: NDArray,
            u: NDArray,
            k: NDArray,
            algebraic: NDArray,
            u_map: NDict,
            y_map: NDict,
            k_map: NDict,
            algebraic_map: NDict,
            ) -> NDArray:
        pass
    
    @staticmethod
    def _rhs_wrapper(
            t: float, 
            y: NDArray, 
            u: NDArray,
            k: NDArray,
            y_map: NDict,
            u_map: NDict,
            k_map: NDict,
            algebraic_map: NDict,
            algebraic_size: int,
            algebraic_values_function: Callable,
            rhs_function: Callable,
            ) -> NDArray:
        """
        A concrete wrapper called by the solver. It recalculates algebraic states
        before calling the user-defined `rhs` for the derivatives. This ensures
        correctness even when the solver rejects and retries steps.
        """
        algebraic_array = algebraic_values_function(
            y = y,
            u = u,
            k = k,
            y_map = y_map,
            u_map = u_map,
            k_map = k_map,
            algebraic_map = algebraic_map,
            algebraic_size = algebraic_size
        )
        return rhs_function(
            t,
            y = y,
            u = u,
            k = k,
            algebraic = algebraic_array, 
            y_map = y_map,
            u_map = u_map,
            k_map = k_map,
            algebraic_map = algebraic_map
            )
    def step(self, nsteps: int = 1) -> None:
        """
        The main public method to advance the simulation by one time step.

        Performs one integration step by calling the ODE solver.

        It configures and runs `solve_ivp`, then updates the system's algebraic
        states with the final, successful result.
        """
        global StaticMapEnum

        y_map = self._params['y_map']
        u_map = self._params['u_map']
        k_map = self._params['k_map']
        algebraic_map = self._params['algebraic_map']
        algebraic_size = self._params['algebraic_size']
        k = self._params['k']
        algebraic_values_function = self.calculate_algebraic_values_fast
        rhs_function = self.rhs_fast
        
        for _ in range(nsteps):
            y0, u0 = self._pre_integration_step()
            final_y = y0
            if self.measurable_quantities.states:
                result = solve_ivp(
                    fun = self._rhs_wrapper,
                    t_span = (self._t, self._t + self.dt),
                    y0 = y0,
                    args = (u0, k, y_map, u_map, k_map, algebraic_map, algebraic_size, algebraic_values_function, rhs_function),
                    **self.solver_options
                )
                final_y = result.y[:, -1]
                self.measurable_quantities.states.update_from_array(final_y)

            # After the final SUCCESSFUL step, update the actual algebraic_states object.
            if self.measurable_quantities.algebraic_states:
                final_algebraic_values = algebraic_values_function(
                    final_y,u0, k, y_map, u_map, k_map, algebraic_map, algebraic_size
                )
                self.measurable_quantities.algebraic_states.update_from_array(final_algebraic_values)
            
            self._t += self.dt
            self._update_history()
        return 

    def rhs(**kwargs):
        raise NotImplementedError
    
    def calculate_algebraic_values(**kwargs):
        raise NotImplementedError
    


