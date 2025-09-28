from typing import List, Mapping
from numpy.typing import NDArray
from pydantic import BaseModel, PrivateAttr, ConfigDict
from modular_simulation.framework.utils import create_system
from modular_simulation.measurables import States, AlgebraicStates, Constants, ControlElements
import numpy as np
from modular_simulation.framework.system import System


class Plant(BaseModel):
    """
    A collection of multiple systems that is then treated as one. 
    """
    systems: List["System"]
    dt: float # overwrites the individual systems' dt with this

    _composite_system: System = PrivateAttr()

    def model_post_init(self, context):
        # create a system that encomposes all provided systems
        states = {}
        algebraic_states = {}
        control_elements = {}
        constants = {}
        sensors = []
        calculations = []
        controllers = []
        rhs_list = []
        alg_list = []
        dt = np.inf

        for system in self.systems:
            measurable = system.measurable_quantities
            states.update(measurable.states.model_dump())
            control_elements.update(measurable.control_elements.model_dump())
            algebraic_states.update(measurable.algebraic_states.model_dump())#TODO validate for overlap between systems - algebraic states should be unique to each system.
            constants.update(measurable.constants.model_dump())

            usable = system.usable_quantities
            sensors.extend(usable.sensors)
            calculations.extend(usable.calculations)

            controllable = system.controllable_quantities
            controllers.extend(controllable.controllers)

            rhs_list.append(system.rhs)
            alg_list.append(system.calculate_algebraic_values)

            dt = min(dt, system.dt)

        class CompositeSystem(System):

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

                dy = np.zeros_like(y)
                for func in rhs_list:
                    dy += func(t, y, u, k, algebraic, u_map, y_map, k_map, algebraic_map)
                return dy
            
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
                alg = np.zeros(algebraic_size)
                for func in alg_list:
                    # each func returns a sparse array. No overlap between iterations 
                    # should be possible due to validations. 
                    alg += func(y, u, k, y_map, u_map, k_map, algebraic_map, algebraic_size)
                return alg

        class AllowExtraS(States):
            model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
        class AllowExtraAS(AlgebraicStates):
            model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
        class AllowExtraCE(ControlElements):
            model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
        class AllowExtraC(Constants):
            model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
        self._composite_system = create_system(
            dt = self.dt,
            system_class = CompositeSystem,
            initial_algebraic = AllowExtraAS(**algebraic_states),
            initial_controls = AllowExtraCE(**control_elements),
            initial_states = AllowExtraS(**states),
            system_constants = AllowExtraC(**constants), 
            sensors = sensors,
            calculations = calculations,
            controllers = controllers,
            use_numba = False, 
        )
            
    def step(self, nsteps:int = 1):
        self._composite_system.step(nsteps)



