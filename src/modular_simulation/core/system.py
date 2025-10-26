"""Runtime orchestrator wiring the dynamic model and operator interface."""
from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping
from functools import cached_property
from operator import attrgetter
from typing import Any, TYPE_CHECKING

from astropy.units import Quantity
from numba import jit, types  # type: ignore
from numba.typed.typeddict import Dict as NDict  # type: ignore
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from scipy.integrate import solve_ivp  # type: ignore
from tqdm import tqdm  # type: ignore

from modular_simulation.core.dynamic_model import DynamicModel
from modular_simulation.interfaces import ModelInterface
from modular_simulation.interfaces.controllers.controller_base import ControllerMode
from modular_simulation.validation.exceptions import (
    ControllerConfigurationError,
    SensorConfigurationError,
)

if TYPE_CHECKING:
    from modular_simulation.interfaces import ControllerBase, Trajectory
    from modular_simulation.interfaces.tag_info import TagData

logger = logging.getLogger(__name__)


class System(BaseModel):
    """Co-ordinates the simulation loop for a :class:`DynamicModel`."""

    dt: Quantity = Field(..., description="Simulation sampling period.")
    dynamic_model: DynamicModel = Field(
        ..., description="Container for differential, algebraic, and control variables."
    )
    model_interface: ModelInterface = Field(
        ..., description="Sensors, calculations, and controllers attached to the model."
    )
    solver_options: dict[str, Any] = Field(
        default_factory=lambda: {"method": "LSODA"},
        description="Arguments forwarded to :func:`scipy.integrate.solve_ivp`.",
    )
    use_numba: bool = Field(
        default=True,
        description="Whether to JIT-compile the model callbacks with Numba for speed.",
    )
    numba_options: dict[str, Any] = Field(
        default_factory=lambda: {"nopython": True, "cache": True},
        description="Options passed to :func:`numba.jit` when compiling callbacks.",
    )
    record_history: bool = Field(
        default=False,
        description="Record time-series history of model attributes for later inspection.",
    )
    show_progress: bool = Field(
        default=True,
        description="Display a progress bar when stepping the simulation for multiple intervals.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _history: dict[str, list[ArrayLike]] = PrivateAttr(default_factory=dict)
    _history_slots: list[tuple[Callable[[], Any], list]] = PrivateAttr(default_factory=list)
    _params: dict[str, Any] = PrivateAttr()
    _t: float = PrivateAttr(default=0.0)

    @model_validator(mode="after")
    def _validate(self) -> "System":
        exception_group: list[Exception] = []
        exception_group.extend(self._validate_sensors_resolvable())
        exception_group.extend(self._validate_controllers_resolvable())
        if exception_group:
            raise ExceptionGroup(
                "errors encountered during model interface instantiation:",
                exception_group,
            )
        self.model_interface._initialize(self.dynamic_model)
        return self

    def model_post_init(self, __context: Any) -> None:
        if self.use_numba:
            self._construct_fast_params()
        else:
            self._construct_params()

        if self.record_history:
            for tag in self.dynamic_model.tag_list:
                storage: list[ArrayLike] = []
                self._history[tag] = storage
                getter = attrgetter(tag)
                self._history_slots.append(
                    (lambda g=getter: g(self.dynamic_model), storage)
                )
            self._history["time"] = []

        algebraic_function = self._params["algebraic_values_function"]
        initial_algebraic = algebraic_function(
            y=self.dynamic_model.states.to_array(),
            u=self.dynamic_model.control_elements.to_array(),
            k=self._params["k"],
            y_map=self._params["y_map"],
            u_map=self._params["u_map"],
            k_map=self._params["k_map"],
            algebraic_map=self._params["algebraic_map"],
            algebraic_size=self._params["algebraic_size"],
        )
        self.dynamic_model.algebraic_states.update_from_array(initial_algebraic)

    def _construct_fast_params(self) -> None:
        y_map = NDict.empty(key_type=types.unicode_type, value_type=types.slice2_type)
        for member, slice_ in self.dynamic_model.states._index_map.items():
            y_map[member] = slice_

        u_map = NDict.empty(key_type=types.unicode_type, value_type=types.slice2_type)
        for member, slice_ in self.dynamic_model.control_elements._index_map.items():
            u_map[member] = slice_

        k_map = NDict.empty(key_type=types.unicode_type, value_type=types.slice2_type)
        for member, slice_ in self.dynamic_model.constants._index_map.items():
            k_map[member] = slice_

        algebraic_map = NDict.empty(key_type=types.unicode_type, value_type=types.slice2_type)
        for member, slice_ in self.dynamic_model.algebraic_states._index_map.items():
            algebraic_map[member] = slice_

        algebraic_size = self.dynamic_model.algebraic_states._array_size

        self._params = {
            "y_map": y_map,
            "u_map": u_map,
            "k_map": k_map,
            "algebraic_map": algebraic_map,
            "algebraic_size": algebraic_size,
            "k": self.dynamic_model.constants.to_array(),
            "algebraic_values_function": jit(**self.numba_options)(
                self.dynamic_model.calculate_algebraic_values
            ),
            "rhs_function": jit(**self.numba_options)(self.dynamic_model.rhs),
        }

    def _construct_params(self) -> None:
        self._params = {
            "y_map": self.dynamic_model.states._index_map,
            "u_map": self.dynamic_model.control_elements._index_map,
            "k_map": self.dynamic_model.constants._index_map,
            "algebraic_map": self.dynamic_model.algebraic_states._index_map,
            "algebraic_size": self.dynamic_model.algebraic_states._array_size,
            "k": self.dynamic_model.constants.to_array(),
            "algebraic_values_function": self.dynamic_model.calculate_algebraic_values,
            "rhs_function": self.dynamic_model.rhs,
        }

    def _validate_sensors_resolvable(self) -> list[SensorConfigurationError]:
        unavailable_measurement_tags: list[str] = []
        available_tags = set(self.dynamic_model.tag_list)
        for sensor in self.model_interface.sensors:
            if sensor.measurement_tag not in available_tags:
                unavailable_measurement_tags.append(sensor.measurement_tag)
        if not unavailable_measurement_tags:
            return []
        return [
            SensorConfigurationError(
                "The following measurement tag(s) are not defined in the dynamic model: "
                + ", ".join(sorted(set(unavailable_measurement_tags)))
                + "."
            )
        ]

    def _validate_controllers_resolvable(self) -> list[ControllerConfigurationError]:
        exception_group: list[ControllerConfigurationError] = []
        available_ce_tags = set(self.dynamic_model.control_elements.tag_list)
        improper_ce_tags = [
            controller.mv_tag
            for controller in self.model_interface.controllers
            if controller.mv_tag not in available_ce_tags
        ]
        if improper_ce_tags:
            exception_group.append(
                ControllerConfigurationError(
                    "The following controlled variables are not defined as system control elements: "
                    + ", ".join(sorted(set(improper_ce_tags)))
                    + "."
                )
            )
        return exception_group

    def _update_history(self) -> None:
        if not self.record_history:
            return
        for getter, storage in self._history_slots:
            storage.append(getter())
        self._history["time"].append(self._t)

    def _pre_integration_step(self) -> tuple[NDArray, NDArray]:
        self.model_interface.update(self._t)
        return (
            self.dynamic_model.states.to_array(),
            self.dynamic_model.control_elements.to_array(),
        )

    @staticmethod
    def _rhs_wrapper(
        t: float,
        y: NDArray,
        u: NDArray,
        k: NDArray,
        y_map: Mapping[str, slice],
        u_map: Mapping[str, slice],
        k_map: Mapping[str, slice],
        algebraic_map: Mapping[str, slice],
        algebraic_values_function: Callable[..., NDArray],
        rhs_function: Callable[..., NDArray],
        algebraic_size: int,
    ) -> NDArray:
        algebraic_array = algebraic_values_function(
            y=y,
            u=u,
            k=k,
            y_map=y_map,
            u_map=u_map,
            k_map=k_map,
            algebraic_map=algebraic_map,
            algebraic_size=algebraic_size,
        )
        return rhs_function(
            t,
            y=y,
            u=u,
            k=k,
            algebraic=algebraic_array,
            y_map=y_map,
            u_map=u_map,
            k_map=k_map,
            algebraic_map=algebraic_map,
        )

    def _single_step(self) -> None:
        y_map = self._params["y_map"]
        u_map = self._params["u_map"]
        k_map = self._params["k_map"]
        algebraic_map = self._params["algebraic_map"]
        k = self._params["k"]
        algebraic_values_function = self._params["algebraic_values_function"]
        rhs_function = self._params["rhs_function"]
        algebraic_size = self._params["algebraic_size"]

        y0, u0 = self._pre_integration_step()
        final_y = y0
        if self.dynamic_model.states:
            result = solve_ivp(
                fun=self._rhs_wrapper,
                t_span=(self._t, self._t + self.dt.value),
                y0=y0,
                args=(
                    u0,
                    k,
                    y_map,
                    u_map,
                    k_map,
                    algebraic_map,
                    algebraic_values_function,
                    rhs_function,
                    algebraic_size,
                ),
                **self.solver_options,
            )
            final_y = result.y[:, -1]
            self.dynamic_model.states.update_from_array(final_y)

        if self.dynamic_model.algebraic_states:
            final_algebraic = algebraic_values_function(
                y=final_y,
                u=u0,
                k=k,
                y_map=y_map,
                u_map=u_map,
                k_map=k_map,
                algebraic_map=algebraic_map,
                algebraic_size=algebraic_size,
            )
            self.dynamic_model.algebraic_states.update_from_array(final_algebraic)

        self._t += self.dt.value
        self._update_history()

    def step(self, duration: Quantity | None = None) -> None:
        if duration is None:
            nsteps = 1
        else:
            nsteps = round(duration.to(self.dt.unit).value / self.dt.value)
        show_progress = nsteps > 1 and self.show_progress and logger.level == logging.NOTSET
        progress = tqdm(total=nsteps) if show_progress else None
        for _ in range(int(nsteps)):
            self._single_step()
            if progress is not None:
                progress.update(1)
        if progress is not None:
            progress.close()

    def get_state(self) -> dict:
        state: dict[str, Any] = {
            "dynamic_model": self.dynamic_model.model_dump(serialize_as_any=True),
            "sensors": [sensor.get_state() for sensor in self.model_interface.sensors],
            "calculations": [
                calculation.get_state() for calculation in self.model_interface.calculations
            ],
            "controllers": [
                controller.get_state() for controller in self.model_interface.controllers
            ],
        }
        return state

    def set_state(self, state: dict) -> None:
        self.dynamic_model = self.dynamic_model.model_copy(update=state["dynamic_model"])
        for calculation in self.model_interface.calculations:
            state.update(calculation.get_state())
        for controller in self.model_interface.controllers:
            state.update(controller.get_state())

    def extend_controller_trajectory(
        self, cv_tag: str, value: float | None = None
    ) -> "Trajectory":
        if cv_tag not in self.controller_dictionary:
            raise ValueError(
                f"Specified cv_tag '{cv_tag}' to extend the trajectory for does not correspond "
                f"to any defined controllers. Available controller cv_tags are {self.cv_tag_list}."
            )
        controller = self.controller_dictionary[cv_tag]
        if controller.mode != ControllerMode.AUTO:
            warnings.warn(
                "Tried to change trajectory of '%s' controller but failed - controller must be in AUTO mode, "
                "but is %s.",
                (cv_tag, controller.mode.name),
                stacklevel=2,
            )
        active_trajectory = controller.sp_trajectory
        if value is None:
            value = active_trajectory(self._t)

        old_value = active_trajectory(self._t)
        active_trajectory.set_now(self._t, value)
        new_value = active_trajectory(self._t)
        logger.info(
            "Setpoint trajectory for '%(tag)s' controller changed at time %(time)0.0f from %(old)0.1e to %(new)0.1e",
            {"tag": cv_tag, "time": self._t, "old": old_value, "new": new_value},
        )
        return active_trajectory

    def set_controller_mode(self, cv_tag: str, mode: ControllerMode | str) -> ControllerMode:
        if cv_tag not in self.controller_dictionary:
            raise ValueError(
                f"Specified cv_tag '{cv_tag}' does not correspond to any defined controllers. "
                f"Available controller cv_tags are {self.cv_tag_list}."
            )
        controller = self.controller_dictionary[cv_tag]
        controller.change_control_mode(mode)
        return controller.mode

    @cached_property
    def controller_dictionary(self) -> dict[str, "ControllerBase"]:
        return_dict: dict[str, "ControllerBase"] = {}
        for controller in self.model_interface.controllers:
            active = controller
            return_dict[active.cv_tag] = active
            while active.cascade_controller is not None:
                active = active.cascade_controller
                return_dict[active.cv_tag] = active
        return return_dict

    @cached_property
    def cv_tag_list(self) -> list[str]:
        return list(self.controller_dictionary.keys())

    @property
    def time(self) -> float:
        return self._t

    @property
    def measured_history(self) -> dict[str, Any]:
        sensors_detail: dict[str, list["TagData"]] = {}
        calculations_detail: dict[str, list["TagData"]] = {}
        history: dict[str, Any] = {
            "sensors": sensors_detail,
            "calculations": calculations_detail,
        }
        for sensor in self.model_interface.sensors:
            sensors_detail[sensor.alias_tag] = sensor._tag_info.history
        for calculation in self.model_interface.calculations:
            for output_tag_name, output_tag_info in calculation._output_tag_info_dict.items():
                calculations_detail[output_tag_name] = output_tag_info.history
        return history

    @property
    def history(self) -> dict[str, list]:
        if not self.record_history:
            return {}
        return self._history

    @property
    def setpoint_history(self) -> dict[str, dict[str, list]]:
        history: dict[str, dict[str, list]] = {}
        for controller in self.model_interface.controllers:
            history.update(controller.sp_history)
        return history


__all__ = ["System"]
