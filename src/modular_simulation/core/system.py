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
from modular_simulation.interfaces import (
    CalculationBase,
    ControllerBase,
    ControllerMode,
    SensorBase,
)
from modular_simulation.interfaces.tag_info import TagInfo
from modular_simulation.validation.exceptions import (
    CalculationConfigurationError,
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
    sensors: list[SensorBase] = Field(
        default_factory=list,
        description="Sensors attached to the model for runtime measurements.",
    )
    calculations: list[CalculationBase] = Field(
        default_factory=list,
        description="Calculations that post-process measured data into usable tags.",
    )
    controllers: list[ControllerBase] = Field(
        default_factory=list,
        description="Controllers acting on measured/calculated tags to drive control elements.",
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
    _components_initialized: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _validate(self) -> "System":
        exception_group: list[Exception] = []
        exception_group.extend(self._validate_duplicate_tags())
        exception_group.extend(self._validate_calculation_inputs())
        exception_group.extend(self._validate_controller_tags())
        exception_group.extend(self._validate_sensors_resolvable())
        exception_group.extend(self._validate_controllers_resolvable())
        if exception_group:
            raise ExceptionGroup(
                "errors encountered during model interface instantiation:",
                exception_group,
            )
        self._initialize_components()
        return self

    def _initialize_components(self) -> None:
        for sensor in self.sensors:
            sensor._initialize(self.dynamic_model)
        tag_infos = self._tag_infos
        for calculation in self.calculations:
            calculation._initialize(tag_infos)
        for controller in self.controllers:
            controller._initialize(
                tag_infos,
                sensors=self.sensors,
                calculations=self.calculations,
                control_elements=self.dynamic_model.control_elements,
            )
        self._components_initialized = True

    def update(self, t: float | None = None) -> None:
        if not self._components_initialized:
            raise RuntimeError(
                "system interface components are not initialized. Ensure the system has been constructed."
            )
        time = self._t if t is None else t
        for sensor in self.sensors:
            sensor.measure(time)
        for calculation in self.calculations:
            calculation.calculate(time)
        for controller in self.controllers:
            controller.update(time)

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

    def _validate_duplicate_tags(self) -> list[SensorConfigurationError]:
        exception_group: list[SensorConfigurationError] = []
        seen_tags: list[str] = []
        duplicate_tags: list[str] = []
        for tag in (tag_info.tag for tag_info in self._tag_infos):
            if tag in seen_tags and ".sp" not in tag:
                duplicate_tags.append(tag)
            else:
                seen_tags.append(tag)
        if duplicate_tags:
            exception_group.append(
                SensorConfigurationError(
                    "The following duplicate tag(s) found: "
                    + ", ".join(sorted(set(duplicate_tags)))
                    + "."
                )
            )
        return exception_group

    def _validate_calculation_inputs(self) -> list[CalculationConfigurationError]:
        exception_group: list[CalculationConfigurationError] = []
        available_tags = {tag_info.tag for tag_info in self._tag_infos}
        for calculation in self.calculations:
            missing_tags = [
                info.tag
                for info in calculation._input_tag_info_dict.values()
                if info.tag not in available_tags
            ]
            if missing_tags:
                exception_group.append(
                    CalculationConfigurationError(
                        "The following input tag(s) required by "
                        f"'{calculation.__class__.__name__}' are not available: "
                        + ", ".join(sorted(set(missing_tags)))
                        + "."
                    )
                )
        return exception_group

    def _validate_controller_tags(self) -> list[ControllerConfigurationError]:
        exception_group: list[ControllerConfigurationError] = []
        available_tags = {tag_info.tag for tag_info in self._tag_infos}
        missing_cv_tags: list[str] = []
        missing_mv_tags: list[str] = []
        for controller in self.controllers:
            if controller.cv_tag not in available_tags:
                missing_cv_tags.append(controller.cv_tag)
            if controller.mv_tag not in available_tags:
                missing_mv_tags.append(controller.mv_tag)
        if missing_cv_tags:
            exception_group.append(
                ControllerConfigurationError(
                    "The following controlled variables are not available as "
                    "either measurements or calculations: "
                    + ", ".join(sorted(set(missing_cv_tags)))
                    + "."
                )
            )
        if missing_mv_tags:
            exception_group.append(
                ControllerConfigurationError(
                    "The following manipulated variables are not available as "
                    "either measurements or calculations: "
                    + ", ".join(sorted(set(missing_mv_tags)))
                    + "."
                )
            )
        return exception_group

    def _validate_sensors_resolvable(self) -> list[SensorConfigurationError]:
        unavailable_measurement_tags: list[str] = []
        available_tags = set(self.dynamic_model.tag_list)
        for sensor in self.sensors:
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
            for controller in self.controllers
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
        self.update()
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
            warn_str = (
                "Tried to change trajectory of '%s' controller but failed - controller must be in AUTO mode, "
                "but is %s."
            )
            warnings.warn(warn_str.format(cv_tag, controller.mode.name))
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
    def _tag_infos(self) -> list[TagInfo]:
        infos: list[TagInfo] = []
        for sensor in self.sensors:
            infos.append(sensor._tag_info)
        for calculation in self.calculations:
            infos.extend(calculation._output_tag_info_dict.values())
        for controller in self.controllers:
            active = controller
            while True:
                infos.append(active._make_sp_tag_info(infos))
                if active.cascade_controller is None:
                    break
                active = active.cascade_controller
        return infos

    @cached_property
    def tag_infos(self) -> list[TagInfo]:
        return self._tag_infos

    @cached_property
    def tag_list(self) -> list[str]:
        return [tag_info.tag for tag_info in self._tag_infos]

    @cached_property
    def controller_dictionary(self) -> dict[str, ControllerBase]:
        return_dict: dict[str, ControllerBase] = {}
        for controller in self.controllers:
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
        for sensor in self.sensors:
            sensors_detail[sensor.alias_tag] = sensor._tag_info.history
        for calculation in self.calculations:
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
        for controller in self.controllers:
            history.update(controller.sp_history)
        return history


__all__ = ["System"]
