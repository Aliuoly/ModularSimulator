from __future__ import annotations
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from typing import Any, Callable, override, TYPE_CHECKING, cast
from collections.abc import Sequence
from functools import cached_property
from modular_simulation.components.control_system.control_element import ControlElement
from modular_simulation.utils.typing import Seconds, StateValue, TimeQuantity
from modular_simulation.components.point import PointRegistry, Point, DataValue

from modular_simulation.utils.wrappers import second
from modular_simulation.measurables.process_model import ProcessModel
from modular_simulation.components.sensors.abstract_sensor import AbstractSensor
from modular_simulation.components.calculations.abstract_calculation import AbstractCalculation
from modular_simulation.components.control_system.controller_mode import ControllerMode
from modular_simulation.components.control_system.abstract_controller import AbstractController
from modular_simulation.validation.exceptions import (
    SensorConfigurationError,
    ControllerConfigurationError,
)
import logging
from tqdm import tqdm

if TYPE_CHECKING:
    from modular_simulation.components.control_system.trajectory import Trajectory

logger = logging.getLogger(__name__)


class System(BaseModel):
    """ """

    dt: Seconds
    """
    How often the system's sensors, calculations, and controllers update. "
    The solver takes adaptive 'internal steps' regardless of this value to update the system's states. "
    """
    process_model: ProcessModel
    """
    A container for all constants, state, control, and algebraic variables that can be measured or recorded. "
    """
    sensors: Sequence[AbstractSensor]
    calculations: Sequence[AbstractCalculation]
    control_elements: Sequence[ControlElement]
    solver_options: dict[str, Any] = Field(  # pyright: ignore[reportExplicitAny]
        default_factory=lambda: {"method": "LSODA"},
    )
    record_history: bool = Field(
        default=False,
        description=(
            "Whether or not to historize the underlying measurable_quantities of the system. "
            "The usable_quantities, which are measured or calculation, are always historized regardless. "
        ),
    )
    show_progress: bool = Field(
        default=True,
        description=(
            "Whether or not to show simulation progress when simulating multiple steps at once."
        ),
    )

    _point_registry: PointRegistry = PrivateAttr(default_factory=PointRegistry)
    _history_updaters: list[Callable[[Seconds], None]] = PrivateAttr(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def model_post_init(self, context: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
        self._validate()

        self.process_model.attach_system(self)

        # Register all process variables in the tag store BEFORE components
        # so that components can find them during their initialization.
        for raw_state_name in self.process_model.state_list:
            metadata = self.process_model.state_metadata_dict[raw_state_name]
            initial_value = getattr(self.process_model, raw_state_name)
            tag_info = Point(
                tag=f"(raw, {metadata.type.name})_{raw_state_name}",
                unit=metadata.unit,
                type="raw",
                description=metadata.description,
                _raw_tag=raw_state_name,
            )
            # Initialize data immediately so components reading it see valid values
            # cast to StateValue to handle potential type discrepancies or numpy scalars
            val = cast(StateValue, initial_value)
            tag_info.data = DataValue(value=val, time=self.process_model.t, ok=True)
            self._point_registry.add(tag_info)

        # importantly, the order must be
        # 1. sensors, 2. calculations, 3. controllers
        # as calculations depend on sensors, and controllers depend on
        # both sensors and calculations.
        for sensor in self.sensors:
            error, successful = self.add_component(sensor)
            if not successful and error is not None:
                raise error

        # calculations may be defined out of order
        # such that calculation 1 depends on result of calculation 2
        # in such case, the wiring of calculation 1 will FAIL in the first pass.
        # We will try to iteratively resolve calculations for up to NUM_CALCULATION passes.

        attempts = 0
        max_attempts = len(self.calculations)
        resolution_done = False if max_attempts > 0 else True
        pending_calculation_indeces = list(range(len(self.calculations)))
        errors: list[Exception] = []
        while not resolution_done and attempts < max_attempts:
            attempts += 1
            errors.clear()
            for calculation_index in pending_calculation_indeces:
                error, successful = self.add_component(self.calculations[calculation_index])
                if error is not None:
                    errors.append(error)
                if successful:
                    pending_calculation_indeces.remove(calculation_index)
                if successful and error is not None:
                    pass
                if not successful and error is None:
                    pass
            if len(pending_calculation_indeces) == 0:
                resolution_done = True
        if len(errors) > 0:
            raise ExceptionGroup(
                f"{len(pending_calculation_indeces)} out of {len(self.calculations)} calculation(s) could not be resolved: ",
                errors,
            )
        for control_element in self.control_elements:
            error, successful = self.add_component(control_element)
            if not successful and error is not None:
                raise error

        if True:  # Always create updaters to keep TagStore in sync

            def make_history_updater(tag_info: Point) -> Callable[[Seconds], None]:
                process_model = self.process_model
                tag = tag_info.raw_tag

                def updater(time: Seconds):
                    value = cast(StateValue, getattr(process_model, tag))
                    tag_info.data = DataValue(
                        value=value, time=time, ok=True
                    )  # the setter of .data updates Point.history as well

                return updater

            for raw_state_name in self.process_model.state_list:
                # Need to find the point we already added
                metadata = self.process_model.state_metadata_dict[raw_state_name]
                tag = f"(raw, {metadata.type.name})_{raw_state_name}"
                tag_info = self._point_registry.get(tag)

                # Should not be None as we just added it
                if tag_info:
                    self._history_updaters.append(make_history_updater(tag_info))

    def _validate(self):
        exception_group: list[Exception] = []
        exception_group.extend(self._validate_sensors_resolvable())
        exception_group.extend(self._validate_controllers_resolvable())

        if len(exception_group) > 0:
            raise ExceptionGroup(
                "errors encountered during usable quantity instantiation:", exception_group
            )

    def _validate_sensors_resolvable(self) -> list[SensorConfigurationError]:
        exception_group: list[SensorConfigurationError] = []
        unavailable_measurement_tags: list[str] = []
        available_measurement_tags = self.process_model.state_list

        for sensor in self.sensors:
            measurement_tag = sensor.measurement_tag
            if measurement_tag not in available_measurement_tags:
                unavailable_measurement_tags.append(measurement_tag)

        if len(unavailable_measurement_tags) > 0:
            exception_group.append(
                SensorConfigurationError(
                    "The following measurement tag(s) are not defined in measurable quantities: "
                    + ", ".join(unavailable_measurement_tags)
                )
            )
        return exception_group

    def _validate_controllers_resolvable(self) -> list[ControllerConfigurationError]:
        exception_group: list[ControllerConfigurationError] = []
        available_ce_tags = self.process_model.controlled_view.state_list
        improper_ce_tags = [
            c.mv_tag for c in self.control_elements if c.mv_tag not in available_ce_tags
        ]

        if len(improper_ce_tags) > 0:
            exception_group.append(
                ControllerConfigurationError(
                    "The following controlled variables are not defined as system control elements:"
                    + ", ".join(improper_ce_tags)
                )
            )
        return exception_group

    def _update_components(self) -> None:
        """
        Handles the control loop logic before integration.

        This method updates all sensors and controllers based on the current state,
        """
        t = self.time
        for s in self.sensors:
            if s.should_update(t):
                _ = s.update(t)
        for c in self.calculations:
            if c.should_update(t):
                _ = c.update(t)
        for c in self.control_elements:
            if c.should_update(t):
                _ = c.update(t)

    def add_component(
        self, component: AbstractSensor | AbstractCalculation | ControlElement
    ) -> tuple[Exception | None, bool]:
        """
        Adds sensors, calculations, and controllers to the system.
        Performs the necessary wiring and initialization steps to bind
        them to the system.
        """
        error = None
        successful = True
        # for sensor and controller, if an error was raised, it is fatal, so
        # the following code assumes if it returned, we are ok.
        # for calculation, that is a different issue.
        if isinstance(component, AbstractSensor):
            self.point_registry.add(component.point)
            # don't care about sensor lol, only way it fails is if
            # state value is bad, in which case scipy solver will fail later.
            exceptions = component.install(self)
            if not exceptions:
                successful = True
            else:
                error = exceptions[0]
                successful = False
        elif isinstance(component, AbstractCalculation):
            # output tag info available before wiring inputs, but
            # certain calculations have some preprocessing during the wiring stage,
            # such as resolving unit issues. Thus, put it after wiring inputs.
            exceptions = component.install(self)
            if not exceptions:
                self.point_registry.add(component.output_point_dict)
                successful = True
            else:
                error = exceptions[0]
                successful = False
        elif isinstance(component, ControlElement):  # pyright: ignore[reportUnnecessaryIsInstance]
            exceptions = component.install(self)
            if not exceptions:
                self.point_registry.add(component.controller_sp_point_dict)
                successful = True
            else:
                error = exceptions[0]
                successful = False
        else:
            raise ValueError(f"Unknown component type: {type(component)}")  # pyright: ignore[reportUnreachable]
        return error, successful

    def step(self, duration: Seconds | TimeQuantity | None = None) -> None:
        """
        The main public method to advance the simulation by one time step.

        Performs one integration step by calling the ODE solver.

        """
        show_progress = False
        duration = second(duration) if duration is not None else self.dt
        nsteps = round(duration / self.dt)
        if nsteps > 1:
            if logger.level == logging.NOTSET:
                # dont show progress bar if we are logging.
                show_progress = self.show_progress
        pbar = None
        if show_progress:
            pbar = tqdm(total=nsteps)
        # update sensors, calculations, and controllers
        # and then update the states with single_step()
        # raise RuntimeError(f"DEBUG: Pre-loop. nsteps={nsteps}")
        for i in range(int(nsteps)):
            self._update_components()
            self.process_model.step(self.dt)
            for updater in self._history_updaters:  # only populated if record history is True
                updater(self.time)
            if pbar:
                _ = pbar.update(1)
        if pbar:
            pbar.close()

    def extend_controller_sp_trajectory(
        self, cv_tag: str, value: StateValue | None = None
    ) -> "Trajectory":
        """
        used to 'extend' the setpoint trajectory of a controller from the current time onwards.
        If the trajectory has already been defined into the future, the trajectory is trimmed
        back to the current time.

        cv_tag is used to specify which controller

        """
        if cv_tag not in self.controller_dictionary:
            raise ValueError(
                f"Specified cv_tag '{cv_tag}' to extend the trajectory for does not correspond"
                + f" to any defined controllers. Available controller cv_tags are {self.cv_tag_list}."
            )

        controller = self.controller_dictionary[cv_tag]
        if controller.mode != ControllerMode.AUTO:
            logging.info(
                f"Tried to change trajectory of '{controller.cv_tag}' controller but failed - "
                + f"controller must be in AUTO mode, but is {controller.mode.name}. "
            )
        active_trajectory = controller.sp_trajectory
        if active_trajectory is None:
            raise RuntimeError("Somehow the controller has no sp_trajectory. Report bug.")
        if value is None:
            value = active_trajectory(self.time)

        old_value = active_trajectory(self.time)
        _ = active_trajectory.set(t=self.time, value=value)
        new_value = active_trajectory(self.time)
        logger.info(
            "Setpoint trajectory for '%(tag)s' controller changed at time %(time)0.0f "
            + "from %(old)0.1e to %(new)0.1e",
            {"tag": cv_tag, "time": self.time, "old": old_value, "new": new_value},
        )
        return active_trajectory

    def extend_control_element_mv_trajectory(
        self, mv_tag: str, value: StateValue | None = None
    ) -> "Trajectory":
        """
        used to 'extend' the manipulated variable trajectory of a controller from the current time onwards.
        If the trajectory has already been defined into the future, the trajectory is trimmed
        back to the current time.

        !! Changes the corresponding controller's mode to MANUAL

        mv_tag is used to specify which controller! different from extend_controller_sp_trajectory!

        """
        if mv_tag not in self.control_element_mv_dictionary:
            raise ValueError(
                f"Specified mv_tag '{mv_tag}' to extend the trajectory for does not correspond"
                + f" to any defined control elements. Available mv_tags are {self.mv_tag_list}."
            )

        control_element = self.control_element_mv_dictionary[mv_tag]
        control_element.change_control_mode(ControllerMode.MANUAL)
        active_trajectory = control_element.mv_trajectory
        if active_trajectory is None:
            raise RuntimeError("Somehow the control element has no mv_trajectory. Report bug.")

        if value is None:
            value = active_trajectory(self.time)
        old_value = active_trajectory(self.time)
        _ = active_trajectory.set(t=self.time, value=value)
        new_value = active_trajectory(self.time)
        logger.info(
            "MV trajectory for '%(tag)s' controller changed at time %(time)0.0f "
            + "from %(old)0.1e to %(new)0.1e",
            {"tag": mv_tag, "time": self.time, "old": old_value, "new": new_value},
        )
        return active_trajectory

    def set_controller_mode(self, cv_tag: str, mode: ControllerMode | str) -> ControllerMode:
        """Change the mode of a controller identified by ``cv_tag``."""
        if cv_tag not in self.controller_dictionary:
            raise ValueError(
                f"Specified cv_tag '{cv_tag}' does not correspond to any defined controllers. "
                + f"Available controller cv_tags are {self.cv_tag_list}."
            )
        controller = self.controller_dictionary[cv_tag]
        controller.change_control_mode(mode)
        return controller.mode

    @property
    def point_registry(self) -> PointRegistry:
        return self._point_registry

    @cached_property
    def controller_dictionary(self) -> dict[str, AbstractController]:
        return_dict: dict[str, AbstractController] = {}
        for control_element in self.control_elements:
            controller = control_element.controller
            while controller is not None:
                return_dict.update({controller.cv_tag: controller})
                controller = controller.cascade_controller
        return return_dict

    @cached_property
    def control_element_mv_dictionary(self) -> dict[str, ControlElement]:
        return_dict: dict[str, ControlElement] = {}
        for control_element in self.control_elements:
            return_dict.update({control_element.mv_tag: control_element})
        return return_dict

    @property
    def time(self) -> Seconds:
        return self.process_model.t

    @cached_property
    def cv_tag_list(self) -> list[str]:
        return list(self.controller_dictionary.keys())

    @cached_property
    def mv_tag_list(self) -> list[str]:
        return list(self.control_element_mv_dictionary.keys())

    @property
    def history(self) -> dict[str, list[DataValue]]:
        """Returns historized measurements and calculations."""
        return self._point_registry.history

    def save(self) -> dict[str, Any]:
        """Persist minimal configuration and component snapshots for reconstruction."""

        return {
            "system": {
                "dt": self.dt,
                "solver_options": self.solver_options,
                "record_history": self.record_history,
                "show_progress": self.show_progress,
            },
            "process_model": self.process_model.save(),
            "sensors": [sensor.save() for sensor in self.sensors],
            "calculations": [calc.save() for calc in self.calculations],
            "control_elements": [ce.save() for ce in self.control_elements],
        }

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "System":
        """Recreate a system and its components from serialized snapshots."""

        system_config = payload["system"]
        process_model = ProcessModel.load(payload["process_model"])
        sensors = [AbstractSensor.load(s) for s in payload.get("sensors", [])]
        calculations = [AbstractCalculation.load(c) for c in payload.get("calculations", [])]
        control_elements = [ControlElement.load(ce) for ce in payload.get("control_elements", [])]

        return cls(
            dt=system_config["dt"],
            process_model=process_model,
            sensors=sensors,
            calculations=calculations,
            control_elements=control_elements,
            solver_options=system_config.get("solver_options", {"method": "LSODA"}),
            record_history=system_config.get("record_history", False),
            show_progress=system_config.get("show_progress", True),
        )
