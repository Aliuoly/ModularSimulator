from __future__ import annotations

from enum import Enum
from typing import Union

from pydantic import Field, PrivateAttr, model_validator

from modular_simulation.control_system.controllers.controller import Controller
from modular_simulation.control_system.trajectory import Trajectory
from modular_simulation.usables import TimeValueQualityTriplet

class ControllerMode(Enum):
    auto    = 1
    cascade = 2

class CascadeController(Controller):
    """
    A simple wrapper used to define hiearchical structure of controllers,
    where an "inner loop" can be cascaded to an "outer loop"; the outer loop 
    then sets the setpoint of the inner loop, and the inner loop uses said
    setpoint to perform normal control. 

    outer loop(outer loop sp) -> setpoint for inner loop -> 
        inner loop(inner loop setpoint) -> inner loop control output

    The final output of the update method is then the inner loop control output
    """
    inner_loop: Controller = Field(
        ...,
        description="Inner loop controller that acts directly on the control element.",
    )
    outer_loop: Union[Controller, "CascadeController"] = Field(
        ...,
        description=(
            "Outer loop controller. When in cascade mode its output becomes the setpoint for "
            "the inner loop. When in auto mode, the outer loop tracks its measured value."
        ),
    )
    mode: ControllerMode = Field(
        default=ControllerMode.cascade,
        description=(
            "Controller mode of this cascade-capable controller. "
            "auto = outer_loop is bypassed, inner_loop follows its own trajectory. "
            "cascade = outer_loop dictates the trajectory of the inner_loop."
        ),
    )

    _auto_source: Trajectory | Controller = PrivateAttr()

    @model_validator(mode="before")
    def _inherit_loop_metadata(cls, data: dict) -> dict:
        inner = data.get("inner_loop")
        outer = data.get("outer_loop")
        if inner is not None:
            data.setdefault("mv_tag", inner.mv_tag)
            data.setdefault("mv_range", inner.mv_range)
        if outer is not None:
            data.setdefault("cv_tag", outer.cv_tag)
            data.setdefault("sp_trajectory", outer)
        return data

    @model_validator(mode="after")
    def _configure_structure(self) -> "CascadeController":
        if isinstance(self.inner_loop, CascadeController):
            raise TypeError("CascadeController inner_loop must be a non-cascade Controller instance.")

        if self.mv_tag != self.inner_loop.mv_tag:
            raise ValueError("CascadeController mv_tag must match the inner loop controller mv_tag.")
        if self.cv_tag != self.outer_loop.cv_tag:
            raise ValueError("CascadeController cv_tag must match the outer loop controller cv_tag.")

        object.__setattr__(self, "mv_range", self.inner_loop.mv_range)
        self._auto_source = self.inner_loop.sp_trajectory
        self._apply_mode()
        return self

    def _initialize(self, usable_quantities, control_elements) -> None:  # type: ignore[override]
        self._usables = usable_quantities

        self.outer_loop._initialize(usable_quantities, control_elements)
        self.inner_loop._initialize(usable_quantities, control_elements)

        self._cv_getter = self.outer_loop._cv_getter
        self._mv_setter = self.inner_loop._mv_setter
        self._last_value = self.inner_loop._last_value

    def _apply_mode(self) -> None:
        if self.mode is ControllerMode.cascade:
            if self.inner_loop.sp_trajectory is not self.outer_loop:
                self.inner_loop.sp_trajectory = self.outer_loop
        elif self.mode is ControllerMode.auto:
            if self.inner_loop.sp_trajectory is not self._auto_source:
                self.inner_loop.sp_trajectory = self._auto_source
        else:
            raise ValueError(f"Unsupported controller mode: {self.mode!r}")

    def update(self, t: float) -> TimeValueQualityTriplet:  # type: ignore[override]
        result = self._run(t, apply_mv=True)
        self._last_value = result
        return result

    def __call__(self, t: float) -> TimeValueQualityTriplet:  # type: ignore[override]
        return self._run(t, apply_mv=False)

    def _run(self, t: float, *, apply_mv: bool) -> TimeValueQualityTriplet:
        self._apply_mode()

        if self.mode is ControllerMode.auto:
            self.outer_loop.track_pv(t)

        result = self.inner_loop.update(t) if apply_mv else self.inner_loop.__call__(t)
        if apply_mv:
            self._last_value = result
        return result

    def active_sp_trajectory(self) -> Trajectory:  # type: ignore[override]
        if self.mode is ControllerMode.auto:
            source = self._auto_source
        else:
            source = self.outer_loop

        if isinstance(source, Trajectory):
            return source
        return source.active_sp_trajectory()

    def _control_algorithm(self, cv_value, sp_value, t):  # type: ignore[override]
        raise RuntimeError("CascadeController does not use the base control algorithm pathway.")

