from modular_simulation.control_system.controllers.controller import Controller
from pydantic import Field, BaseModel
from enum import Enum
from functools import cached_property

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
        description = "inner loop controller, whose output is the output of this CascadeController" \
                        "and whose setpoint is the output of the outer loop controller IF in cascade mode." \
    )
    outer_loop: Controller = Field(
        ...,
        description = "outer loop controller, whose output is the setpoint of the inner_loop IF in cascade mode." \
                        "IF not in cascade mode, the setpoint of this loop tracks its pv."
    )
    mode: ControllerMode = Field(
        default = ControllerMode.cascade,
        description = "Controller mode of this cascade-capable controller." \
                        "auto = outer_loop is bypassed, inner_loop is controlled as is using its internal trajectory." \
                        "cascade = outer_loop dictates the trajectory of the inner_loop. outer_loop setpoint follows its internal trajectory."
    )

    @cached_property
    def cv_tag(self):
        return self.inner_loop.cv_tag
    
    @cached_property
    def mv_tag(self):
        if self.inner_loop.mv_tag != self.outer_loop.cv_tag:
            raise RuntimeError(f"Cascade controller '{self.cv_tag}' has an outer loop whose "\
                               "cv_tag ({self.outer_loop.cv_tag}) is not the mv_tag of the inner loop ({self.inner_loop.mv_tag})")
        return self.inner_loop.mv_tag
    

    def _initialize_mv_setter(self, control_elements):
        # overwriting Controller's method to suite cascade controller requirements
        # where multi-layer loops (e.g., A cascaded to B cascaded to C)
        # could have a cv_tag that does not correspond to a defined control_elements of the system.
        if isinstance(self.inner_loop, self.__class__):
            # update the trajectory of the inner loop
            for control_element_name in control_elements.__class__.model_fields:
                if control_element_name == self.inner_loop.mv_tag:
                    self._mv_setter = lambda value : setattr(control_elements, self.mv_tag, value)
            self._mv_setter = lambda : None
        else:
            self.inner_loop._initialize_mv_setter(control_elements)

    def _control_algorithm(self, t: float):
        
        if self.mode is ControllerMode.auto:
            # if auto, make the outer_loop track its pv
            self.outer_loop.track_pv(t)
        if self.mode is ControllerMode.cascade:
            # if cascade, get outer_loop output as the inner_loop_sp
            inner_loop_sp = self.outer_loop.update(t)
            self.inner_loop.update_trajectory(t, inner_loop_sp)
        
        inner_loop_output = self.inner_loop.update(t)

        return inner_loop_output

