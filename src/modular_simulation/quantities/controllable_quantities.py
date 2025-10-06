from typing import List, Dict
from modular_simulation.usables import TagData
from modular_simulation.measurables import ControlElements
from modular_simulation.control_system.controller import Controller
from modular_simulation.quantities import UsableQuantities
from modular_simulation.validation.exceptions import ControllerConfigurationError
from pydantic import BaseModel, model_validator, Field, PrivateAttr
from textwrap import dedent
import warnings

class ControllableQuantities(BaseModel):
    """Container for the controllers acting on the system's control elements."""
    controllers: List[Controller] = Field(
        ...,
        description = "List of controllers applied to the system. "
    )
    control_elements: ControlElements = Field(
        ...,
        description = "Object defining and holding the control elements of the system."
    )
    usable_quantities: UsableQuantities = Field(
        ...,
        description = "Object defining the sensors and calculations of the system."
    )

    _control_outputs: Dict[str, TagData] = PrivateAttr(default_factory=dict)

    @model_validator(mode = 'after')
    def check_duplicate_controllers(self):
        # 1. check for controlled variables controlled by multiple controllers (conflicting control)
        
        exception_group = []

        def check_and_append_inplace(tag: str, seen_list: List[str], duplicate_list: List[str]):
            if tag in seen_list:
                duplicate_list.append(tag)
            seen_list.append(tag)

        seen_cv_tags = []
        seen_mv_tags = []
        duplicate_mv_tags = []
        duplicate_cv_tags = []
        for controller in self.controllers:
            check_and_append_inplace(controller.cv_tag, seen_cv_tags, duplicate_cv_tags)
            check_and_append_inplace(controller.mv_tag, seen_mv_tags, duplicate_mv_tags)
            # iterate over cascade controllers for the same stuff
            while controller.cascade_controller is not None:
                controller = controller.cascade_controller
                check_and_append_inplace(controller.cv_tag, seen_cv_tags, duplicate_cv_tags)
                check_and_append_inplace(controller.mv_tag, seen_mv_tags, duplicate_mv_tags)

        if len(duplicate_cv_tags) > 0:
            warnings.warn(
                "The following controlled variables were subject to control by multiple controllers: "
                f"{', '.join(duplicate_cv_tags)}. Verify that this is intentional. "
            )

        if len(duplicate_mv_tags) > 0:
            exception_group.append(
                ControllerConfigurationError(
                    "The following manipulated variables were subject to manipulation by multiple controllers: "
                    f"{', '.join(duplicate_mv_tags)}."
                )
            )

        # 2. check that all cvs and mvs are in the usable quantities definition
        available_usable_tags = self.usable_quantities.tag_list
        improper_cv_tags = []
        improper_mv_tags = []
        for usable_tag in seen_cv_tags:
            if usable_tag not in available_usable_tags:
                improper_cv_tags.append(usable_tag)
        for usable_tag in seen_mv_tags:
            if usable_tag not in available_usable_tags:
                improper_mv_tags.append(usable_tag)
        
        if len(improper_cv_tags) > 0:
            exception_group.append(
                ControllerConfigurationError(
                    "The following controlled variables are not available as either measurements or calculations: "
                    f"{', '.join(improper_cv_tags)}"
                )
            )
                
        if len(improper_mv_tags) > 0:
            exception_group.append(
                ControllerConfigurationError(
                    "The following manipulated variables are not available as either measurements or calculations:"
                    f"{', '.join(improper_mv_tags)}"
                )
            )

        # 3. check that final control element designate controlled variables (i.e., most-inner loops)
        #       are defined as control elements in the system measurables. 
        improper_ce_tags = []
        available_ce_tags = self.control_elements.tag_list
        for tag in [c.mv_tag for c in self.controllers]: # ignoring the cascade controllers, these mvs must be control elements
            if tag not in available_ce_tags:
                improper_ce_tags.append(tag)
        
        if len(improper_ce_tags) > 0:
            exception_group.append(
                ControllerConfigurationError(
                    "The following controlled variables are not defined as system control elements:"
                    f"{', '.join(improper_ce_tags)}."
                )
            )

        # raise error if necessary
        if len(exception_group) > 0:
            msg = dedent("""Additional info for the sub-exceptions:
                Defined control element tags are: {0}.
                Defined measurement and calculation tags are: {1}.
                """
            ).format(
                ", ".join(available_ce_tags),
                ", ".join(available_usable_tags),
            )
            raise ExceptionGroup(msg, exception_group)

        # initialize controllers once validated
        for controller in self.controllers:
            controller._initialize(self.usable_quantities.tag_infos, self.control_elements)
        # unlike the usable quantities, a manual .update is not necessary
        # as it is essentially done in the _initialize_mv_setter step for controllers.

        return self

    def update(self, t: float) -> Dict[str, TagData]:
        """updates the controllers available. Controllers are linked to the instance of ControlElement
        internally, so the results are reflected in the simulation automatically without having
        to return anything here. However, it is still returned for tracking purposes."""
        return {}
        #self._control_outputs.update(
       #     {controller.mv_tag: controller.update(t) for controller in self.controllers}
        #)
        
        #return self._control_outputs
    
