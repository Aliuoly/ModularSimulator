"""Utilities for validating system configurations before simulation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable

if TYPE_CHECKING:  # pragma: no cover - only for type checking to avoid circular imports
    from modular_simulation.system import System
    from modular_simulation.quantities import (
        MeasurableQuantities,
        UsableQuantities,
        ControllableQuantities,
    )


class ConfigurationError(Exception):
    """Raised when the supplied system configuration is invalid."""


def validate_and_link(system: "System") -> None:
    """Validate the system configuration and link dependent components.

    This ensures that:
        * every control element has a corresponding controller.
        * each controller's process-variable tag exists and is linked to a sensor.
        * measurements and calculations can be evaluated successfully.
    """
    measurable = system.measurable_quantities
    usable = system.usable_quantities
    controllable = system.controllable_quantities

    _ensure_control_elements_have_controllers(measurable, controllable)
    _link_controllers_to_sensors(controllable, usable)
    _initialize_usable_results(usable, measurable)


def _ensure_control_elements_have_controllers(
    measurable: "MeasurableQuantities",
    controllable: "ControllableQuantities",
) -> None:
    control_element_tags = measurable.control_elements.__class__.model_fields.keys()
    controller_tags = controllable.control_definitions.keys()
    missing_tags = _find_missing_tags(control_element_tags, controller_tags)
    if missing_tags:
        joined = ", ".join(missing_tags)
        raise ConfigurationError(
            f"Control element tag(s) {joined} do not have corresponding controller definitions."
            f" Available controller tags are: {', '.join(controller_tags)}"
        )


def _link_controllers_to_sensors(
    controllable: "ControllableQuantities",
    usable: "UsableQuantities",
) -> None:
    for controller_tag, controller in controllable.control_definitions.items():
        pv_tag = controller.pv_tag
        sensor = _get_sensor_for_tag(usable.measurement_definitions, pv_tag, controller_tag)
        controller.link_pv_sensor(sensor)


def _get_sensor_for_tag(
    sensors: Dict[str, object],
    pv_tag: str,
    controller_tag: str,
):
    try:
        return sensors[pv_tag]
    except KeyError as exc:  # pragma: no cover - trivial guard
        raise ConfigurationError(
            f"Controller '{controller_tag}' references pv_tag '{pv_tag}' which is not defined in the "
            "measurement_definitions. Available pv tags for controllers are: "
            f"{', '.join(sensors.keys())}"
        ) from exc


def _initialize_usable_results(
    usable: "UsableQuantities",
    measurable: "MeasurableQuantities",
) -> None:
    usable._usable_results = {}  # type: ignore[attr-defined]
    for tag, sensor in usable.measurement_definitions.items():
        try:
            usable._usable_results[tag] = sensor.measure(measurable, 0.0)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive error handling
            raise ConfigurationError(
                f"Error processing measurement '{tag}': {exc}. "
                "Verify that all dependencies are correctly defined and ordered."
                
            ) from exc
    for tag, calculation in usable.calculation_definitions.items():
        try:
            usable._usable_results[tag] = calculation.calculate(usable._usable_results)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive error handling
            raise ConfigurationError(
                f"Error processing calculation '{tag}': {exc}. "
                "Verify that all dependencies are correctly defined and available."
            ) from exc


def _find_missing_tags(
    required_tags: Iterable[str],
    available_tags: Iterable[str],
) -> list[str]:
    available = set(available_tags)
    return [tag for tag in required_tags if tag not in available]
