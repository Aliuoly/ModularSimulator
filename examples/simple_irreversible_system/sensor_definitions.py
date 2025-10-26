"""Sensor configuration for the simple irreversible reaction example."""
from __future__ import annotations

from astropy.units import Unit

from modular_simulation.interfaces import SampledDelayedSensor


def create_sensors() -> list[SampledDelayedSensor]:
    """Return the collection of sensors attached to the system."""

    return [
        SampledDelayedSensor(measurement_tag="F_out", unit=Unit("L/s")),
        SampledDelayedSensor(
            measurement_tag="F_in",
            unit=Unit("L/s"),
            coefficient_of_variance=0.05,
        ),
        SampledDelayedSensor(
            measurement_tag="B",
            unit=Unit("mol/L"),
            coefficient_of_variance=0.05,
            sampling_period=900.0,
            deadtime=900.0,
        ),
        SampledDelayedSensor(
            measurement_tag="V",
            unit=Unit("L"),
            faulty_probability=0.01,
            faulty_aware=True,
        ),
    ]


__all__ = ["create_sensors"]
