from __future__ import annotations
import collections
from typing import TYPE_CHECKING, override
from pydantic import Field, PrivateAttr
from modular_simulation.components.sensors.abstract_sensor import AbstractSensor
from modular_simulation.utils.typing import Seconds
from modular_simulation.components.abstract_component import ComponentUpdateResult
from modular_simulation.components.point import DataValue

if TYPE_CHECKING:
    from modular_simulation.framework.system import System


class SampledDelayedSensor(AbstractSensor):
    """
    A sensor with a set sampling frequency, measurement deadtime, and gaussian noise.
    """

    deadtime: Seconds = Field(
        default=0.0,
        ge=0.0,
        description="The measurement deadtime (a.k.a delay).",
    )

    sampling_period: Seconds = Field(
        default=0.0,
        ge=0.0,
        description=(
            "The sampling period of the sensor. "
            "This is how long it takes for new measurements to become available."
        ),
    )
    _first_sample: bool = PrivateAttr(default=True)
    _sample_queue: collections.deque[DataValue] = PrivateAttr(default_factory=collections.deque)
    _last_sample_t: Seconds = PrivateAttr(default=-1.0)

    # ============================================================
    # ===                    LOGIC METHODS                    ====
    # ============================================================

    def _consume_samples(self, target_t: Seconds) -> DataValue:
        """Return the sample corresponding to target_t, discarding older ones."""
        q = self._sample_queue
        # ensure we have something
        if not q:
            return DataValue()

        while len(q) > 1 and q[1].time <= (target_t + 1e-9):
            q.popleft()

        return q[0]

    @override
    def _install(self, system: System) -> list[Exception]:
        exceptions = super()._install(system)
        if not exceptions:
            self._last_sample_t = system.time
        return exceptions

    @override
    def _update(self, t: Seconds) -> ComponentUpdateResult:
        """Update logic with sampling period and deadtime."""
        self._t = t

        # Check if we should sample
        # Add a small epsilon to handle floating point inaccuracies in time iteration
        epsilon = 1e-9
        should_sample = self._first_sample or t >= (
            self._last_sample_t + self.sampling_period - epsilon
        )

        if should_sample:
            # It IS time to sample.
            # Calling super()._measurement_getter directly as super()._update logic is slightly different
            raw_data = self._measurement_getter()

            # --- SampledDelayed Logic ---
            self._sample_queue.append(raw_data)
            self._last_sample_t = t
            self._first_sample = False

        # Always check for emerging samples
        target_t = t - self.deadtime
        measurement = self._consume_samples(target_t)
        # ----------------------------

        # Layer on fault logic (copied from base for now as per previous refactor decision)
        if self.faulty_probability > 0:
            self._is_faulty = self._generator.random() < self.faulty_probability

        if self._is_faulty:
            measurement.ok = False

        self._point.data = measurement
        return ComponentUpdateResult(data_value=measurement, exceptions=[])
