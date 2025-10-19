from __future__ import annotations

import numpy as np
from astropy.units import Unit
from typing import Annotated

from modular_simulation.measurables.base_classes import AlgebraicStates, Constants, ControlElements, States
from modular_simulation.measurables.measurable_quantities import MeasurableQuantities
from modular_simulation.ui import SimulationBuilder
from modular_simulation.usables.sensors.sampled_delayed_sensor import SampledDelayedSensor
from modular_simulation.usables.controllers.PID import PIDController
from modular_simulation.framework.system import System


class SimpleStates(States):
    x: Annotated[float, Unit(1)] = 1.0


class SimpleControls(ControlElements):
    u: Annotated[float, Unit(1)] = 0.0


class SimpleAlgebraic(AlgebraicStates):
    a: Annotated[float, Unit(1)] = 0.0


class SimpleConstants(Constants):
    k: Annotated[float, Unit("1/s")] = 1.0


class SimpleSystem(System):
    @staticmethod
    def calculate_algebraic_values(
        y,
        u,
        k,
        y_map,
        u_map,
        k_map,
        algebraic_map,
        algebraic_size,
    ):
        result = np.zeros(algebraic_size)
        result[algebraic_map["a"]] = y[y_map["x"]]
        return result

    @staticmethod
    def rhs(
        t,
        y,
        u,
        k,
        algebraic,
        y_map,
        u_map,
        k_map,
        algebraic_map,
    ):
        dy = np.zeros_like(y)
        x = y[y_map["x"]][0]
        u_val = u[u_map["u"]][0]
        k_val = k[k_map["k"]][0]
        dy[y_map["x"]] = -k_val * x + u_val
        return dy


def test_builder_adds_components_and_runs():
    measurables = MeasurableQuantities(
        states=SimpleStates(x=1.0),
        control_elements=SimpleControls(u=0.0),
        algebraic_states=SimpleAlgebraic(a=0.0),
        constants=SimpleConstants(k=1.0),
    )

    builder = SimulationBuilder(
        system_class=SimpleSystem,
        measurable_quantities=measurables,
        dt=1.0 * Unit("s"),
        use_numba=False,
    )

    builder.add_sensor(
        SampledDelayedSensor.__name__,
        {
            "measurement_tag": "x",
            "alias_tag": "x_meas",
            "unit": "1",
            "sampling_period": 0.0,
            "deadtime": 0.0,
        },
    )

    builder.add_sensor(
        SampledDelayedSensor.__name__,
        {
            "measurement_tag": "u",
            "alias_tag": "u",
            "unit": "1",
            "sampling_period": 0.0,
            "deadtime": 0.0,
        },
    )

    builder.add_controller(
        PIDController.__name__,
        {
            "mv_tag": "u",
            "cv_tag": "x_meas",
            "mv_range": {
                "lower": {"value": 0.0, "unit": "1"},
                "upper": {"value": 10.0, "unit": "1"},
            },
            "Kp": 1.0,
            "Ti": 5.0,
            "Td": 0.0,
        },
        trajectory={
            "y0": 0.0,
            "unit": "1",
            "segments": [
                {"type": "step", "magnitude": 1.0},
            ],
        },
    )

    builder.set_plot_layout(
        1,
        1,
        [
            {"panel": 0, "tag": "x_meas", "label": "Measurement"},
        ],
    )

    result = builder.run(1.0 * Unit("s"))

    assert result["time"] > 0
    assert "x_meas" in result["outputs"]["sensors"]
    assert result["figure"] is not None
    assert result["figure"].startswith("data:image/png;base64,")


def test_builder_produces_default_plot_without_layout():
    measurables = MeasurableQuantities(
        states=SimpleStates(x=1.0),
        control_elements=SimpleControls(u=0.0),
        algebraic_states=SimpleAlgebraic(a=0.0),
        constants=SimpleConstants(k=1.0),
    )

    builder = SimulationBuilder(
        system_class=SimpleSystem,
        measurable_quantities=measurables,
        dt=1.0 * Unit("s"),
        use_numba=False,
    )

    builder.add_sensor(
        SampledDelayedSensor.__name__,
        {
            "measurement_tag": "x",
            "unit": "1",
            "sampling_period": 0.0,
            "deadtime": 0.0,
        },
    )

    result = builder.run(1.0 * Unit("s"))

    assert result["figure"] is not None
    assert result["figure"].startswith("data:image/png;base64,")
