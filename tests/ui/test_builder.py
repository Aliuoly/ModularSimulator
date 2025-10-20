from __future__ import annotations

import numpy as np
import pytest
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
    series = result["outputs"]["sensors"]["x_meas"]
    assert "ok" in series
    assert len(series["time"]) == len(series["value"]) == len(series["ok"])
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


def test_invalidating_system_freezes_controller_trajectory_at_last_value():
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

    controller_cfg = builder.add_controller(
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
                {"type": "hold", "duration": 5.0, "value": 0.0},
                {"type": "hold", "duration": 5.0, "value": 2.0},
            ],
        },
    )

    builder.run(12.0 * Unit("s"))
    system = builder.system
    assert system is not None
    controller = system.controller_dictionary[controller_cfg.cv_tag]
    last_value = controller.sp_trajectory.current_value(system.time)

    builder.add_sensor(
        SampledDelayedSensor.__name__,
        {
            "measurement_tag": "x",
            "alias_tag": "x_meas_secondary",
            "unit": "1",
            "sampling_period": 0.0,
            "deadtime": 0.0,
        },
    )

    frozen_spec = builder.controller_configs[controller_cfg.id].frozen_trajectory
    assert frozen_spec is not None
    assert frozen_spec.segments == []
    assert frozen_spec.y0 == pytest.approx(last_value)

    result = builder.run(1.0 * Unit("s"))
    sp_series = result["outputs"]["setpoints"][f"{controller_cfg.cv_tag}.sp"]
    assert sp_series["value"][-1] == pytest.approx(last_value)


def test_updating_trajectory_after_freeze_restores_segments():
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

    controller_cfg = builder.add_controller(
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
                {"type": "hold", "duration": 5.0, "value": 0.0},
                {"type": "hold", "duration": 5.0, "value": 2.0},
            ],
        },
    )

    builder.run(12.0 * Unit("s"))

    builder.add_sensor(
        SampledDelayedSensor.__name__,
        {
            "measurement_tag": "x",
            "alias_tag": "x_meas_secondary",
            "unit": "1",
            "sampling_period": 0.0,
            "deadtime": 0.0,
        },
    )

    cfg = builder.controller_configs[controller_cfg.id]
    assert cfg.frozen_trajectory is not None

    new_segments = [
        {"type": "hold", "duration": 3.0, "value": 3.0},
        {"type": "hold", "duration": 3.0, "value": 4.0},
    ]
    builder.update_controller_trajectory(
        controller_cfg.id,
        {
            "y0": 1.5,
            "unit": "1",
            "segments": new_segments,
        },
    )

    cfg = builder.controller_configs[controller_cfg.id]
    assert cfg.frozen_trajectory is None
    assert cfg.trajectory.segments == new_segments

    result = builder.run(6.0 * Unit("s"))
    sp_series = result["outputs"]["setpoints"][f"{controller_cfg.cv_tag}.sp"]
    assert sp_series["value"][-1] == pytest.approx(4.0)
