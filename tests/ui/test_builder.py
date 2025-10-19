from __future__ import annotations

import math

import numpy as np
from astropy.units import Quantity, Unit
from pydantic import Field
from typing import Annotated

from modular_simulation.measurables.base_classes import AlgebraicStates, Constants, ControlElements, States
from modular_simulation.measurables.measurable_quantities import MeasurableQuantities
from modular_simulation.ui import SimulationBuilder
from modular_simulation.ui.app import create_app
from modular_simulation.usables.sensors.sampled_delayed_sensor import SampledDelayedSensor
from modular_simulation.usables.controllers.controller import Controller
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


def build_configured_builder() -> SimulationBuilder:
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

    return builder


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
    builder = build_configured_builder()

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


def test_controller_payload_is_json_serializable():
    builder = build_configured_builder()

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

    app = create_app(builder)
    client = app.test_client()
    response = client.get("/api/controllers")

    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload, list)
    assert payload
    controller = payload[0]
    assert "sp_trajectory" not in controller["params"]
    assert controller["trajectory"]["segments"][0]["magnitude"] == 1.0


def test_controller_validation_error_returns_json():
    builder = build_configured_builder()
    app = create_app(builder)
    client = app.test_client()

    response = client.post(
        "/api/controllers",
        json={
            "type": PIDController.__name__,
            "params": {},
            "trajectory": {"y0": 0.0, "unit": "1", "segments": []},
        },
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "Invalid controller configuration."
    assert any(error["loc"][-1] == "mv_tag" for error in payload["details"])


def test_metadata_serializes_infinite_quantity_defaults():
    builder = build_configured_builder()

    class InfiniteRangeController(Controller):
        extra_limit: Quantity = Field(default=math.inf * Unit("1"))

        def _control_algorithm(self, t, cv, sp):
            return 0.0

    builder.controller_types["InfiniteRangeController"] = InfiniteRangeController

    app = create_app(builder)
    client = app.test_client()

    response = client.get("/api/metadata")

    assert response.status_code == 200
    payload = response.get_json()
    controller = next(
        item for item in payload["controller_types"] if item["name"] == "InfiniteRangeController"
    )
    extra_field = next(field for field in controller["fields"] if field["name"] == "extra_limit")

    assert extra_field["default"]["value"] == "Infinity"


def test_sensor_payload_handles_composite_units():
    builder = build_configured_builder()
    builder.add_sensor(
        SampledDelayedSensor.__name__,
        {
            "measurement_tag": "x",
            "alias_tag": "x_complex",
            "unit": "kg/(m s2)",
            "sampling_period": 0.0,
            "deadtime": 0.0,
        },
    )

    app = create_app(builder)
    client = app.test_client()
    response = client.get("/api/sensors")

    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload, list)
    composite_units = [sensor["params"]["unit"] for sensor in payload if sensor["params"]["alias_tag"] == "x_complex"]
    assert composite_units == ["kg / (m s2)"]
