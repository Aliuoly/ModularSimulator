from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from astropy.units import Quantity  # type: ignore
from flask import Flask, jsonify, render_template, request
from pydantic import ValidationError as PydanticValidationError
from pydantic_core import ValidationError as CoreValidationError

from .builder import (
    SimulationBuilder,
    _parse_quantity,
)


def _sensor_to_payload(config) -> Dict[str, Any]:
    return {
        "id": config.id,
        "type": config.name,
        "params": config.raw,
    }


def _controller_to_payload(config) -> Dict[str, Any]:
    payload = {
        "id": config.id,
        "type": config.name,
        "params": config.raw,
        "trajectory": {
            "y0": config.trajectory.y0,
            "unit": config.trajectory.unit,
            "segments": config.trajectory.segments,
        },
        "parent_id": config.parent_id,
        "child_id": config.child_id,
    }
    return payload


def _calculation_to_payload(config) -> Dict[str, Any]:
    return {
        "id": config.id,
        "type": config.name,
        "params": config.raw,
        "outputs": list(config.output_tags),
    }


ValidationErrorTypes = (PydanticValidationError, CoreValidationError)


def _validation_details(exc: Exception) -> Any:
    if hasattr(exc, "json"):
        try:
            return json.loads(exc.json())
        except Exception:
            pass
    if hasattr(exc, "errors"):
        try:
            return exc.errors()  # type: ignore[no-any-return]
        except Exception:
            pass
    return str(exc)


def _error_response(message: str, *, details: Optional[Any] = None, status: int = 400):
    payload: Dict[str, Any] = {"error": message}
    if details is not None:
        payload["details"] = details
    return jsonify(payload), status


def create_app(builder: SimulationBuilder) -> Flask:
    app = Flask(
        __name__,
        static_folder=str(Path(__file__).with_name("static")),
        template_folder=str(Path(__file__).with_name("templates")),
    )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/metadata")
    def metadata():
        return jsonify(
            {
                "measurables": builder.measurable_metadata(),
                "control_elements": builder.control_element_tags(),
                "sensor_types": builder.available_sensor_types(),
                "controller_types": builder.available_controller_types(),
                "calculation_types": builder.available_calculation_types(),
                "usable_tags": builder.available_usable_tags(),
                "setpoint_tags": builder.available_setpoint_tags(),
                "messages": builder.messages(),
            }
        )

    @app.get("/api/sensors")
    def list_sensors():
        return jsonify([_sensor_to_payload(cfg) for cfg in builder.sensor_configs])

    @app.post("/api/sensors")
    def add_sensor():
        data = request.get_json(force=True)
        try:
            config = builder.add_sensor(data["type"], data.get("params", {}))
        except ValidationErrorTypes as exc:
            return _error_response("Invalid sensor configuration.", details=_validation_details(exc))
        except ValueError as exc:
            return _error_response(str(exc))
        return jsonify(_sensor_to_payload(config)), 201

    @app.delete("/api/sensors/<sensor_id>")
    def delete_sensor(sensor_id: str):
        builder.remove_sensor(sensor_id)
        return ("", 204)

    @app.get("/api/controllers")
    def list_controllers():
        return jsonify([_controller_to_payload(cfg) for cfg in builder.controller_configs.values()])

    @app.post("/api/controllers")
    def add_controller():
        data = request.get_json(force=True)
        try:
            config = builder.add_controller(
                data["type"],
                data.get("params", {}),
                data.get("trajectory", {}),
                parent_id=data.get("parent_id"),
            )
        except ValidationErrorTypes as exc:
            return _error_response("Invalid controller configuration.", details=_validation_details(exc))
        except ValueError as exc:
            return _error_response(str(exc))
        return jsonify(_controller_to_payload(config)), 201

    @app.delete("/api/controllers/<controller_id>")
    def delete_controller(controller_id: str):
        builder.remove_controller(controller_id)
        return ("", 204)

    @app.get("/api/calculations")
    def list_calculations():
        return jsonify([_calculation_to_payload(cfg) for cfg in builder.calculation_configs.values()])

    @app.post("/api/calculations")
    def add_calculation():
        data = request.get_json(force=True)
        try:
            config = builder.add_calculation(data["type"], data.get("params", {}))
        except ValidationErrorTypes as exc:
            return _error_response("Invalid calculation configuration.", details=_validation_details(exc))
        except ValueError as exc:
            return _error_response(str(exc))
        return jsonify(_calculation_to_payload(config)), 201

    @app.delete("/api/calculations/<calc_id>")
    def delete_calculation(calc_id: str):
        builder.remove_calculation(calc_id)
        return ("", 204)

    @app.post("/api/calculations/upload")
    def upload_calculation():
        if "file" not in request.files:
            return _error_response("No file uploaded.")
        file = request.files["file"]
        if not file.filename.endswith(".py"):
            return _error_response("Upload a .py file.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
            file.save(tmp.name)
            try:
                module = builder.register_calculation_module(tmp.name)
            except ValueError as exc:
                return _error_response(str(exc))
        payload = {
            "module_id": module.id,
            "classes": [cls.__name__ for cls in module.classes.values()],
        }
        return jsonify(payload), 201

    @app.get("/api/plots")
    def get_plots():
        layout = builder.plot_layout
        return jsonify(
            {
                "rows": layout.rows,
                "cols": layout.cols,
                "lines": [
                    {
                        "panel": line.panel,
                        "tag": line.tag,
                        "label": line.label,
                        "color": line.color,
                        "style": line.style,
                    }
                    for line in layout.lines
                ],
            }
        )

    @app.post("/api/plots")
    def set_plots():
        data = request.get_json(force=True)
        try:
            layout = builder.set_plot_layout(
                int(data.get("rows", 1)),
                int(data.get("cols", 1)),
                data.get("lines", []),
            )
        except ValueError as exc:
            return _error_response(str(exc))
        return jsonify(
            {
                "rows": layout.rows,
                "cols": layout.cols,
                "lines": [
                    {
                        "panel": line.panel,
                        "tag": line.tag,
                        "label": line.label,
                        "color": line.color,
                        "style": line.style,
                    }
                    for line in layout.lines
                ],
            }
        )

    @app.post("/api/run")
    def run_simulation():
        data = request.get_json(force=True) if request.data else {}
        duration: Optional[Quantity] = None
        if data and data.get("duration"):
            duration = _parse_quantity(data["duration"])
        try:
            result = builder.run(duration)
        except ValueError as exc:
            return _error_response(str(exc))
        return jsonify(
            {
                "time": result["time"],
                "outputs": result["outputs"],
                "figure": result["figure"],
                "messages": result.get("messages", []),
            }
        )

    return app


def launch_ui(
    system_class,
    measurable_quantities,
    dt,
    *,
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    solver_options: Optional[Dict[str, Any]] = None,
    use_numba: bool = False,
    numba_options: Optional[Dict[str, Any]] = None,
    record_history: bool = True,
) -> None:
    builder = SimulationBuilder(
        system_class=system_class,
        measurable_quantities=measurable_quantities,
        dt=dt,
        solver_options=solver_options,
        use_numba=use_numba,
        numba_options=numba_options,
        record_history=record_history,
    )
    app = create_app(builder)
    app.run(host=host, port=port, debug=debug)


__all__ = ["create_app", "launch_ui"]
