from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request

from examples.simple_irreversible_system.run_simulation import make_systems

from .simulation import SimulationManager


def create_app(system: Any | None = None) -> Flask:
    """Create a Flask application with a running simulation."""
    if system is None:
        system = make_systems()["normal"]

    manager = SimulationManager(system)
    manager.start()

    package_root = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(package_root / "templates"),
        static_folder=str(package_root / "static"),
    )
    app.config["simulation_manager"] = manager
    atexit.register(manager.stop)

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/api/trends")
    def trends():
        max_points = request.args.get("points", default=None, type=int)
        return jsonify(manager.snapshot(max_points))

    @app.route("/api/controllers")
    def controllers():
        return jsonify(manager.controllers_snapshot())

    @app.post("/api/controllers/<cv_tag>/mode")
    def change_mode(cv_tag: str):
        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode")
        if mode is None:
            abort(400, "Missing 'mode' in request body.")
        try:
            snapshot = manager.set_controller_mode(cv_tag, mode)
        except ValueError as exc:
            abort(400, str(exc))
        return jsonify(snapshot)

    @app.post("/api/controllers/<cv_tag>/setpoint")
    def change_setpoint(cv_tag: str):
        payload = request.get_json(silent=True) or {}
        if "setpoint" not in payload:
            abort(400, "Missing 'setpoint' in request body.")
        try:
            setpoint_value = float(payload["setpoint"])
        except (TypeError, ValueError) as exc:
            abort(400, f"Invalid setpoint value: {exc}")
        try:
            snapshot = manager.set_controller_setpoint(cv_tag, setpoint_value)
        except ValueError as exc:
            abort(400, str(exc))
        return jsonify(snapshot)

    @app.post("/api/speed")
    def update_speed():
        payload = request.get_json(silent=True) or {}
        if "factor" not in payload:
            abort(400, "Missing 'factor' in request body.")
        try:
            factor = float(payload["factor"])
        except (TypeError, ValueError) as exc:
            abort(400, f"Invalid speed factor: {exc}")
        try:
            speed = manager.set_speed(factor)
        except ValueError as exc:
            abort(400, str(exc))
        return jsonify({"speed": speed})

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(debug=True, host="0.0.0.0", port=5000)
