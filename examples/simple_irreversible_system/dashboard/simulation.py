from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, List

import numpy as np
from astropy.units import UnitBase

from modular_simulation.control_system.controller import Controller, ControllerMode
from modular_simulation.framework.system import System
from modular_simulation.usables.tag_info import TagData


def _format_unit(unit: UnitBase | None) -> str:
    if unit is None:
        return ""
    return str(unit)


def _coerce_value(value: Any) -> float | List[float]:
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr.tolist()


class SimulationManager:
    """Run a system simulation in the background and expose structured history."""

    def __init__(self, system: System, *, history_window: int = 2000) -> None:
        self.system = system
        self.history_window = history_window
        self.speed_factor: float = 1.0

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._dt_seconds = self._extract_seconds(system.dt)

        self._top_level_controllers: List[Controller] = list(system.usable_quantities.controllers)

        self._sensor_units: Dict[str, str] = {
            sensor.alias_tag: _format_unit(sensor.unit)
            for sensor in system.usable_quantities.sensors
        }
        self._sensor_history: Dict[str, Deque[Dict[str, Any]]] = {
            tag: deque(maxlen=history_window) for tag in self._sensor_units
        }

        control_elements = system.measurable_quantities.control_elements
        self._manipulated_units: Dict[str, str] = {
            tag: _format_unit(unit) for tag, unit in control_elements.tag_unit_info.items()
        }
        self._manipulated_history: Dict[str, Deque[Dict[str, Any]]] = {
            tag: deque(maxlen=history_window) for tag in control_elements.tag_list
        }

        self._setpoint_units: Dict[str, str] = {}
        self._setpoint_history: Dict[str, Deque[Dict[str, Any]]] = {}
        for controller in self._iter_controllers():
            self._setpoint_units[controller.cv_tag] = _format_unit(controller.sp_trajectory.unit)
            self._setpoint_history[controller.cv_tag] = deque(maxlen=history_window)

        with self._lock:
            # Prime the usable quantities so that measurement history exists even before stepping.
            self.system.usable_quantities.update(self.system.time)
            self._collect_latest_samples()

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def set_speed(self, factor: float) -> float:
        if factor <= 0:
            raise ValueError("Speed factor must be positive.")
        with self._lock:
            self.speed_factor = float(factor)
            return self.speed_factor

    def controllers_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._build_controller_payload()

    def set_controller_mode(self, cv_tag: str, mode: ControllerMode | str) -> Dict[str, Any]:
        with self._lock:
            self.system.set_controller_mode(cv_tag, mode)
            self._collect_latest_samples()
            return self._build_controller_payload()

    def set_controller_setpoint(self, cv_tag: str, value: float) -> Dict[str, Any]:
        with self._lock:
            controller = self.system.controller_dictionary.get(cv_tag)
            if controller is None:
                raise ValueError(
                    f"Specified cv_tag '{cv_tag}' does not correspond to any defined controllers. "
                )
            if controller.mode != ControllerMode.AUTO:
                raise ValueError(
                    f"Controller '{cv_tag}' must be in AUTO mode before changing its setpoint."
                )
            self.system.extend_controller_trajectory(cv_tag, float(value))
            self._collect_latest_samples()
            return self._build_controller_payload()

    def snapshot(self, max_points: int | None = None) -> Dict[str, Any]:
        with self._lock:
            return {
                "time": float(self.system.time),
                "speed": self.speed_factor,
                "sensors": self._history_payload(self._sensor_history, self._sensor_units, max_points),
                "setpoints": self._history_payload(
                    self._setpoint_history, self._setpoint_units, max_points
                ),
                "manipulated": self._history_payload(
                    self._manipulated_history, self._manipulated_units, max_points
                ),
            }

    def _build_controller_payload(self) -> Dict[str, Any]:
        controllers = [self._controller_to_dict(controller) for controller in self._top_level_controllers]
        return {
            "time": float(self.system.time),
            "speed": self.speed_factor,
            "controllers": controllers,
        }

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()
            with self._lock:
                self.system.step(self.system.dt)
                self._collect_latest_samples()
                sleep_target = max(self._dt_seconds / self.speed_factor, 0.0)
            elapsed = time.time() - start
            remaining = max(sleep_target - elapsed, 0.0)
            if self._stop_event.wait(remaining):
                break

    def _collect_latest_samples(self) -> None:
        self._record_sensor_samples()
        self._record_setpoints()
        self._record_manipulated()

    def _record_sensor_samples(self) -> None:
        for sensor in self.system.usable_quantities.sensors:
            history = sensor.measurement_history
            if not history:
                continue
            sample = history[-1]
            self._sensor_history[sensor.alias_tag].append(self._serialize_tag_data(sample))

    def _record_setpoints(self) -> None:
        current_time = float(self.system.time)
        for controller in self._iter_controllers():
            sp_value = controller.sp_trajectory(current_time)
            self._setpoint_history[controller.cv_tag].append(
                {"time": current_time, "value": _coerce_value(sp_value), "ok": True}
            )

    def _record_manipulated(self) -> None:
        current_time = float(self.system.time)
        control_elements = self.system.measurable_quantities.control_elements
        for tag in control_elements.tag_list:
            value = getattr(control_elements, tag)
            self._manipulated_history[tag].append(
                {"time": current_time, "value": _coerce_value(value), "ok": True}
            )

    def _controller_to_dict(self, controller: Controller) -> Dict[str, Any]:
        pv_data = self._safe_cv_data(controller)
        control_elements = self.system.measurable_quantities.control_elements
        mv_value = getattr(control_elements, controller.mv_tag)
        controller_info: Dict[str, Any] = {
            "cv_tag": controller.cv_tag,
            "mode": controller.mode.name,
            "available_modes": [mode.name for mode in ControllerMode],
            "setpoint": {
                "value": _coerce_value(controller.sp_trajectory(self.system.time)),
                "unit": self._setpoint_units.get(controller.cv_tag, ""),
            },
            "pv": {
                "value": _coerce_value(pv_data.value) if pv_data is not None else None,
                "ok": bool(pv_data.ok) if pv_data is not None else False,
                "time": float(pv_data.time) if pv_data is not None else float(self.system.time),
                "unit": self._setpoint_units.get(controller.cv_tag, ""),
            },
            "manipulated": {
                "tag": controller.mv_tag,
                "value": _coerce_value(mv_value),
                "unit": self._manipulated_units.get(controller.mv_tag, ""),
            },
            "children": [],
        }
        if controller.cascade_controller is not None:
            controller_info["children"] = [self._controller_to_dict(controller.cascade_controller)]
        return controller_info

    def _history_payload(
        self,
        history_map: Dict[str, Deque[Dict[str, Any]]],
        unit_map: Dict[str, str],
        max_points: int | None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for tag, history in history_map.items():
            entries = list(history)
            if max_points is not None and len(entries) > max_points:
                entries = entries[-max_points:]
            payload[tag] = {"unit": unit_map.get(tag, ""), "data": entries}
        return payload

    def _iter_controllers(self) -> Iterable[Controller]:
        for controller in self._top_level_controllers:
            current: Controller | None = controller
            while current is not None:
                yield current
                current = current.cascade_controller

    def _safe_cv_data(self, controller: Controller) -> TagData | None:
        try:
            return controller._cv_getter()  # type: ignore[attr-defined]
        except Exception:
            return None

    def _serialize_tag_data(self, tag_data: TagData) -> Dict[str, Any]:
        return {
            "time": float(tag_data.time),
            "value": _coerce_value(tag_data.value),
            "ok": bool(tag_data.ok),
        }

    @staticmethod
    def _extract_seconds(dt: Any) -> float:
        if hasattr(dt, "to_value"):
            try:
                return float(dt.to_value("s"))  # type: ignore[arg-type]
            except Exception:
                pass
            value = getattr(dt, "value", dt)
            return float(value)
        return float(dt)

