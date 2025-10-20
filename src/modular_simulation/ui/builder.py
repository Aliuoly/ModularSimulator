from __future__ import annotations

import base64
import copy
import importlib
import importlib.util
import inspect
import io
import math
import pkgutil
import types
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Type, Union, get_args, get_origin

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from astropy.units import Quantity, Unit, UnitBase  # type: ignore
from pydantic_core import PydanticUndefined
from modular_simulation.framework.system import System
from modular_simulation.measurables.measurable_quantities import MeasurableQuantities
from modular_simulation.usables.calculations.calculation import Calculation, TagMetadata
from modular_simulation.usables.controllers.controller import Controller, ControllerMode
from modular_simulation.usables.controllers.trajectory import Trajectory
from modular_simulation.usables.sensors.sensor import Sensor
from modular_simulation.usables.tag_info import TagData
from modular_simulation.usables.usable_quantities import UsableQuantities
from numpy.typing import NDArray


@dataclass
class SensorConfig:
    id: str
    name: str
    cls: Type[Sensor]
    args: Dict[str, Any]
    raw: Dict[str, Any]

    @property
    def alias_tag(self) -> str:
        value = self.args.get("alias_tag")
        if value is None:
            return self.args.get("measurement_tag", "")
        return value

    @property
    def measurement_tag(self) -> str:
        return self.args.get("measurement_tag", "")


@dataclass
class TrajectorySpec:
    y0: float
    unit: str
    segments: List[Dict[str, Any]]


@dataclass
class ControllerConfig:
    id: str
    name: str
    cls: Type[Controller]
    args: Dict[str, Any]
    raw: Dict[str, Any]
    trajectory: TrajectorySpec
    parent_id: Optional[str] = None
    child_id: Optional[str] = None

    @property
    def cv_tag(self) -> str:
        return self.args.get("cv_tag", "")

    @property
    def mv_tag(self) -> str:
        return self.args.get("mv_tag", "")

    @property
    def setpoint_tag(self) -> str:
        cv = self.cv_tag
        if not cv:
            return ""
        return f"{cv}.sp"


@dataclass
class SensorModule:
    id: str
    module: types.ModuleType
    classes: Dict[str, Type[Sensor]]


@dataclass
class ControllerModule:
    id: str
    module: types.ModuleType
    classes: Dict[str, Type[Controller]]


@dataclass
class CalculationModule:
    id: str
    module: types.ModuleType
    classes: Dict[str, Type[Calculation]]


@dataclass
class CalculationConfig:
    id: str
    name: str
    cls: Type[Calculation]
    args: Dict[str, Any]
    raw: Dict[str, Any]
    output_units: Dict[str, UnitBase] = field(default_factory=dict)

    @property
    def output_tags(self) -> Iterable[str]:
        return self.raw.get("outputs", [])


@dataclass
class PlotLine:
    panel: int
    tag: str
    label: Optional[str] = None
    color: Optional[str] = None
    style: Optional[str] = None


@dataclass
class PlotLayout:
    rows: int = 1
    cols: int = 1
    lines: List[PlotLine] = field(default_factory=list)


def _discover_subclasses(base: Type[Any], package: str) -> Dict[str, Type[Any]]:
    module = importlib.import_module(package)
    discovered: Dict[str, Type[Any]] = {}
    if not hasattr(module, "__path__"):
        for name, attr in inspect.getmembers(module, inspect.isclass):
            if issubclass(attr, base) and attr is not base:
                discovered[name] = attr
        return discovered

    prefix = module.__name__ + "."
    for finder, name, is_pkg in pkgutil.walk_packages(module.__path__, prefix):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for cls_name, cls_obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(cls_obj, base) and cls_obj is not base and cls_obj.__module__.startswith(prefix):
                discovered[cls_name] = cls_obj
    return discovered


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _parse_unit(value: Any) -> UnitBase | None:
    if value is None or value == "":
        return None
    return Unit(str(value))


def _parse_quantity(value: Mapping[str, Any]) -> Quantity:
    try:
        magnitude = float(value["value"])
    except (KeyError, TypeError) as exc:  # pragma: no cover - validated by front-end
        raise ValueError("Quantity payload must include 'value'.") from exc
    unit = Unit(str(value.get("unit", "")))
    return magnitude * unit


def _parse_numeric_tuple(value: Mapping[str, Any]) -> Tuple[float, float]:
    return (float(value.get("min", 0.0)), float(value.get("max", 0.0)))


def _parse_quantity_range(value: Mapping[str, Any]) -> Tuple[Quantity, Quantity]:
    lower = _parse_quantity(value.get("lower"))
    upper = _parse_quantity(value.get("upper"))
    return (lower, upper)


def _sanitize_number(value: float) -> float | None:
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _unit_metadata(unit: UnitBase | str | None) -> Dict[str, Any]:
    """Return serialization-friendly metadata for a unit reference."""

    normalized: UnitBase | None
    if isinstance(unit, UnitBase):
        normalized = unit
    elif isinstance(unit, str):
        stripped = unit.strip()
        normalized = Unit(stripped) if stripped else None
    else:
        normalized = None

    unit_text = str(normalized) if normalized is not None else ""
    aliases: List[str] = []

    if normalized is not None:
        raw_aliases = getattr(normalized, "aliases", ())
        aliases = sorted({str(alias) for alias in raw_aliases if str(alias)})

    return {
        "unit": unit_text,
        "unit_aliases": aliases,
    }
def _serialize_value(value: Any) -> Any:
    if isinstance(value, Quantity):
        magnitude = _sanitize_number(value.value)
        if magnitude is None:
            return None
        return {"value": magnitude, "unit": str(value.unit)}
    if isinstance(value, UnitBase) or isinstance(value, Unit):
        return str(value)
    if isinstance(value, ControllerMode):
        return value.name
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        sanitized = _sanitize_number(value)
        return sanitized
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.generic):
        return _serialize_value(value.item())
    if isinstance(value, tuple):
        return [_serialize_value(v) for v in value]
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _serialize_tag_history(series: List[TagData]) -> Dict[str, List[float]]:
    times: List[float] = []
    values: List[float] = []
    ok_flags: List[bool] = []

    for sample in series:
        times.append(float(sample.time))
        ok_flags.append(bool(getattr(sample, "ok", True)))

        value = sample.value
        sanitized: Optional[float]
        if isinstance(value, (np.ndarray, np.generic, list, tuple)):
            arr = np.asarray(value).reshape(-1)
            sanitized = _sanitize_number(float(arr[0]))
        else:
            sanitized = _sanitize_number(float(value))

        if sanitized is None:
            values.append(float("nan"))
        else:
            values.append(sanitized)

    return {"time": times, "value": values, "ok": ok_flags}


def _series_to_arrays(series: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Convert a serialized history mapping into NumPy arrays."""

    times = np.asarray(series.get("time", []), dtype=float).reshape(-1)
    values = np.asarray(series.get("value", []), dtype=float).reshape(-1)

    if times.shape != values.shape:
        length = min(times.shape[0], values.shape[0])
        times = times[:length]
        values = values[:length]

    ok_like = series.get("ok")
    ok: Optional[np.ndarray]
    if ok_like is None:
        ok = None
    else:
        ok = np.asarray(ok_like, dtype=bool).reshape(-1)
        if ok.shape != times.shape:
            ok = ok[: times.shape[0]]

    return times, values, ok


class SimulationBuilder:
    """Stateful helper that wires sensors, controllers, and calculations."""

    def __init__(
        self,
        system_class: Type[System],
        measurable_quantities: MeasurableQuantities,
        dt: Quantity,
        *,
        solver_options: Optional[Dict[str, Any]] = None,
        use_numba: bool = False,
        numba_options: Optional[Dict[str, Any]] = None,
        record_history: bool = True,
    ) -> None:
        self.system_class = system_class
        self.base_measurables = measurable_quantities
        self.dt = dt
        self.solver_options = solver_options or {"method": "LSODA"}
        self.use_numba = use_numba
        self.numba_options = numba_options or {"nopython": True, "cache": True}
        self.record_history = record_history

        self.sensor_types = _discover_subclasses(Sensor, "modular_simulation.usables.sensors")
        self.controller_types = _discover_subclasses(Controller, "modular_simulation.usables.controllers")
        self.calculation_types = _discover_subclasses(Calculation, "modular_simulation.usables.calculations")

        self.sensor_modules: Dict[str, SensorModule] = {}
        self.controller_modules: Dict[str, ControllerModule] = {}
        self.sensor_configs: List[SensorConfig] = []
        self.controller_configs: Dict[str, ControllerConfig] = {}
        self.calculation_configs: Dict[str, CalculationConfig] = {}
        self.calculation_modules: Dict[str, CalculationModule] = {}
        self.plot_layout = PlotLayout()
        self.system: Optional[System] = None
        self._messages: List[str] = []
        self._suppress_invalidations = False
        self._suppressed_messages: List[str] = []
        self._history_cache = self._empty_history_structure()
        self._history_lengths = self._empty_length_structure()
        self._elapsed_time = 0.0
        self._time_offset = 0.0
        self._history_bounds: Dict[str, Optional[float]] = {"min": None, "max": None}
        self._time_axis_limits: Tuple[Optional[float], Optional[float]] = (None, None)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_history_structure() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        return {"sensors": {}, "calculations": {}, "setpoints": {}}

    @staticmethod
    def _empty_length_structure() -> Dict[str, Dict[str, int]]:
        return {"sensors": {}, "calculations": {}, "setpoints": {}}

    @property
    def elapsed_time(self) -> float:
        return self._elapsed_time

    def measurable_metadata(self) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        for category in ("states", "algebraic_states", "control_elements", "constants"):
            model = getattr(self.base_measurables, category)
            for tag in model.tag_list:
                unit = model.tag_unit_info[tag]
                metadata = _unit_metadata(unit)
                items.append({
                    "tag": tag,
                    "category": category,
                    **metadata,
                })
        return items

    def control_element_tags(self) -> List[str]:
        return list(self.base_measurables.control_elements.tag_list)

    def control_element_unit_options(self) -> Dict[str, Dict[str, List[str] | str]]:
        options: Dict[str, Dict[str, List[str] | str]] = {}
        for tag, unit in self.base_measurables.control_elements.tag_unit_info.items():
            options[tag] = _unit_metadata(unit)
        return options

    def usable_tag_unit_options(self) -> Dict[str, Dict[str, List[str] | str]]:
        options: Dict[str, Dict[str, List[str] | str]] = {}
        for tag, unit in self.base_measurables.tag_unit_info.items():
            options[tag] = _unit_metadata(unit)
        for cfg in self.sensor_configs:
            resolved_unit = cfg.args.get("unit")
            measurement_unit = self.base_measurables.tag_unit_info.get(cfg.measurement_tag)
            if resolved_unit is None:
                resolved_unit = measurement_unit
            if resolved_unit is None:
                continue
            options[cfg.alias_tag] = _unit_metadata(resolved_unit)
        for calc in self.calculation_configs.values():
            for tag, unit in calc.output_units.items():
                options[tag] = _unit_metadata(unit)
        return options

    def available_sensor_types(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for cls in self.sensor_types.values():
            if cls is Sensor:
                continue
            items.append(self._describe_model(cls))
        for module in self.sensor_modules.values():
            for cls in module.classes.values():
                items.append(self._describe_model(cls))
        return items

    def available_controller_types(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for cls in self.controller_types.values():
            if cls is Controller:
                continue
            info = self._describe_model(cls, exclude_fields={"sp_trajectory", "cascade_controller"})
            items.append(info)
        for module in self.controller_modules.values():
            for cls in module.classes.values():
                info = self._describe_model(
                    cls,
                    exclude_fields={"sp_trajectory", "cascade_controller"},
                )
                items.append(info)
        return items

    def available_calculation_types(self) -> List[Dict[str, Any]]:
        builtin = []
        for cls in self.calculation_types.values():
            if cls is Calculation:
                continue
            builtin.append(self._describe_model(cls))
        uploaded = []
        for module in self.calculation_modules.values():
            for cls in module.classes.values():
                uploaded.append(self._describe_model(cls))
        return builtin + uploaded

    def messages(self) -> List[str]:
        msgs = list(self._messages)
        self._messages.clear()
        return msgs

    # ------------------------------------------------------------------
    # CRUD for sensors
    # ------------------------------------------------------------------
    def add_sensor(self, sensor_type: str, params: Dict[str, Any]) -> SensorConfig:
        cls = self._resolve_sensor_type(sensor_type)
        args = self._convert_arguments(cls, params)
        measurement_tag = args.get("measurement_tag")
        if measurement_tag is None:
            raise ValueError("Sensor configuration requires a 'measurement_tag'.")
        measurement_unit = self.base_measurables.tag_unit_info.get(measurement_tag)
        if measurement_unit is None:
            raise ValueError(
                f"Measurement tag '{measurement_tag}' is not defined in the measurable quantities."
            )
        sensor_unit = args.get("unit")
        if sensor_unit is None:
            args["unit"] = measurement_unit
        elif not sensor_unit.is_equivalent(measurement_unit):
            raise ValueError(
                "Sensor unit must be compatible with the measurement tag's unit. "
                f"Received '{sensor_unit}' for tag '{measurement_tag}' ({measurement_unit})."
            )
        instance = cls(**args)
        config = SensorConfig(
            id=str(uuid.uuid4()),
            name=cls.__name__,
            cls=cls,
            args=args,
            raw=self._serialize_model(instance),
        )
        self.sensor_configs.append(config)
        self.invalidate("Sensor definitions changed; system will be rebuilt on next run.")
        return config

    def remove_sensor(self, sensor_id: str) -> None:
        before = len(self.sensor_configs)
        self.sensor_configs = [cfg for cfg in self.sensor_configs if cfg.id != sensor_id]
        if len(self.sensor_configs) != before:
            self.invalidate("Sensor removed; system will restart from initial conditions on next run.")

    # ------------------------------------------------------------------
    # CRUD for controllers
    # ------------------------------------------------------------------
    def add_controller(
        self,
        controller_type: str,
        params: Dict[str, Any],
        trajectory: Dict[str, Any],
        *,
        parent_id: Optional[str] = None,
    ) -> ControllerConfig:
        cls = self._resolve_controller_type(controller_type)
        args = self._convert_arguments(cls, params, exclude={"sp_trajectory", "cascade_controller"})
        traj_spec = TrajectorySpec(
            y0=float(trajectory["y0"]),
            unit=str(trajectory["unit"]),
            segments=list(trajectory.get("segments", [])),
        )
        instance = cls(sp_trajectory=self._build_trajectory(traj_spec), **args)
        raw = self._serialize_model(instance)
        raw["trajectory"] = trajectory

        controller_id = str(uuid.uuid4())
        config = ControllerConfig(
            id=controller_id,
            name=cls.__name__,
            cls=cls,
            args=args,
            raw=raw,
            trajectory=traj_spec,
            parent_id=parent_id,
        )
        self.controller_configs[controller_id] = config
        if parent_id:
            parent = self.controller_configs[parent_id]
            parent.child_id = controller_id
        self.invalidate("Controller definitions changed; system will be rebuilt on next run.")
        return config

    def remove_controller(self, controller_id: str) -> None:
        cfg = self.controller_configs.pop(controller_id, None)
        if cfg is None:
            return
        if cfg.parent_id and cfg.parent_id in self.controller_configs:
            parent = self.controller_configs[cfg.parent_id]
            if parent.child_id == controller_id:
                parent.child_id = None
        self.invalidate("Controller removed; system will restart on next run.")

    def update_controller(
        self, controller_id: str, params: Optional[Dict[str, Any]]
    ) -> ControllerConfig:
        config = self.controller_configs.get(controller_id)
        if config is None:
            raise ValueError(f"Unknown controller id '{controller_id}'.")
        if params is None:
            raise ValueError("Controller update requires a 'params' mapping.")

        updates = self._convert_arguments(
            config.cls, params, exclude={"sp_trajectory", "cascade_controller"}
        )
        new_args = dict(config.args)
        new_args.update(updates)

        instance = config.cls(
            sp_trajectory=self._build_trajectory(config.trajectory), **new_args
        )
        raw = self._serialize_model(instance)
        raw["trajectory"] = {
            "y0": config.trajectory.y0,
            "unit": config.trajectory.unit,
            "segments": [dict(segment) for segment in config.trajectory.segments],
        }

        config.args = new_args
        config.raw = raw
        self.invalidate(
            "Controller parameters updated; system will be rebuilt on next run."
        )
        return config

    def update_controller_trajectory(
        self, controller_id: str, trajectory: Optional[Dict[str, Any]]
    ) -> ControllerConfig:
        config = self.controller_configs.get(controller_id)
        if config is None:
            raise ValueError(f"Unknown controller id '{controller_id}'.")
        if trajectory is None:
            raise ValueError("Trajectory payload is required.")

        try:
            y0 = float(trajectory.get("y0", config.trajectory.y0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Trajectory requires a numeric 'y0' value.") from exc
        unit = str(trajectory.get("unit", config.trajectory.unit))
        segments_payload = trajectory.get("segments", [])
        if not isinstance(segments_payload, list):
            raise ValueError("Trajectory 'segments' must be a list.")

        segments = [dict(segment) for segment in segments_payload]
        traj_spec = TrajectorySpec(y0=y0, unit=unit, segments=segments)
        instance = config.cls(sp_trajectory=self._build_trajectory(traj_spec), **config.args)
        raw = self._serialize_model(instance)
        raw["trajectory"] = {
            "y0": traj_spec.y0,
            "unit": traj_spec.unit,
            "segments": [dict(segment) for segment in traj_spec.segments],
        }

        config.trajectory = traj_spec
        config.raw = raw
        self.invalidate("Controller trajectory updated; system will be rebuilt on next run.")
        return config

    # ------------------------------------------------------------------
    # CRUD for calculations
    # ------------------------------------------------------------------
    def add_calculation(self, calculation_type: str, params: Dict[str, Any]) -> CalculationConfig:
        cls = self._resolve_calculation_type(calculation_type)
        args = self._convert_arguments(cls, params)
        instance = cls(**args)
        raw = self._serialize_model(instance)
        raw["outputs"] = list(instance._output_tag_info_dict.keys())
        output_units = {
            tag: info.unit for tag, info in instance._output_tag_info_dict.items()
        }
        config = CalculationConfig(
            id=str(uuid.uuid4()),
            name=cls.__name__,
            cls=cls,
            args=args,
            raw=raw,
            output_units=output_units,
        )
        self.calculation_configs[config.id] = config
        self.invalidate("Calculation definitions changed; system will be rebuilt on next run.")
        return config

    def remove_calculation(self, calculation_id: str) -> None:
        if calculation_id in self.calculation_configs:
            del self.calculation_configs[calculation_id]
            self.invalidate("Calculation removed; system will restart on next run.")

    def register_sensor_module(self, file_path: str) -> SensorModule:
        module_id = str(uuid.uuid4())
        spec = importlib.util.spec_from_file_location(f"user_sensor_{module_id}", file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load sensor module.")
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        loader.exec_module(module)  # type: ignore[call-arg]
        classes: Dict[str, Type[Sensor]] = {}
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Sensor) and cls is not Sensor:
                classes[cls.__name__] = cls
        self._replace_uploaded_sensor_classes(classes)
        sensor_module = SensorModule(id=module_id, module=module, classes=classes)
        if classes:
            self.sensor_modules[module_id] = sensor_module
        return sensor_module

    def register_controller_module(self, file_path: str) -> ControllerModule:
        module_id = str(uuid.uuid4())
        spec = importlib.util.spec_from_file_location(f"user_controller_{module_id}", file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load controller module.")
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        loader.exec_module(module)  # type: ignore[call-arg]
        classes: Dict[str, Type[Controller]] = {}
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Controller) and cls is not Controller:
                classes[cls.__name__] = cls
        self._replace_uploaded_controller_classes(classes)
        controller_module = ControllerModule(id=module_id, module=module, classes=classes)
        if classes:
            self.controller_modules[module_id] = controller_module
        return controller_module

    def register_calculation_module(self, file_path: str) -> CalculationModule:
        module_id = str(uuid.uuid4())
        spec = importlib.util.spec_from_file_location(f"user_calculation_{module_id}", file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load calculation module.")
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        loader.exec_module(module)  # type: ignore[call-arg]
        classes: Dict[str, Type[Calculation]] = {}
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Calculation) and cls is not Calculation:
                classes[cls.__name__] = cls
        self._replace_uploaded_calculation_classes(classes)
        calc_module = CalculationModule(id=module_id, module=module, classes=classes)
        if classes:
            self.calculation_modules[module_id] = calc_module
        return calc_module

    # ------------------------------------------------------------------
    # Plot configuration
    # ------------------------------------------------------------------
    def set_plot_layout(self, rows: int, cols: int, lines: List[Dict[str, Any]]) -> PlotLayout:
        allowed_tags = set(self.available_usable_tags()) | set(
            self.available_setpoint_tags()
        )
        parsed_lines = [
            PlotLine(
                panel=int(line.get("panel", 0)),
                tag=str(line["tag"]),
                label=line.get("label"),
                color=line.get("color"),
                style=line.get("style"),
            )
            for line in lines
        ]
        for parsed in parsed_lines:
            if parsed.tag not in allowed_tags:
                raise ValueError(
                    f"Plot tag '{parsed.tag}' is not available from sensors, calculations, or controller setpoints."
                )
        self.plot_layout = PlotLayout(rows=max(rows, 1), cols=max(cols, 1), lines=parsed_lines)
        return self.plot_layout

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, duration: Optional[Quantity] = None) -> Dict[str, Any]:
        system_was_none = self.system is None
        system = self._ensure_system()
        if system_was_none:
            self._history_lengths = self._empty_length_structure()
            self._time_offset = self._elapsed_time
        if duration is None:
            system.step()
        else:
            system.step(duration)
        return self._collect_results(system)

    def _collect_results(self, system: System) -> Dict[str, Any]:
        outputs = self._collect_outputs(system)
        self._update_history_cache(outputs, self._time_offset)
        history = self.get_history_outputs()
        figure = self._render_plot(history, self._time_axis_limits)
        self._elapsed_time = max(self._elapsed_time, self._time_offset + system.time)
        return {
            "time": self._elapsed_time,
            "outputs": history,
            "figure": figure,
            "messages": self.messages(),
            "time_range": self.history_range(),
            "time_axis": self.time_axis_limits(),
        }

    def _collect_outputs(self, system: System) -> Dict[str, Any]:
        measured = system.measured_history
        setpoints = system.setpoint_history
        outputs: Dict[str, Any] = {
            "sensors": {},
            "calculations": {},
            "setpoints": {},
        }
        for tag, series in measured["sensors"].items():
            outputs["sensors"][tag] = _serialize_tag_history(series)
        for tag, series in measured["calculations"].items():
            outputs["calculations"][tag] = _serialize_tag_history(series)
        for tag, series in setpoints.items():
            outputs["setpoints"][tag] = _serialize_tag_history(series)
        return outputs

    def _render_plot(
        self,
        outputs: Dict[str, Any],
        time_axis: Tuple[Optional[float], Optional[float]] | None = None,
    ) -> Optional[str]:
        if not self.plot_layout.lines:
            return self._render_default_plot(outputs, time_axis)
        rows = max(self.plot_layout.rows, 1)
        cols = max(self.plot_layout.cols, 1)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
        axes_flat = axes.ravel()
        for line in self.plot_layout.lines:
            if line.panel >= len(axes_flat):
                continue
            ax = axes_flat[line.panel]
            series = self._resolve_plot_series(line.tag, outputs)
            if series is None:
                continue
            times, values, ok = _series_to_arrays(series)
            mask = np.isfinite(times) & np.isfinite(values)
            if not mask.any():
                continue
            line_kwargs: Dict[str, Any] = {}
            if line.color:
                line_kwargs["color"] = line.color
            if line.style:
                line_kwargs["linestyle"] = line.style
            label = line.label or line.tag
            ax.plot(times[mask], values[mask], label=label, **line_kwargs)
            if ok is not None:
                bad_mask = (~ok.astype(bool)) & mask
                if bad_mask.any():
                    ax.scatter(times[bad_mask], values[bad_mask], marker="x", color="black", label="_nolegend_", zorder=3)
            if not ax.get_ylabel():
                ax.set_ylabel(label)

        for ax in axes_flat:
            if not ax.has_data():
                continue
            ax.grid(True)
            ax.set_xlabel("Time")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best")
            if time_axis:
                lower, upper = time_axis
                if lower is not None or upper is not None:
                    ax.set_xlim(left=lower, right=upper)

        plt.tight_layout()
        return self._finalize_figure(fig)

    def _update_history_cache(self, outputs: Dict[str, Any], offset: float) -> None:
        for category in ("sensors", "calculations", "setpoints"):
            category_outputs = outputs.get(category, {})
            length_tracker = self._history_lengths[category]
            cache = self._history_cache[category]
            for tag, series in category_outputs.items():
                times = list(series.get("time", []))
                values = list(series.get("value", []))
                ok_like = series.get("ok")
                ok_flags = list(ok_like) if ok_like is not None else []

                total_len = len(times)
                previous_len = length_tracker.get(tag, 0)
                if total_len < previous_len:
                    previous_len = 0

                new_times = times[previous_len:]
                new_values = values[previous_len:]
                new_ok = ok_flags[previous_len:] if ok_flags else []

                if not new_times:
                    length_tracker[tag] = total_len
                    continue

                adjusted_times = [float(t) + offset for t in new_times]
                entry = cache.setdefault(tag, {"time": [], "value": [], "ok": []})
                if entry["time"] and adjusted_times and entry["time"][-1] == adjusted_times[0]:
                    adjusted_times.pop(0)
                    if new_values:
                        new_values.pop(0)
                    if new_ok:
                        new_ok.pop(0)
                if not adjusted_times:
                    length_tracker[tag] = total_len
                    continue
                entry["time"].extend(adjusted_times)
                entry["value"].extend(new_values)
                if new_ok:
                    entry["ok"].extend(new_ok)
                else:
                    entry["ok"].extend([True] * len(new_values))

                length_tracker[tag] = total_len

                min_time = adjusted_times[0]
                max_time = adjusted_times[-1]
                current_min = self._history_bounds["min"]
                current_max = self._history_bounds["max"]
                if current_min is None or min_time < current_min:
                    self._history_bounds["min"] = min_time
                if current_max is None or max_time > current_max:
                    self._history_bounds["max"] = max_time

    def get_history_outputs(self) -> Dict[str, Any]:
        return copy.deepcopy(self._history_cache)

    def history_range(self) -> Dict[str, Optional[float]]:
        return {"min": self._history_bounds["min"], "max": self._history_bounds["max"]}

    def time_axis_limits(self) -> Dict[str, Optional[float]]:
        lower, upper = self._time_axis_limits
        return {"min": lower, "max": upper}

    def update_time_axis(
        self,
        lower: Optional[float],
        upper: Optional[float],
    ) -> Dict[str, Any]:
        limits = (
            float(lower) if lower is not None else None,
            float(upper) if upper is not None else None,
        )
        self._time_axis_limits = limits
        figure = self._render_plot(self.get_history_outputs(), limits)
        return {
            "figure": figure,
            "time_axis": self.time_axis_limits(),
            "time_range": self.history_range(),
        }

    def reset_runtime(self) -> None:
        self.system = None
        self._history_cache = self._empty_history_structure()
        self._history_lengths = self._empty_length_structure()
        self._elapsed_time = 0.0
        self._time_offset = 0.0
        self._history_bounds = {"min": None, "max": None}
        self._time_axis_limits = (None, None)
        self._messages.append("Simulation reset to initial conditions.")

    def _render_default_plot(
        self,
        outputs: Dict[str, Any],
        time_axis: Tuple[Optional[float], Optional[float]] | None = None,
    ) -> Optional[str]:
        sensor_series = list(outputs["sensors"].items())
        calculation_series = list(outputs["calculations"].items())
        setpoint_series = list(outputs["setpoints"].items())

        if sensor_series:
            series_to_plot = sensor_series
        elif calculation_series:
            series_to_plot = calculation_series
        else:
            series_to_plot = setpoint_series

        if not series_to_plot:
            return None

        max_panels = max(1, min(len(series_to_plot), 6))
        series_subset = series_to_plot[:max_panels]

        fig, axes = plt.subplots(max_panels, 1, figsize=(8, 3 * max_panels), squeeze=False, sharex=True)
        axes_flat = axes.ravel()

        for ax, (tag, series) in zip(axes_flat, series_subset, strict=False):
            time = series.get("time", [])
            values = series.get("value", [])
            ax.plot(time, values, label=tag)
            ax.set_ylabel(tag)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        axes_flat[max_panels - 1].set_xlabel("Time")
        if time_axis:
            lower, upper = time_axis
            for ax in axes_flat:
                if lower is not None or upper is not None:
                    ax.set_xlim(left=lower, right=upper)
        plt.tight_layout()
        return self._finalize_figure(fig)

    @staticmethod
    def _finalize_figure(fig: Figure) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _resolve_plot_series(self, tag: str, outputs: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
        if tag in outputs["sensors"]:
            return outputs["sensors"][tag]
        if tag in outputs["calculations"]:
            return outputs["calculations"][tag]
        if tag in outputs["setpoints"]:
            return outputs["setpoints"][tag]
        return None

    def invalidate(self, message: str) -> None:
        self.system = None
        if self._suppress_invalidations:
            self._suppressed_messages.append(message)
        else:
            self._messages.append(message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _describe_model(
        self,
        cls: Type[Any],
        *,
        exclude_fields: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        exclude = set(exclude_fields or [])
        fields = []
        for name, field in cls.model_fields.items():
            if name.startswith("_") or name in exclude:
                continue
            field_type = self._field_type_label(field.annotation)
            default_value = None
            if not field.is_required() and field.default is not PydanticUndefined:
                default_value = _serialize_value(field.default) if field.default is not None else None
            tag_metadata = self._field_tag_metadata(field)
            fields.append(
                {
                    "name": name,
                    "type": field_type,
                    "required": field.is_required(),
                    "default": default_value,
                    "description": field.description,
                    "tag_metadata": tag_metadata,
                }
            )
        return {
            "name": cls.__name__,
            "module": cls.__module__,
            "fields": fields,
            "doc": inspect.getdoc(cls) or "",
        }

    def _field_tag_metadata(self, field: Any) -> Optional[Dict[str, Any]]:
        for metadata in getattr(field, "metadata", []) or []:
            if isinstance(metadata, TagMetadata):
                unit = str(metadata.unit) if metadata.unit is not None else ""
                return {
                    "type": metadata.type.name.lower(),
                    "unit": unit,
                    "description": metadata.description,
                }
        return None

    def _replace_uploaded_sensor_classes(self, classes: Mapping[str, Type[Sensor]]) -> None:
        if not classes:
            return
        existing_map: Dict[str, str] = {}
        for module_id, module in list(self.sensor_modules.items()):
            for class_name in list(module.classes.keys()):
                existing_map[class_name] = module_id

        for class_name, cls in classes.items():
            module_id = existing_map.get(class_name)
            if module_id is None:
                continue
            module = self.sensor_modules.get(module_id)
            if module is None:
                continue
            module.classes.pop(class_name, None)
            if not module.classes:
                self.sensor_modules.pop(module_id, None)
            self._reload_existing_sensor_configs(class_name, cls)
        if classes:
            self.invalidate("Sensor modules updated; system will be rebuilt on next run.")

    def _replace_uploaded_controller_classes(
        self, classes: Mapping[str, Type[Controller]]
    ) -> None:
        if not classes:
            return
        existing_map: Dict[str, str] = {}
        for module_id, module in list(self.controller_modules.items()):
            for class_name in list(module.classes.keys()):
                existing_map[class_name] = module_id

        for class_name, cls in classes.items():
            module_id = existing_map.get(class_name)
            if module_id is None:
                continue
            module = self.controller_modules.get(module_id)
            if module is None:
                continue
            module.classes.pop(class_name, None)
            if not module.classes:
                self.controller_modules.pop(module_id, None)
            self._reload_existing_controller_configs(class_name, cls)
        if classes:
            self.invalidate("Controller modules updated; system will be rebuilt on next run.")

    def _replace_uploaded_calculation_classes(self, classes: Mapping[str, Type[Calculation]]) -> None:
        if not classes:
            return
        existing_map: Dict[str, str] = {}
        for module_id, module in list(self.calculation_modules.items()):
            for class_name in list(module.classes.keys()):
                existing_map[class_name] = module_id

        for class_name, cls in classes.items():
            module_id = existing_map.get(class_name)
            if module_id is None:
                continue
            module = self.calculation_modules.get(module_id)
            if module is None:
                continue
            module.classes.pop(class_name, None)
            if not module.classes:
                self.calculation_modules.pop(module_id, None)
            self._reload_existing_calculation_configs(class_name, cls)
        if classes:
            self.invalidate("Calculation modules updated; system will be rebuilt on next run.")

    def _ordered_controller_configs(self) -> List[ControllerConfig]:
        ordered: List[ControllerConfig] = []
        visited: Set[str] = set()

        def traverse(config: ControllerConfig) -> None:
            if config.id in visited:
                return
            visited.add(config.id)
            ordered.append(config)
            if config.child_id and config.child_id in self.controller_configs:
                traverse(self.controller_configs[config.child_id])

        for config in self.controller_configs.values():
            if config.parent_id is None:
                traverse(config)
        for config in self.controller_configs.values():
            if config.id not in visited:
                ordered.append(config)
        return ordered

    def export_configuration(self) -> Dict[str, Any]:
        sensors = [
            {
                "id": cfg.id,
                "type": cfg.name,
                "params": _serialize_value(cfg.raw),
            }
            for cfg in self.sensor_configs
        ]

        controllers: List[Dict[str, Any]] = []
        for cfg in self._ordered_controller_configs():
            params = _serialize_value(cfg.raw)
            params.pop("sp_trajectory", None)
            controllers.append(
                {
                    "id": cfg.id,
                    "type": cfg.name,
                    "params": params,
                    "trajectory": {
                        "y0": cfg.trajectory.y0,
                        "unit": cfg.trajectory.unit,
                        "segments": [dict(segment) for segment in cfg.trajectory.segments],
                    },
                    "parent_id": cfg.parent_id,
                    "child_id": cfg.child_id,
                }
            )

        calculations = [
            {
                "id": cfg.id,
                "type": cfg.name,
                "params": {
                    key: value
                    for key, value in _serialize_value(cfg.raw).items()
                    if key != "outputs"
                },
                "outputs": list(cfg.output_tags),
            }
            for cfg in self.calculation_configs.values()
        ]

        plot = {
            "rows": self.plot_layout.rows,
            "cols": self.plot_layout.cols,
            "lines": [
                {
                    "panel": line.panel,
                    "tag": line.tag,
                    "label": line.label,
                    "color": line.color,
                    "style": line.style,
                }
                for line in self.plot_layout.lines
            ],
        }

        return {
            "sensors": sensors,
            "controllers": controllers,
            "calculations": calculations,
            "plot": plot,
            "time_axis": self.time_axis_limits(),
        }

    def load_configuration(self, payload: Mapping[str, Any]) -> None:
        self.sensor_configs = []
        self.controller_configs = {}
        self.calculation_configs = {}
        self.plot_layout = PlotLayout()
        self.system = None
        self._history_cache = self._empty_history_structure()
        self._history_lengths = self._empty_length_structure()
        self._elapsed_time = 0.0
        self._time_offset = 0.0
        self._history_bounds = {"min": None, "max": None}
        self._time_axis_limits = (None, None)

        sensors = payload.get("sensors", []) or []
        controllers = payload.get("controllers", []) or []
        calculations = payload.get("calculations", []) or []
        plot = payload.get("plot") or None
        time_axis = payload.get("time_axis") or None

        self._suppress_invalidations = True
        self._suppressed_messages.clear()
        try:
            for sensor in sensors:
                sensor_type = sensor.get("type")
                if not sensor_type:
                    raise ValueError("Sensor configuration entries require a 'type'.")
                params = sensor.get("params") or {}
                self.add_sensor(sensor_type, params)

            controller_id_map: Dict[str, str] = {}
            remaining = list(controllers)
            while remaining:
                pending: List[Mapping[str, Any]] = []
                progress = False
                for controller in remaining:
                    controller_type = controller.get("type")
                    if not controller_type:
                        raise ValueError("Controller configuration entries require a 'type'.")
                    parent_old = controller.get("parent_id")
                    if parent_old is not None and parent_old not in controller_id_map:
                        pending.append(controller)
                        continue
                    params = controller.get("params") or {}
                    trajectory = controller.get("trajectory") or {"y0": 0.0, "unit": "", "segments": []}
                    parent_new = controller_id_map.get(parent_old) if parent_old is not None else None
                    config = self.add_controller(
                        controller_type,
                        params,
                        trajectory,
                        parent_id=parent_new,
                    )
                    original_id = controller.get("id")
                    if original_id:
                        controller_id_map[str(original_id)] = config.id
                    progress = True
                if not progress and pending:
                    raise ValueError("Unable to resolve controller parent relationships in configuration.")
                remaining = pending

            for calculation in calculations:
                calculation_type = calculation.get("type")
                if not calculation_type:
                    raise ValueError("Calculation configuration entries require a 'type'.")
                params = dict(calculation.get("params") or {})
                params.pop("outputs", None)
                self.add_calculation(calculation_type, params)

            if plot:
                rows = int(plot.get("rows", 1))
                cols = int(plot.get("cols", 1))
                lines = plot.get("lines", []) or []
                self.set_plot_layout(rows, cols, lines)
        finally:
            self._suppress_invalidations = False
            self._suppressed_messages.clear()

        if time_axis:
            lower = time_axis.get("min")
            upper = time_axis.get("max")

            def _normalize(value: Any) -> Optional[float]:
                if value is None:
                    return None
                if isinstance(value, str) and value.strip() == "":
                    return None
                return float(value)

            self._time_axis_limits = (_normalize(lower), _normalize(upper))

        self._messages.append("Configuration loaded; system will be rebuilt on next run.")

    def _reload_existing_calculation_configs(self, class_name: str, cls: Type[Calculation]) -> None:
        for config in self.calculation_configs.values():
            if config.name != class_name:
                continue
            try:
                instance = cls(**config.args)
            except Exception as exc:  # pragma: no cover - defensive
                self._messages.append(
                    f"Failed to reload calculation '{class_name}' with uploaded version: {exc}"
                )
                continue
            raw = self._serialize_model(instance)
            raw["outputs"] = list(instance._output_tag_info_dict.keys())
            output_units = {tag: info.unit for tag, info in instance._output_tag_info_dict.items()}
            config.cls = cls
            config.name = cls.__name__
            config.raw = raw
            config.output_units = output_units

    def _reload_existing_sensor_configs(self, class_name: str, cls: Type[Sensor]) -> None:
        for config in self.sensor_configs:
            if config.name != class_name:
                continue
            try:
                instance = cls(**config.args)
            except Exception as exc:  # pragma: no cover - defensive
                self._messages.append(
                    f"Failed to reload sensor '{class_name}' with uploaded version: {exc}"
                )
                continue
            config.cls = cls
            config.name = cls.__name__
            config.raw = self._serialize_model(instance)

    def _reload_existing_controller_configs(
        self, class_name: str, cls: Type[Controller]
    ) -> None:
        for config in self.controller_configs.values():
            if config.name != class_name:
                continue
            try:
                instance = cls(
                    sp_trajectory=self._build_trajectory(config.trajectory),
                    **config.args,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._messages.append(
                    f"Failed to reload controller '{class_name}' with uploaded version: {exc}"
                )
                continue
            raw = self._serialize_model(instance)
            raw["trajectory"] = {
                "y0": config.trajectory.y0,
                "unit": config.trajectory.unit,
                "segments": [dict(segment) for segment in config.trajectory.segments],
            }
            config.cls = cls
            config.name = cls.__name__
            config.raw = raw

    def _field_type_label(self, annotation: Any) -> str:
        annotation = _strip_optional(annotation)
        origin = get_origin(annotation)
        if annotation in (str,):
            return "string"
        if annotation in (int,):
            return "integer"
        if annotation in (float, np.float64):
            return "number"
        if annotation is bool:
            return "boolean"
        if annotation is UnitBase or annotation is Unit:
            return "unit"
        if annotation is Quantity:
            return "quantity"
        if origin in (tuple, Tuple):
            args = get_args(annotation)
            if all(arg in (float, int, np.float64) for arg in args):
                return "tuple[number]"
            if all(arg is Quantity for arg in args):
                return "quantity_range"
        if inspect.isclass(annotation) and issubclass(annotation, ControllerMode):
            return "enum"
        return "string"

    def _convert_arguments(
        self,
        cls: Type[Any],
        params: Dict[str, Any],
        *,
        exclude: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        exclude_set = set(exclude or [])
        converted: Dict[str, Any] = {}
        for name, field in cls.model_fields.items():
            if name.startswith("_") or name in exclude_set:
                continue
            if name not in params:
                continue
            value = params[name]
            annotation = _strip_optional(field.annotation)
            origin = get_origin(annotation)
            if annotation is UnitBase or annotation is Unit:
                converted[name] = _parse_unit(value)
            elif annotation is Quantity:
                converted[name] = _parse_quantity(value)
            elif origin in (tuple, Tuple):
                args = get_args(annotation)
                if all(arg in (float, int, np.float64) for arg in args):
                    converted[name] = tuple(float(v) for v in value)
                elif all(arg is Quantity for arg in args):
                    converted[name] = _parse_quantity_range(value)
            elif inspect.isclass(annotation) and issubclass(annotation, ControllerMode):
                if isinstance(value, int):
                    converted[name] = ControllerMode(value)
                else:
                    converted[name] = ControllerMode[value]
            else:
                converted[name] = value
        return converted

    def _build_trajectory(self, spec: TrajectorySpec) -> Trajectory:
        traj = Trajectory(spec.y0, Unit(spec.unit))
        for segment in spec.segments:
            seg_type = segment.get("type")
            if seg_type == "hold":
                duration = float(segment.get("duration", 0.0))
                value = segment.get("value")
                if value is not None:
                    traj.hold(duration=duration, value=float(value))
                else:
                    traj.hold(duration=duration)
            elif seg_type == "step":
                traj.step(float(segment.get("magnitude", 0.0)))
            elif seg_type == "ramp":
                kwargs: Dict[str, Any] = {}
                if "magnitude" in segment:
                    kwargs["magnitude"] = float(segment["magnitude"])
                if "duration" in segment:
                    kwargs["duration"] = float(segment["duration"])
                if "ramprate" in segment:
                    kwargs["ramprate"] = float(segment["ramprate"])
                traj.ramp(**kwargs)
            elif seg_type == "random_walk":
                traj.random_walk(
                    std=float(segment.get("std", 0.1)),
                    duration=float(segment.get("duration", 1.0)),
                    dt=float(segment.get("dt", 1.0)),
                    min=segment.get("min"),
                    max=segment.get("max"),
                    seed=int(segment.get("seed", 0)),
                )
        return traj

    def _serialize_model(self, instance: Any) -> Dict[str, Any]:
        data = {}
        for field_name in instance.__class__.model_fields:
            if field_name.startswith("_"):
                continue
            value = getattr(instance, field_name)
            data[field_name] = _serialize_value(value)
        return data

    def _resolve_sensor_type(self, sensor_type: str) -> Type[Sensor]:
        cls = self.sensor_types.get(sensor_type)
        if cls is not None and cls is not Sensor:
            return cls
        for module in self.sensor_modules.values():
            if sensor_type in module.classes:
                return module.classes[sensor_type]
        raise ValueError(f"Unknown sensor type '{sensor_type}'.")

    def _resolve_controller_type(self, controller_type: str) -> Type[Controller]:
        cls = self.controller_types.get(controller_type)
        if cls is not None and cls is not Controller:
            return cls
        for module in self.controller_modules.values():
            if controller_type in module.classes:
                return module.classes[controller_type]
        raise ValueError(f"Unknown controller type '{controller_type}'.")

    def _resolve_calculation_type(self, calculation_type: str) -> Type[Calculation]:
        if calculation_type in self.calculation_types:
            return self.calculation_types[calculation_type]
        for module in self.calculation_modules.values():
            if calculation_type in module.classes:
                return module.classes[calculation_type]
        raise ValueError(f"Unknown calculation type '{calculation_type}'.")

    def _ensure_system(self) -> System:
        if self.system is not None:
            return self.system
        measurables = MeasurableQuantities(
            states=self.base_measurables.states.model_copy(),
            control_elements=self.base_measurables.control_elements.model_copy(),
            algebraic_states=self.base_measurables.algebraic_states.model_copy(),
            constants=self.base_measurables.constants.model_copy(),
        )
        sensors = [cfg.cls(**cfg.args) for cfg in self.sensor_configs]
        calculations = [cfg.cls(**cfg.args) for cfg in self.calculation_configs.values()]

        controller_instances: List[Controller] = []
        roots = [cfg for cfg in self.controller_configs.values() if cfg.parent_id is None]
        for root in roots:
            controller_instances.append(self._instantiate_controller_chain(root))

        usables = UsableQuantities(
            sensors=sensors,
            calculations=calculations,
            controllers=controller_instances,
            measurable_quantities=measurables,
        )

        self.system = self.system_class(
            dt=self.dt,
            measurable_quantities=measurables,
            usable_quantities=usables,
            solver_options=self.solver_options,
            use_numba=self.use_numba,
            numba_options=self.numba_options,
            record_history=self.record_history,
        )
        return self.system

    def _instantiate_controller_chain(self, config: ControllerConfig) -> Controller:
        controller = config.cls(sp_trajectory=self._build_trajectory(config.trajectory), **config.args)
        if config.child_id is not None:
            child_cfg = self.controller_configs.get(config.child_id)
            if child_cfg is not None:
                controller.cascade_controller = self._instantiate_controller_chain(child_cfg)
        return controller

    def available_usable_tags(self) -> List[str]:
        tags = [cfg.alias_tag for cfg in self.sensor_configs]
        for calc in self.calculation_configs.values():
            tags.extend(calc.output_tags)
        return sorted(set(tags))

    def available_setpoint_tags(self) -> List[str]:
        tags = []
        for controller in self.controller_configs.values():
            tag = controller.setpoint_tag
            if tag:
                tags.append(tag)
        return sorted(set(tags))


__all__ = ["SimulationBuilder"]
