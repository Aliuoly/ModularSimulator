from __future__ import annotations

import base64
import importlib
import importlib.util
import inspect
import io
import math
import pkgutil
import types
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type, Union, get_args, get_origin

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from astropy.units import Quantity, Unit, UnitBase  # type: ignore
from pydantic_core import PydanticUndefined
from modular_simulation.framework.system import System
from modular_simulation.measurables.measurable_quantities import MeasurableQuantities
from modular_simulation.usables.calculations.calculation import Calculation
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

        self.sensor_configs: List[SensorConfig] = []
        self.controller_configs: Dict[str, ControllerConfig] = {}
        self.calculation_configs: Dict[str, CalculationConfig] = {}
        self.calculation_modules: Dict[str, CalculationModule] = {}
        self.plot_layout = PlotLayout()
        self.system: Optional[System] = None
        self._messages: List[str] = []

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def measurable_metadata(self) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        for category in ("states", "algebraic_states", "control_elements", "constants"):
            model = getattr(self.base_measurables, category)
            for tag in model.tag_list:
                unit = model.tag_unit_info[tag]
                items.append({
                    "tag": tag,
                    "category": category,
                    "unit": str(unit),
                })
        return items

    def control_element_tags(self) -> List[str]:
        return list(self.base_measurables.control_elements.tag_list)

    def available_sensor_types(self) -> List[Dict[str, Any]]:
        return [self._describe_model(cls) for cls in self.sensor_types.values()]

    def available_controller_types(self) -> List[Dict[str, Any]]:
        items = []
        for cls in self.controller_types.values():
            if cls is Controller:
                continue
            info = self._describe_model(cls, exclude_fields={"sp_trajectory", "cascade_controller"})
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
        cls = self.sensor_types.get(sensor_type)
        if cls is None:
            raise ValueError(f"Unknown sensor type '{sensor_type}'.")
        args = self._convert_arguments(cls, params)
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
        cls = self.controller_types.get(controller_type)
        if cls is None:
            raise ValueError(f"Unknown controller type '{controller_type}'.")
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

    # ------------------------------------------------------------------
    # CRUD for calculations
    # ------------------------------------------------------------------
    def add_calculation(self, calculation_type: str, params: Dict[str, Any]) -> CalculationConfig:
        cls = self._resolve_calculation_type(calculation_type)
        args = self._convert_arguments(cls, params)
        instance = cls(**args)
        raw = self._serialize_model(instance)
        raw["outputs"] = list(instance._output_tag_info_dict.keys())
        config = CalculationConfig(
            id=str(uuid.uuid4()),
            name=cls.__name__,
            cls=cls,
            args=args,
            raw=raw,
        )
        self.calculation_configs[config.id] = config
        self.invalidate("Calculation definitions changed; system will be rebuilt on next run.")
        return config

    def remove_calculation(self, calculation_id: str) -> None:
        if calculation_id in self.calculation_configs:
            del self.calculation_configs[calculation_id]
            self.invalidate("Calculation removed; system will restart on next run.")

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
        calc_module = CalculationModule(id=module_id, module=module, classes=classes)
        self.calculation_modules[module_id] = calc_module
        return calc_module

    # ------------------------------------------------------------------
    # Plot configuration
    # ------------------------------------------------------------------
    def set_plot_layout(self, rows: int, cols: int, lines: List[Dict[str, Any]]) -> PlotLayout:
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
        self.plot_layout = PlotLayout(rows=max(rows, 1), cols=max(cols, 1), lines=parsed_lines)
        return self.plot_layout

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, duration: Optional[Quantity] = None) -> Dict[str, Any]:
        system = self._ensure_system()
        if duration is None:
            system.step()
        else:
            system.step(duration)
        return self._collect_results(system)

    def _collect_results(self, system: System) -> Dict[str, Any]:
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
        figure = self._render_plot(system, outputs)
        return {
            "time": system.time,
            "outputs": outputs,
            "figure": figure,
            "messages": self.messages(),
        }

    def _render_plot(self, system: System, outputs: Dict[str, Any]) -> Optional[str]:
        if not self.plot_layout.lines:
            return self._render_default_plot(outputs)
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

        plt.tight_layout()
        return self._finalize_figure(fig)

    def _render_default_plot(self, outputs: Dict[str, Any]) -> Optional[str]:
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
            fields.append(
                {
                    "name": name,
                    "type": field_type,
                    "required": field.is_required(),
                    "default": default_value,
                    "description": field.description,
                }
            )
        return {
            "name": cls.__name__,
            "module": cls.__module__,
            "fields": fields,
            "doc": inspect.getdoc(cls) or "",
        }

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
