from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

import modular_simulation.connection.hydraulic_solver as hydraulic_solver
from modular_simulation.connection.hydraulic_solver import (
    HydraulicSolveResult,
    HydraulicSystemDefinition,
    build_index_maps,
    solve_hydraulic_system,
)


@dataclass(frozen=True)
class GraphCompileDiagnostic:
    code: str
    message: str
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        ordered_details = tuple(sorted((str(key), str(value)) for key, value in self.details))
        object.__setattr__(self, "details", ordered_details)


class GraphCompileError(ValueError):
    diagnostics: tuple[GraphCompileDiagnostic, ...]

    def __init__(self, diagnostics: tuple[GraphCompileDiagnostic, ...]):
        if not diagnostics:
            raise ValueError("GraphCompileError requires at least one diagnostic")
        self.diagnostics = tuple(
            sorted(diagnostics, key=lambda item: (item.code, item.message, item.details))
        )
        super().__init__(_format_diagnostic_message(self.diagnostics))


@dataclass(frozen=True)
class CompiledHydraulicGraph:
    graph_revision: str
    system: HydraulicSystemDefinition
    unknown_order: tuple[str, ...]
    residual_order: tuple[str, ...]


@dataclass(frozen=True)
class HydraulicSolverInputs:
    graph_revision: str
    system: HydraulicSystemDefinition
    unknown_order: tuple[str, ...]
    residual_order: tuple[str, ...]
    unknown_index_map: Mapping[str, int]
    residual_index_map: Mapping[str, int]


class _SolvabilityValidator(Protocol):
    def __call__(
        self,
        system: HydraulicSystemDefinition,
        *,
        unknown_index_map: Mapping[str, int],
        residual_index_map: Mapping[str, int],
    ) -> None: ...


class HydraulicCompileLifecycle:
    def __init__(self, active_compiled: CompiledHydraulicGraph | None = None):
        self._active_compiled: CompiledHydraulicGraph | None
        self._active_compiled = active_compiled

    @property
    def active_compiled(self) -> CompiledHydraulicGraph | None:
        return self._active_compiled

    @property
    def active_graph_revision(self) -> str | None:
        if self._active_compiled is None:
            return None
        return self._active_compiled.graph_revision

    def compile(
        self,
        *,
        system: HydraulicSystemDefinition,
        graph_revision: str,
    ) -> CompiledHydraulicGraph:
        return compile_hydraulic_graph(system=system, graph_revision=graph_revision)

    def compile_candidate(
        self,
        *,
        system: HydraulicSystemDefinition,
        graph_revision: str,
    ) -> CompiledHydraulicGraph:
        return compile_hydraulic_graph(system=system, graph_revision=graph_revision)

    def activate(self, compiled: CompiledHydraulicGraph) -> None:
        self._active_compiled = compiled


def validate_hydraulic_graph(
    *,
    system: HydraulicSystemDefinition,
) -> tuple[GraphCompileDiagnostic, ...]:
    try:
        unknown_index_map, residual_index_map = build_index_maps(system)
    except ValueError as error:
        return (
            GraphCompileDiagnostic(
                code="index_definition_invalid",
                message=str(error),
            ),
        )

    unknown_count = len(unknown_index_map)
    residual_count = len(residual_index_map)
    if unknown_count != residual_count:
        return (
            GraphCompileDiagnostic(
                code="invalid_cardinality",
                message="hydraulic system must be square for compile validation",
                details=(
                    ("unknown_count", str(unknown_count)),
                    ("residual_count", str(residual_count)),
                ),
            ),
        )

    if unknown_count == 0:
        return (
            GraphCompileDiagnostic(
                code="invalid_topology",
                message="hydraulic system has no unknowns",
            ),
        )

    solvability_validator = cast(
        _SolvabilityValidator,
        getattr(hydraulic_solver, "_validate_solvability"),
    )
    try:
        solvability_validator(
            system,
            unknown_index_map=unknown_index_map,
            residual_index_map=residual_index_map,
        )
    except ValueError as error:
        return (_solvability_error_to_diagnostic(message=str(error)),)

    return ()


def compile_hydraulic_graph(
    *,
    system: HydraulicSystemDefinition,
    graph_revision: str,
) -> CompiledHydraulicGraph:
    diagnostics = validate_hydraulic_graph(system=system)
    if diagnostics:
        raise GraphCompileError(diagnostics)

    unknown_index_map, residual_index_map = build_index_maps(system)
    unknown_order = tuple(sorted(unknown_index_map, key=unknown_index_map.__getitem__))
    residual_order = tuple(sorted(residual_index_map, key=residual_index_map.__getitem__))
    return CompiledHydraulicGraph(
        graph_revision=graph_revision,
        system=system,
        unknown_order=unknown_order,
        residual_order=residual_order,
    )


def build_hydraulic_solver_inputs(compiled: CompiledHydraulicGraph) -> HydraulicSolverInputs:
    system = HydraulicSystemDefinition(
        equations=compiled.system.equations,
        linear_residual_equations=compiled.system.linear_residual_equations,
        unknown_order=compiled.unknown_order,
        residual_order=compiled.residual_order,
    )
    unknown_index_map, residual_index_map = build_index_maps(system)
    unknown_order = tuple(sorted(unknown_index_map, key=unknown_index_map.__getitem__))
    residual_order = tuple(sorted(residual_index_map, key=residual_index_map.__getitem__))
    return HydraulicSolverInputs(
        graph_revision=compiled.graph_revision,
        system=system,
        unknown_order=unknown_order,
        residual_order=residual_order,
        unknown_index_map=unknown_index_map,
        residual_index_map=residual_index_map,
    )


def solve_compiled_hydraulic_graph(
    compiled: CompiledHydraulicGraph,
    *,
    initial_unknowns: Mapping[str, float] | None = None,
    warm_start: NDArray[np.float64] | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 25,
    min_damping: float = 1.0e-6,
) -> HydraulicSolveResult:
    solver_inputs = build_hydraulic_solver_inputs(compiled)
    return solve_hydraulic_system(
        solver_inputs.system,
        initial_unknowns=initial_unknowns,
        warm_start=warm_start,
        tolerance=tolerance,
        max_iterations=max_iterations,
        min_damping=min_damping,
    )


def _solvability_error_to_diagnostic(*, message: str) -> GraphCompileDiagnostic:
    normalized = message.lower()
    if "missing pressure reference" in normalized:
        code = "missing_pressure_reference"
    elif "dof mismatch" in normalized:
        code = "invalid_topology"
    else:
        code = "invalid_topology"
    return GraphCompileDiagnostic(code=code, message=message)


def _format_diagnostic_message(diagnostics: tuple[GraphCompileDiagnostic, ...]) -> str:
    formatted: list[str] = []
    for diagnostic in diagnostics:
        if diagnostic.details:
            detail_payload = ", ".join(f"{key}={value}" for key, value in diagnostic.details)
            formatted.append(f"{diagnostic.code}: {diagnostic.message} ({detail_payload})")
        else:
            formatted.append(f"{diagnostic.code}: {diagnostic.message}")
    return "graph compile validation failed: " + " | ".join(formatted)


__all__ = [
    "CompiledHydraulicGraph",
    "GraphCompileDiagnostic",
    "GraphCompileError",
    "HydraulicSolverInputs",
    "HydraulicCompileLifecycle",
    "build_hydraulic_solver_inputs",
    "compile_hydraulic_graph",
    "solve_compiled_hydraulic_graph",
    "validate_hydraulic_graph",
]
