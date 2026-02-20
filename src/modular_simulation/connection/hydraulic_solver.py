from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class _ElementParameterSpecLike(Protocol):
    names: tuple[str, ...]


class _ElementUnknownSpecLike(Protocol):
    names: tuple[str, ...]


class _ElementOutputSpecLike(Protocol):
    residual_names: tuple[str, ...]


class HydraulicElementLike(Protocol):
    parameter_spec: _ElementParameterSpecLike
    unknown_spec: _ElementUnknownSpecLike
    output_spec: _ElementOutputSpecLike

    def residuals(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> Mapping[str, float]: ...

    def jacobian(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> Mapping[tuple[str, str], float]: ...


@dataclass(frozen=True)
class ElementEquation:
    name: str
    element: HydraulicElementLike
    inputs: Mapping[str, float]
    parameters: Mapping[str, float]
    unknown_name_map: Mapping[str, str]
    residual_name_map: Mapping[str, str]

    def __post_init__(self) -> None:
        expected_unknowns = set(self.element.unknown_spec.names)
        actual_unknowns = set(self.unknown_name_map.keys())
        if actual_unknowns != expected_unknowns:
            raise ValueError(
                "equation {!r} unknown mapping keys must match {!r}, got {!r}".format(
                    self.name,
                    sorted(expected_unknowns),
                    sorted(actual_unknowns),
                )
            )

        expected_residuals = set(self.element.output_spec.residual_names)
        actual_residuals = set(self.residual_name_map.keys())
        if actual_residuals != expected_residuals:
            raise ValueError(
                "equation {!r} residual mapping keys must match {!r}, got {!r}".format(
                    self.name,
                    sorted(expected_residuals),
                    sorted(actual_residuals),
                )
            )


@dataclass(frozen=True)
class LinearResidualEquation:
    residual_name: str
    coefficients: Mapping[str, float]
    constant: float = 0.0


@dataclass(frozen=True)
class HydraulicSystemDefinition:
    equations: Sequence[ElementEquation]
    linear_residual_equations: Sequence[LinearResidualEquation] = ()
    unknown_order: Sequence[str] | None = None
    residual_order: Sequence[str] | None = None


@dataclass(frozen=True)
class SparseJacobianTriplet:
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    data: tuple[float, ...]
    shape: tuple[int, int]


@dataclass(frozen=True)
class HydraulicSolveResult:
    converged: bool
    iterations: int
    residual_norm: float
    solution_vector: NDArray[np.float64]
    unknowns: Mapping[str, float]
    unknown_index_map: Mapping[str, int]
    residual_index_map: Mapping[str, int]


def _validate_declared_order(
    *,
    declared: Sequence[str] | None,
    discovered: set[str],
    kind: str,
) -> tuple[str, ...]:
    if declared is None:
        return tuple(sorted(discovered))

    ordered = tuple(declared)
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"{kind} order contains duplicates")

    if set(ordered) != discovered:
        raise ValueError(
            f"{kind} order must match discovered names {sorted(discovered)!r}, got {sorted(ordered)!r}"
        )
    return ordered


def build_index_maps(
    system: HydraulicSystemDefinition,
) -> tuple[dict[str, int], dict[str, int]]:
    unknown_names: set[str] = set()
    residual_names: set[str] = set()

    for equation in system.equations:
        unknown_names.update(equation.unknown_name_map.values())
        residual_names.update(equation.residual_name_map.values())

    for equation in system.linear_residual_equations:
        unknown_names.update(equation.coefficients.keys())
        residual_names.add(equation.residual_name)

    unknown_order = _validate_declared_order(
        declared=system.unknown_order,
        discovered=unknown_names,
        kind="unknown",
    )
    residual_order = _validate_declared_order(
        declared=system.residual_order,
        discovered=residual_names,
        kind="residual",
    )

    unknown_index_map = {name: idx for idx, name in enumerate(unknown_order)}
    residual_index_map = {name: idx for idx, name in enumerate(residual_order)}
    return unknown_index_map, residual_index_map


def assemble_unknown_vector(
    system: HydraulicSystemDefinition,
    unknown_values: Mapping[str, float],
) -> NDArray[np.float64]:
    unknown_index_map, _ = build_index_maps(system)
    vector = np.zeros(len(unknown_index_map), dtype=float)
    for name, idx in unknown_index_map.items():
        if name in unknown_values:
            vector[idx] = float(unknown_values[name])
    return vector


def _unknown_vector_to_mapping(
    unknown_vector: NDArray[np.float64],
    unknown_index_map: Mapping[str, int],
) -> dict[str, float]:
    return {name: float(unknown_vector[idx]) for name, idx in unknown_index_map.items()}


def assemble_residual_vector(
    system: HydraulicSystemDefinition,
    unknown_vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict[str, int], dict[str, int]]:
    unknown_index_map, residual_index_map = build_index_maps(system)
    if unknown_vector.shape != (len(unknown_index_map),):
        raise ValueError(
            f"unknown_vector must have shape ({len(unknown_index_map)},), got {unknown_vector.shape}"
        )

    unknown_values = _unknown_vector_to_mapping(unknown_vector, unknown_index_map)
    residual_vector = np.zeros(len(residual_index_map), dtype=float)

    for equation in system.equations:
        local_unknowns = {
            local_name: unknown_values[global_name]
            for local_name, global_name in equation.unknown_name_map.items()
        }
        element_residuals = equation.element.residuals(
            inputs=equation.inputs,
            unknowns=local_unknowns,
            parameters=equation.parameters,
        )
        for local_name, global_name in equation.residual_name_map.items():
            residual_vector[residual_index_map[global_name]] += float(element_residuals[local_name])

    for equation in system.linear_residual_equations:
        value = float(equation.constant)
        for unknown_name, coefficient in equation.coefficients.items():
            value += float(coefficient) * unknown_values[unknown_name]
        residual_vector[residual_index_map[equation.residual_name]] += value

    return residual_vector, unknown_index_map, residual_index_map


def assemble_sparse_jacobian_triplet(
    system: HydraulicSystemDefinition,
    unknown_vector: NDArray[np.float64],
) -> tuple[SparseJacobianTriplet, dict[str, int], dict[str, int]]:
    unknown_index_map, residual_index_map = build_index_maps(system)
    if unknown_vector.shape != (len(unknown_index_map),):
        raise ValueError(
            f"unknown_vector must have shape ({len(unknown_index_map)},), got {unknown_vector.shape}"
        )

    unknown_values = _unknown_vector_to_mapping(unknown_vector, unknown_index_map)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for equation in system.equations:
        local_unknowns = {
            local_name: unknown_values[global_name]
            for local_name, global_name in equation.unknown_name_map.items()
        }
        element_jacobian = equation.element.jacobian(
            inputs=equation.inputs,
            unknowns=local_unknowns,
            parameters=equation.parameters,
        )
        for (local_residual, local_unknown), derivative in element_jacobian.items():
            global_residual = equation.residual_name_map[local_residual]
            global_unknown = equation.unknown_name_map[local_unknown]
            rows.append(residual_index_map[global_residual])
            cols.append(unknown_index_map[global_unknown])
            data.append(float(derivative))

    for equation in system.linear_residual_equations:
        row = residual_index_map[equation.residual_name]
        for unknown_name, coefficient in equation.coefficients.items():
            rows.append(row)
            cols.append(unknown_index_map[unknown_name])
            data.append(float(coefficient))

    triplet = SparseJacobianTriplet(
        rows=tuple(rows),
        cols=tuple(cols),
        data=tuple(data),
        shape=(len(residual_index_map), len(unknown_index_map)),
    )
    return triplet, unknown_index_map, residual_index_map


def triplet_to_dense(jacobian: SparseJacobianTriplet) -> NDArray[np.float64]:
    dense = np.zeros(jacobian.shape, dtype=float)
    for row, col, value in zip(jacobian.rows, jacobian.cols, jacobian.data, strict=True):
        dense[row, col] += value
    return dense


def _residual_norm(residual_vector: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(residual_vector, ord=2))


def _is_pressure_unknown_name(unknown_name: str) -> bool:
    return "pressure" in unknown_name.lower()


def _connected_components(
    system: HydraulicSystemDefinition,
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    adjacency: dict[str, set[str]] = {}

    def _add_node(node: str) -> None:
        _ = adjacency.setdefault(node, set())

    def _connect(lhs: str, rhs: str) -> None:
        _add_node(lhs)
        _add_node(rhs)
        adjacency[lhs].add(rhs)
        adjacency[rhs].add(lhs)

    for equation in system.equations:
        global_unknowns = tuple(sorted(set(equation.unknown_name_map.values())))
        global_residuals = tuple(sorted(set(equation.residual_name_map.values())))
        for unknown in global_unknowns:
            _add_node(f"u:{unknown}")
        for residual in global_residuals:
            _add_node(f"r:{residual}")
        for unknown in global_unknowns:
            for residual in global_residuals:
                _connect(f"u:{unknown}", f"r:{residual}")

    for equation in system.linear_residual_equations:
        residual_node = f"r:{equation.residual_name}"
        _add_node(residual_node)
        for unknown in sorted(set(equation.coefficients.keys())):
            _connect(f"u:{unknown}", residual_node)

    visited: set[str] = set()
    components: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for start in sorted(adjacency.keys()):
        if start in visited:
            continue
        stack = [start]
        component_nodes: set[str] = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component_nodes.add(node)
            stack.extend(sorted(adjacency[node] - visited))

        component_unknowns = tuple(
            sorted(node[2:] for node in component_nodes if node.startswith("u:"))
        )
        component_residuals = tuple(
            sorted(node[2:] for node in component_nodes if node.startswith("r:"))
        )
        components.append((component_unknowns, component_residuals))

    components.sort(key=lambda component: (component[0], component[1]))
    return tuple(components)


def _pressure_reference_unknowns(system: HydraulicSystemDefinition) -> set[str]:
    references: set[str] = set()
    for equation in system.linear_residual_equations:
        if len(equation.coefficients) != 1:
            continue
        unknown_name, coefficient = next(iter(equation.coefficients.items()))
        if float(coefficient) == 0.0:
            continue
        if _is_pressure_unknown_name(unknown_name):
            references.add(unknown_name)
    return references


def _validate_solvability(
    system: HydraulicSystemDefinition,
    *,
    unknown_index_map: Mapping[str, int],
    residual_index_map: Mapping[str, int],
) -> None:
    del unknown_index_map, residual_index_map

    pressure_references = _pressure_reference_unknowns(system)
    components = _connected_components(system)
    for component_number, (component_unknowns, component_residuals) in enumerate(
        components, start=1
    ):
        if len(component_unknowns) != len(component_residuals):
            raise ValueError(
                "hydraulic DOF mismatch in connected component {}: {} unknowns vs {} residuals; make each connected component square with independent equations".format(
                    component_number,
                    len(component_unknowns),
                    len(component_residuals),
                )
            )

        pressure_unknowns = tuple(
            name for name in component_unknowns if _is_pressure_unknown_name(name)
        )
        if not pressure_unknowns:
            continue

        if not any(name in pressure_references for name in pressure_unknowns):
            raise ValueError(
                "missing pressure reference in connected component {}; pressure unknowns={!r}. Add a linear reference equation like pressure_node = constant.".format(
                    component_number,
                    list(pressure_unknowns),
                )
            )


def _validate_jacobian_rank(
    jacobian_dense: NDArray[np.float64],
) -> None:
    rank = int(np.linalg.matrix_rank(jacobian_dense))
    expected_rank = int(jacobian_dense.shape[0])
    if rank < expected_rank:
        raise ValueError(
            "hydraulic Jacobian is singular or underdetermined (rank {} < {}); check connected-component DOF and pressure references".format(
                rank,
                expected_rank,
            )
        )


def solve_hydraulic_system(
    system: HydraulicSystemDefinition,
    *,
    initial_unknowns: Mapping[str, float] | None = None,
    warm_start: NDArray[np.float64] | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 25,
    min_damping: float = 1.0e-6,
) -> HydraulicSolveResult:
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if min_damping <= 0.0 or min_damping > 1.0:
        raise ValueError("min_damping must be in (0, 1]")

    unknown_index_map, residual_index_map = build_index_maps(system)
    if len(unknown_index_map) != len(residual_index_map):
        raise ValueError("hydraulic system must be square for Newton solve")
    if not unknown_index_map:
        raise ValueError("hydraulic system has no unknowns")
    _validate_solvability(
        system,
        unknown_index_map=unknown_index_map,
        residual_index_map=residual_index_map,
    )

    if warm_start is not None:
        x = np.asarray(warm_start, dtype=float).copy()
        if x.shape != (len(unknown_index_map),):
            raise ValueError(
                f"warm_start must have shape ({len(unknown_index_map)},), got {x.shape}"
            )
    else:
        x = np.zeros(len(unknown_index_map), dtype=float)
        if initial_unknowns is not None:
            for name, value in initial_unknowns.items():
                if name not in unknown_index_map:
                    raise KeyError(f"unknown {name!r} not present in system")
                x[unknown_index_map[name]] = float(value)

    residual_vector, _, _ = assemble_residual_vector(system, x)
    residual_norm = _residual_norm(residual_vector)

    initial_triplet, _, _ = assemble_sparse_jacobian_triplet(system, x)
    initial_jacobian_dense = triplet_to_dense(initial_triplet)
    _validate_jacobian_rank(initial_jacobian_dense)

    if residual_norm <= tolerance:
        unknowns = _unknown_vector_to_mapping(x, unknown_index_map)
        return HydraulicSolveResult(
            converged=True,
            iterations=0,
            residual_norm=residual_norm,
            solution_vector=np.asarray(x, dtype=np.float64).copy(),
            unknowns=unknowns,
            unknown_index_map=unknown_index_map,
            residual_index_map=residual_index_map,
        )

    for iteration in range(1, max_iterations + 1):
        if iteration == 1:
            jacobian_dense = initial_jacobian_dense
        else:
            triplet, _, _ = assemble_sparse_jacobian_triplet(system, x)
            jacobian_dense = triplet_to_dense(triplet)
            _validate_jacobian_rank(jacobian_dense)
        rhs = -residual_vector

        try:
            newton_step = np.asarray(np.linalg.solve(jacobian_dense, rhs), dtype=np.float64)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "hydraulic Jacobian is singular while computing Newton step; check component pressure references and independent residual equations"
            ) from exc

        accepted = False
        damping = 1.0
        while damping >= min_damping:
            candidate = np.asarray(x + damping * newton_step, dtype=np.float64)
            candidate_residual, _, _ = assemble_residual_vector(system, candidate)
            candidate_norm = _residual_norm(candidate_residual)
            if candidate_norm < residual_norm:
                x = candidate
                residual_vector = candidate_residual
                residual_norm = candidate_norm
                accepted = True
                break
            damping *= 0.5

        if not accepted:
            x = np.asarray(x + min_damping * newton_step, dtype=np.float64)
            residual_vector, _, _ = assemble_residual_vector(system, x)
            residual_norm = _residual_norm(residual_vector)

        if residual_norm <= tolerance:
            unknowns = _unknown_vector_to_mapping(x, unknown_index_map)
            return HydraulicSolveResult(
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
                solution_vector=np.asarray(x, dtype=np.float64).copy(),
                unknowns=unknowns,
                unknown_index_map=unknown_index_map,
                residual_index_map=residual_index_map,
            )

    unknowns = _unknown_vector_to_mapping(x, unknown_index_map)
    return HydraulicSolveResult(
        converged=False,
        iterations=max_iterations,
        residual_norm=residual_norm,
        solution_vector=np.asarray(x, dtype=np.float64).copy(),
        unknowns=unknowns,
        unknown_index_map=unknown_index_map,
        residual_index_map=residual_index_map,
    )


__all__ = [
    "ElementEquation",
    "HydraulicSolveResult",
    "HydraulicSystemDefinition",
    "LinearResidualEquation",
    "SparseJacobianTriplet",
    "assemble_residual_vector",
    "assemble_sparse_jacobian_triplet",
    "assemble_unknown_vector",
    "build_index_maps",
    "solve_hydraulic_system",
    "triplet_to_dense",
]
