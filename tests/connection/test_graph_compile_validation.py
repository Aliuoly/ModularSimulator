from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from types import MappingProxyType

import pytest

from modular_simulation.connection.hydraulic_compile import (
    GraphCompileDiagnostic,
    GraphCompileError,
    HydraulicCompileLifecycle,
    compile_hydraulic_graph,
)
from modular_simulation.connection.hydraulic_solver import (
    HydraulicSystemDefinition,
    LinearResidualEquation,
)


def _linear_equation(
    residual_name: str,
    coefficients: dict[str, float],
    *,
    constant: float = 0.0,
) -> LinearResidualEquation:
    return LinearResidualEquation(
        residual_name=residual_name,
        coefficients=MappingProxyType(coefficients),
        constant=constant,
    )


def _valid_system() -> HydraulicSystemDefinition:
    return HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("ref_a", {"pressure_a": 1.0}, constant=-5.0),
            _linear_equation("balance_a", {"pressure_a": 1.0, "flow_a": -1.0}),
            _linear_equation("ref_b", {"pressure_b": 1.0}, constant=-7.0),
            _linear_equation("balance_b", {"pressure_b": 1.0, "flow_b": -1.0}),
        ),
    )


def test_compile_happy_path_validates_without_runtime_execution() -> None:
    compiled = compile_hydraulic_graph(system=_valid_system(), graph_revision="rev-1")

    assert compiled.graph_revision == "rev-1"
    assert compiled.unknown_order == ("flow_a", "flow_b", "pressure_a", "pressure_b")
    assert compiled.residual_order == ("balance_a", "balance_b", "ref_a", "ref_b")


def test_compile_missing_pressure_reference_has_deterministic_diagnostics() -> None:
    system = HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("head_balance", {"pressure_node": 1.0, "flow_branch": -1.0}),
            _linear_equation("mass_balance", {"flow_branch": 1.0}),
        ),
    )

    runs: list[tuple[GraphCompileDiagnostic, ...]] = []
    for _ in range(2):
        with pytest.raises(GraphCompileError) as error:
            compile_hydraulic_graph(system=system, graph_revision="rev-bad")
        runs.append(error.value.diagnostics)

    assert runs[0] == runs[1]
    assert runs[0][0].code == "missing_pressure_reference"
    assert "missing pressure reference" in runs[0][0].message
    assert "pressure_node" in runs[0][0].message


def test_compile_disconnected_invalid_topology_reports_component_stably() -> None:
    system = HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("ref_a", {"pressure_a": 1.0}, constant=-5.0),
            _linear_equation("balance_a", {"pressure_a": 1.0, "flow_a": -1.0}),
            _linear_equation("balance_b", {"pressure_b": 1.0, "flow_b": -1.0}),
            _linear_equation("mass_b", {"flow_b": 1.0}),
        ),
    )

    with pytest.raises(GraphCompileError) as error:
        compile_hydraulic_graph(system=system, graph_revision="rev-disconnected")

    diagnostic = error.value.diagnostics[0]
    assert diagnostic.code == "missing_pressure_reference"
    assert "connected component 2" in diagnostic.message
    assert "pressure_b" in diagnostic.message


def test_compile_invalid_cardinality_reports_assertion_friendly_payload() -> None:
    system = HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("r1", {"u": 1.0}),
            _linear_equation("r2", {"u": -1.0}),
        ),
    )

    with pytest.raises(GraphCompileError) as error:
        compile_hydraulic_graph(system=system, graph_revision="rev-cardinality")

    diagnostic = error.value.diagnostics[0]
    assert diagnostic.code == "invalid_cardinality"
    assert diagnostic.details == (("residual_count", "2"), ("unknown_count", "1"))


def test_candidate_compile_does_not_mutate_active_runtime_state() -> None:
    lifecycle = HydraulicCompileLifecycle()

    active_compiled = lifecycle.compile(system=_valid_system(), graph_revision="active-rev")
    lifecycle.activate(active_compiled)
    assert lifecycle.active_graph_revision == "active-rev"

    candidate_valid = lifecycle.compile_candidate(
        system=_valid_system(), graph_revision="candidate-rev"
    )
    assert candidate_valid.graph_revision == "candidate-rev"
    assert lifecycle.active_graph_revision == "active-rev"
    assert lifecycle.active_compiled == active_compiled

    invalid_candidate = HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("head_balance", {"pressure_candidate": 1.0, "flow_c": -1.0}),
            _linear_equation("mass_balance", {"flow_c": 1.0}),
        ),
    )
    with pytest.raises(GraphCompileError):
        lifecycle.compile_candidate(system=invalid_candidate, graph_revision="candidate-invalid")

    assert lifecycle.active_graph_revision == "active-rev"
    assert lifecycle.active_compiled == active_compiled
