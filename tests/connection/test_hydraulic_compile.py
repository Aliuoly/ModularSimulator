from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false

from types import MappingProxyType

import pytest

from modular_simulation.connection.hydraulic_compile import (
    CompiledHydraulicGraph,
    build_hydraulic_solver_inputs,
    compile_hydraulic_graph,
    solve_compiled_hydraulic_graph,
)
from modular_simulation.connection.hydraulic_solver import (
    HydraulicSystemDefinition,
    LinearResidualEquation,
    build_index_maps,
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


def _linear_topology_system() -> HydraulicSystemDefinition:
    return HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("ref_upstream", {"pressure_upstream": 1.0}, constant=-120_000.0),
            _linear_equation(
                "head_balance",
                {"pressure_upstream": 1.0, "pressure_downstream": -1.0, "flow_linear": -1.0},
            ),
            _linear_equation("flow_target", {"flow_linear": 1.0}, constant=-50.0),
        ),
    )


def _branch_topology_system() -> HydraulicSystemDefinition:
    return HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("ref_mix", {"pressure_mix": 1.0}, constant=-110_000.0),
            _linear_equation(
                "head_feed_a",
                {"pressure_mix": 1.0, "flow_feed_a": -1.0},
                constant=-100_000.0,
            ),
            _linear_equation(
                "head_feed_b",
                {"pressure_mix": 1.0, "flow_feed_b": -1.0},
                constant=-95_000.0,
            ),
            _linear_equation(
                "mass_balance",
                {"flow_feed_a": 1.0, "flow_feed_b": 1.0, "flow_outlet": -1.0},
            ),
        ),
    )


def _recycle_topology_system() -> HydraulicSystemDefinition:
    return HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("ref_a", {"pressure_a": 1.0}, constant=-150_000.0),
            _linear_equation(
                "head_recycle",
                {"pressure_a": 1.0, "pressure_b": -1.0, "flow_recycle": -1.0},
            ),
            _linear_equation("recycle_target", {"flow_recycle": 1.0}, constant=-25.0),
        ),
    )


def _disconnected_missing_reference_system() -> HydraulicSystemDefinition:
    return HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("ref_a", {"pressure_a": 1.0}, constant=-5.0),
            _linear_equation("balance_a", {"pressure_a": 1.0, "flow_a": -1.0}),
            _linear_equation("balance_b", {"pressure_b": 1.0, "flow_b": -1.0}),
            _linear_equation("mass_b", {"flow_b": 1.0}),
        ),
    )


@pytest.mark.parametrize(
    ("system_factory", "expected_unknowns", "expected_residuals"),
    (
        (
            _linear_topology_system,
            ("flow_linear", "pressure_downstream", "pressure_upstream"),
            ("flow_target", "head_balance", "ref_upstream"),
        ),
        (
            _branch_topology_system,
            ("flow_feed_a", "flow_feed_b", "flow_outlet", "pressure_mix"),
            ("head_feed_a", "head_feed_b", "mass_balance", "ref_mix"),
        ),
        (
            _recycle_topology_system,
            ("flow_recycle", "pressure_a", "pressure_b"),
            ("head_recycle", "recycle_target", "ref_a"),
        ),
    ),
)
def test_compile_bridge_produces_solver_inputs_with_deterministic_orders(
    system_factory,
    expected_unknowns: tuple[str, ...],
    expected_residuals: tuple[str, ...],
) -> None:
    compiled = compile_hydraulic_graph(system=system_factory(), graph_revision="rev-bridge")

    bridge_inputs = build_hydraulic_solver_inputs(compiled)

    assert bridge_inputs.graph_revision == "rev-bridge"
    assert bridge_inputs.unknown_order == expected_unknowns
    assert bridge_inputs.residual_order == expected_residuals
    assert tuple(bridge_inputs.unknown_index_map) == expected_unknowns
    assert tuple(bridge_inputs.residual_index_map) == expected_residuals


def test_compile_bridge_solution_preserves_declared_unknown_residual_index_order() -> None:
    compiled = compile_hydraulic_graph(system=_branch_topology_system(), graph_revision="rev-order")

    bridge_inputs = build_hydraulic_solver_inputs(compiled)
    unknown_index_map, residual_index_map = build_index_maps(bridge_inputs.system)

    assert tuple(sorted(unknown_index_map, key=unknown_index_map.__getitem__)) == (
        "flow_feed_a",
        "flow_feed_b",
        "flow_outlet",
        "pressure_mix",
    )
    assert tuple(sorted(residual_index_map, key=residual_index_map.__getitem__)) == (
        "head_feed_a",
        "head_feed_b",
        "mass_balance",
        "ref_mix",
    )


def test_compile_bridge_solve_runs_against_existing_solver_contract() -> None:
    compiled = compile_hydraulic_graph(system=_linear_topology_system(), graph_revision="rev-solve")

    solved = solve_compiled_hydraulic_graph(compiled, tolerance=1.0e-12, max_iterations=10)

    assert solved.converged
    assert solved.unknowns["pressure_upstream"] == pytest.approx(120_000.0)
    assert solved.unknowns["flow_linear"] == pytest.approx(50.0)
    assert solved.unknowns["pressure_downstream"] == pytest.approx(119_950.0)


def test_compile_bridge_preserves_component_specific_pressure_reference_diagnostics() -> None:
    system = _disconnected_missing_reference_system()
    unknown_index_map, residual_index_map = build_index_maps(system)
    compiled = CompiledHydraulicGraph(
        graph_revision="rev-diagnostic",
        system=system,
        unknown_order=tuple(sorted(unknown_index_map, key=unknown_index_map.__getitem__)),
        residual_order=tuple(sorted(residual_index_map, key=residual_index_map.__getitem__)),
    )

    messages: list[str] = []
    for _ in range(2):
        with pytest.raises(ValueError) as error:
            solve_compiled_hydraulic_graph(compiled)
        messages.append(str(error.value))

    assert messages[0] == messages[1]
    assert "missing pressure reference" in messages[0]
    assert "connected component 2" in messages[0]
    assert "pressure_b" in messages[0]
