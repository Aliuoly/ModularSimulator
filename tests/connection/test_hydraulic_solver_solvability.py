from __future__ import annotations

import importlib
from types import MappingProxyType

import pytest

hydraulic_solver = importlib.import_module("modular_simulation.connection.hydraulic_solver")

HydraulicSystemDefinition = hydraulic_solver.HydraulicSystemDefinition
LinearResidualEquation = hydraulic_solver.LinearResidualEquation
solve_hydraulic_system = hydraulic_solver.solve_hydraulic_system


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


def test_missing_pressure_reference_in_connected_component_fails_fast() -> None:
    system = HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("head_balance", {"pressure_node": 1.0, "flow_branch": -1.0}),
            _linear_equation("mass_balance", {"flow_branch": 1.0}),
        ),
    )

    with pytest.raises(ValueError) as error:
        solve_hydraulic_system(system)

    message = str(error.value)
    assert "missing pressure reference" in message
    assert "pressure_node" in message


def test_disconnected_component_without_reference_reports_component_deterministically() -> None:
    system = HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("ref_a", {"pressure_a": 1.0}, constant=-5.0),
            _linear_equation("balance_a", {"pressure_a": 1.0, "flow_a": -1.0}),
            _linear_equation("balance_b", {"pressure_b": 1.0, "flow_b": -1.0}),
            _linear_equation("mass_b", {"flow_b": 1.0}),
        ),
    )

    with pytest.raises(ValueError) as error:
        solve_hydraulic_system(system)

    message = str(error.value)
    assert "missing pressure reference" in message
    assert "connected component 2" in message
    assert "pressure_b" in message


def test_disconnected_components_with_references_solve_deterministically() -> None:
    system = HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("ref_a", {"pressure_a": 1.0}, constant=-5.0),
            _linear_equation("balance_a", {"pressure_a": 1.0, "flow_a": -1.0}),
            _linear_equation("ref_b", {"pressure_b": 1.0}, constant=-7.0),
            _linear_equation("balance_b", {"pressure_b": 1.0, "flow_b": -1.0}),
        ),
    )

    result = solve_hydraulic_system(system, tolerance=1.0e-12, max_iterations=10)

    assert result.converged
    assert result.unknowns["pressure_a"] == pytest.approx(5.0)
    assert result.unknowns["flow_a"] == pytest.approx(5.0)
    assert result.unknowns["pressure_b"] == pytest.approx(7.0)
    assert result.unknowns["flow_b"] == pytest.approx(7.0)


def test_connected_component_dof_mismatch_is_detected_before_newton() -> None:
    system = HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("r1", {"u": 1.0}),
            _linear_equation("r2", {"u": -1.0}),
            _linear_equation("s1", {"v1": 1.0, "v2": 1.0}),
        ),
    )

    with pytest.raises(ValueError) as error:
        solve_hydraulic_system(system)

    message = str(error.value)
    assert "DOF mismatch" in message
    assert "connected component" in message


def test_singular_jacobian_fails_with_actionable_message() -> None:
    system = HydraulicSystemDefinition(
        equations=(),
        linear_residual_equations=(
            _linear_equation("eq_1", {"x": 1.0, "y": 1.0}),
            _linear_equation("eq_2", {"x": 2.0, "y": 2.0}),
        ),
    )

    with pytest.raises(ValueError) as error:
        solve_hydraulic_system(system)

    message = str(error.value)
    assert "Jacobian is singular or underdetermined" in message
    assert "rank" in message
