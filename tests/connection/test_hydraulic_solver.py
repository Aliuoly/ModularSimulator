from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pytest

hydraulic_element = importlib.import_module("modular_simulation.connection.hydraulic_element")
hydraulic_solver = importlib.import_module("modular_simulation.connection.hydraulic_solver")

ElementOutputSpec = hydraulic_element.ElementOutputSpec
ElementParameterSpec = hydraulic_element.ElementParameterSpec
ElementUnknownSpec = hydraulic_element.ElementUnknownSpec

ElementEquation = hydraulic_solver.ElementEquation
HydraulicSystemDefinition = hydraulic_solver.HydraulicSystemDefinition
SparseJacobianTriplet = hydraulic_solver.SparseJacobianTriplet
assemble_sparse_jacobian_triplet = hydraulic_solver.assemble_sparse_jacobian_triplet
assemble_unknown_vector = hydraulic_solver.assemble_unknown_vector
solve_hydraulic_system = hydraulic_solver.solve_hydraulic_system
triplet_to_dense = hydraulic_solver.triplet_to_dense


@dataclass(frozen=True)
class _AffineResidualElement:
    coefficient_a: float
    coefficient_b: float

    parameter_spec: ElementParameterSpec = ElementParameterSpec(names=("offset",))
    unknown_spec: ElementUnknownSpec = ElementUnknownSpec(names=("u_a", "u_b"))
    output_spec: ElementOutputSpec = ElementOutputSpec(residual_names=("residual",))

    def residuals(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> Mapping[str, float]:
        del inputs
        value = (
            self.coefficient_a * unknowns["u_a"]
            + self.coefficient_b * unknowns["u_b"]
            + parameters["offset"]
        )
        return {"residual": value}

    def jacobian(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> Mapping[tuple[str, str], float]:
        del inputs, unknowns, parameters
        return {
            ("residual", "u_a"): self.coefficient_a,
            ("residual", "u_b"): self.coefficient_b,
        }


@dataclass(frozen=True)
class _JacobianOnlyElement:
    parameter_spec: ElementParameterSpec = ElementParameterSpec(names=())
    unknown_spec: ElementUnknownSpec = ElementUnknownSpec(names=("u",))
    output_spec: ElementOutputSpec = ElementOutputSpec(residual_names=("r",))

    def residuals(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> Mapping[str, float]:
        raise AssertionError("jacobian assembly must not evaluate residuals")

    def jacobian(
        self,
        *,
        inputs: Mapping[str, float],
        unknowns: Mapping[str, float],
        parameters: Mapping[str, float],
    ) -> Mapping[tuple[str, str], float]:
        del inputs, unknowns, parameters
        return {("r", "u"): 7.5}


def _build_affine_system() -> HydraulicSystemDefinition:
    eq_1 = ElementEquation(
        name="eq_1",
        element=_AffineResidualElement(coefficient_a=1.0, coefficient_b=1.0),
        inputs=MappingProxyType({}),
        parameters=MappingProxyType({"offset": -3.0}),
        unknown_name_map=MappingProxyType({"u_a": "flow_a", "u_b": "flow_b"}),
        residual_name_map=MappingProxyType({"residual": "mass_balance"}),
    )
    eq_2 = ElementEquation(
        name="eq_2",
        element=_AffineResidualElement(coefficient_a=2.0, coefficient_b=-1.0),
        inputs=MappingProxyType({}),
        parameters=MappingProxyType({"offset": 0.0}),
        unknown_name_map=MappingProxyType({"u_a": "flow_a", "u_b": "flow_b"}),
        residual_name_map=MappingProxyType({"residual": "head_balance"}),
    )
    return HydraulicSystemDefinition(equations=(eq_1, eq_2))


def test_solver_converges_on_small_deterministic_network() -> None:
    system = _build_affine_system()

    result = solve_hydraulic_system(
        system,
        initial_unknowns=MappingProxyType({"flow_a": 0.0, "flow_b": 0.0}),
        tolerance=1.0e-12,
        max_iterations=10,
    )

    assert result.converged
    assert result.unknown_index_map == {"flow_a": 0, "flow_b": 1}
    assert result.residual_index_map == {"head_balance": 0, "mass_balance": 1}
    assert result.unknowns["flow_a"] == pytest.approx(1.0)
    assert result.unknowns["flow_b"] == pytest.approx(2.0)


def test_warm_start_does_not_increase_iteration_count() -> None:
    system = _build_affine_system()

    cold_start = solve_hydraulic_system(
        system,
        initial_unknowns=MappingProxyType({"flow_a": 0.0, "flow_b": 0.0}),
        tolerance=1.0e-12,
        max_iterations=10,
    )
    warm_start = solve_hydraulic_system(
        system,
        warm_start=cold_start.solution_vector,
        tolerance=1.0e-12,
        max_iterations=10,
    )

    assert cold_start.converged
    assert warm_start.converged
    assert warm_start.iterations <= cold_start.iterations


def test_sparse_jacobian_assembly_uses_element_derivative_entries() -> None:
    element_equation = ElementEquation(
        name="jacobian_only",
        element=_JacobianOnlyElement(),
        inputs=MappingProxyType({}),
        parameters=MappingProxyType({}),
        unknown_name_map=MappingProxyType({"u": "flow_rate"}),
        residual_name_map=MappingProxyType({"r": "head_balance"}),
    )
    system = HydraulicSystemDefinition(
        equations=(element_equation,),
        unknown_order=("flow_rate",),
        residual_order=("head_balance",),
    )
    unknown_vector = assemble_unknown_vector(system, MappingProxyType({"flow_rate": 12.0}))

    triplet, unknown_index_map, residual_index_map = assemble_sparse_jacobian_triplet(
        system,
        unknown_vector,
    )

    assert isinstance(triplet, SparseJacobianTriplet)
    assert unknown_index_map == {"flow_rate": 0}
    assert residual_index_map == {"head_balance": 0}
    assert triplet.shape == (1, 1)
    assert triplet.rows == (0,)
    assert triplet.cols == (0,)
    assert triplet.data == pytest.approx((7.5,))
    assert triplet_to_dense(triplet)[0, 0] == pytest.approx(7.5)
