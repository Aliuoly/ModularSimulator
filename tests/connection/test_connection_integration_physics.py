from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportMissingParameterType=false

import importlib
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import pytest

hydraulic_element = importlib.import_module("modular_simulation.connection.hydraulic_element")
hydraulic_compile = importlib.import_module("modular_simulation.connection.hydraulic_compile")
hydraulic_solver = importlib.import_module("modular_simulation.connection.hydraulic_solver")
junction = importlib.import_module("modular_simulation.connection.junction")
network = importlib.import_module("modular_simulation.connection.network")
state = importlib.import_module("modular_simulation.connection.state")
transport = importlib.import_module("modular_simulation.connection.transport")

ElementEquation = hydraulic_solver.ElementEquation
ElementOutputSpec = hydraulic_element.ElementOutputSpec
ElementParameterSpec = hydraulic_element.ElementParameterSpec
ElementUnknownSpec = hydraulic_element.ElementUnknownSpec
HydraulicSystemDefinition = hydraulic_solver.HydraulicSystemDefinition
HydraulicCompileLifecycle = hydraulic_compile.HydraulicCompileLifecycle
LagTransportState = transport.LagTransportState
LinearResidualEquation = hydraulic_solver.LinearResidualEquation
MaterialState = state.MaterialState
PipeHydraulicElement = hydraulic_element.PipeHydraulicElement
PortCondition = state.PortCondition
ConnectionNetwork = network.ConnectionNetwork
assemble_residual_vector = hydraulic_solver.assemble_residual_vector
smooth_signed_quadratic_flow_term = hydraulic_element.smooth_signed_quadratic_flow_term
smooth_signed_quadratic_flow_term_derivative = (
    hydraulic_element.smooth_signed_quadratic_flow_term_derivative
)
solve_compiled_hydraulic_graph = hydraulic_compile.solve_compiled_hydraulic_graph
update_lag_transport_state = transport.update_lag_transport_state
mix_junction_state = junction.mix_junction_state


def _single_pipe_system(
    *, upstream_pressure: float, downstream_pressure: float
) -> HydraulicSystemDefinition:
    equation = ElementEquation(
        name="pipe_head",
        element=PipeHydraulicElement(),
        inputs=MappingProxyType(
            {"upstream_pressure": upstream_pressure, "downstream_pressure": downstream_pressure}
        ),
        parameters=MappingProxyType({"pipe_resistance": 2.0e3, "delta": 1.0e-3}),
        unknown_name_map=MappingProxyType({"flow_rate": "edge_flow"}),
        residual_name_map=MappingProxyType({"head_balance": "head_balance"}),
    )
    return HydraulicSystemDefinition(
        equations=(equation,),
        linear_residual_equations=(
            LinearResidualEquation(
                residual_name="ref_upstream",
                coefficients=MappingProxyType({"pressure_upstream": 1.0}),
                constant=-upstream_pressure,
            ),
            LinearResidualEquation(
                residual_name="ref_downstream",
                coefficients=MappingProxyType({"pressure_downstream": 1.0}),
                constant=-downstream_pressure,
            ),
        ),
    )


@dataclass(frozen=True)
class _BoundaryNodeElement:
    boundary_value: float
    node_coefficient: float

    parameter_spec: ElementParameterSpec = ElementParameterSpec(names=("resistance", "delta"))
    unknown_spec: ElementUnknownSpec = ElementUnknownSpec(names=("flow_rate", "node_head"))
    output_spec: ElementOutputSpec = ElementOutputSpec(residual_names=("head_balance",))

    def residuals(self, *, inputs, unknowns, parameters):
        del inputs
        flow_rate = unknowns["flow_rate"]
        node_head = unknowns["node_head"]
        resistance = parameters["resistance"]
        delta = parameters["delta"]
        return {
            "head_balance": self.boundary_value
            + self.node_coefficient * node_head
            - resistance * smooth_signed_quadratic_flow_term(m_dot=flow_rate, delta=delta)
        }

    def jacobian(self, *, inputs, unknowns, parameters):
        del inputs
        flow_rate = unknowns["flow_rate"]
        resistance = parameters["resistance"]
        delta = parameters["delta"]
        return {
            (
                "head_balance",
                "flow_rate",
            ): -resistance
            * smooth_signed_quadratic_flow_term_derivative(m_dot=flow_rate, delta=delta),
            ("head_balance", "node_head"): self.node_coefficient,
        }


def _build_three_branch_topology_system() -> HydraulicSystemDefinition:
    feed_a = ElementEquation(
        name="feed_a_head",
        element=_BoundaryNodeElement(boundary_value=135000.0, node_coefficient=-1.0),
        inputs=MappingProxyType({}),
        parameters=MappingProxyType({"resistance": 1.6e3, "delta": 1.0e-3}),
        unknown_name_map=MappingProxyType({"flow_rate": "flow_feed_a", "node_head": "mix_head"}),
        residual_name_map=MappingProxyType({"head_balance": "head_feed_a"}),
    )
    feed_b = ElementEquation(
        name="feed_b_head",
        element=_BoundaryNodeElement(boundary_value=128000.0, node_coefficient=-1.0),
        inputs=MappingProxyType({}),
        parameters=MappingProxyType({"resistance": 2.1e3, "delta": 1.0e-3}),
        unknown_name_map=MappingProxyType({"flow_rate": "flow_feed_b", "node_head": "mix_head"}),
        residual_name_map=MappingProxyType({"head_balance": "head_feed_b"}),
    )
    outlet = ElementEquation(
        name="outlet_head",
        element=_BoundaryNodeElement(boundary_value=-100000.0, node_coefficient=1.0),
        inputs=MappingProxyType({}),
        parameters=MappingProxyType({"resistance": 1.3e3, "delta": 1.0e-3}),
        unknown_name_map=MappingProxyType({"flow_rate": "flow_outlet", "node_head": "mix_head"}),
        residual_name_map=MappingProxyType({"head_balance": "head_outlet"}),
    )
    return HydraulicSystemDefinition(
        equations=(feed_a, feed_b, outlet),
        linear_residual_equations=(
            LinearResidualEquation(
                residual_name="mass_closure",
                coefficients=MappingProxyType(
                    {
                        "flow_feed_a": 1.0,
                        "flow_feed_b": 1.0,
                        "flow_outlet": -1.0,
                    }
                ),
                constant=0.0,
            ),
        ),
    )


def _compile_single_pipe_network(system: HydraulicSystemDefinition):
    connection_network = ConnectionNetwork(
        compile_lifecycle=HydraulicCompileLifecycle(),
        hydraulic_system_builder=lambda topology: system,
    )
    connection_network.add_process("pipe", inlet_ports=("inlet",), outlet_ports=("outlet",))
    connection_network.add_boundary_source("upstream")
    connection_network.add_boundary_sink("downstream")
    connection_network.connect("upstream", "pipe.inlet")
    connection_network.connect("pipe.outlet", "downstream")
    compiled = connection_network.compile()
    assert compiled.hydraulic is not None
    return compiled.hydraulic


def _compile_three_branch_network(system: HydraulicSystemDefinition):
    connection_network = ConnectionNetwork(
        compile_lifecycle=HydraulicCompileLifecycle(),
        hydraulic_system_builder=lambda topology: system,
    )
    connection_network.add_process(
        "mixer",
        inlet_ports=("feed_a", "feed_b"),
        outlet_ports=("product", "purge"),
    )
    connection_network.add_boundary_source("feed_a_boundary")
    connection_network.add_boundary_source("feed_b_boundary")
    connection_network.add_boundary_sink("product_boundary")
    connection_network.connect("feed_a_boundary", "mixer.feed_a")
    connection_network.connect("feed_b_boundary", "mixer.feed_b")
    connection_network.connect("mixer.product", "product_boundary")
    compiled = connection_network.compile()
    assert compiled.hydraulic is not None
    return compiled


def _run_reversal_sequence() -> tuple[tuple[float, float, float, float, float, bool], ...]:
    state_value = LagTransportState(composition=(0.5, 0.5), temperature=320.0)
    forward_advected = LagTransportState(composition=(0.92, 0.08), temperature=390.0)
    reverse_advected = LagTransportState(composition=(0.08, 0.92), temperature=260.0)

    previous_flow: float | None = None
    warm_start = None
    trace: list[tuple[float, float, float, float, float, bool]] = []
    for step in range(24):
        upstream_pressure = 120000.0 if step % 2 == 0 else 100000.0
        downstream_pressure = 100000.0 if step % 2 == 0 else 120000.0

        hydraulic_result = solve_compiled_hydraulic_graph(
            _compile_single_pipe_network(
                _single_pipe_system(
                    upstream_pressure=upstream_pressure,
                    downstream_pressure=downstream_pressure,
                )
            ),
            warm_start=warm_start,
            tolerance=1.0e-12,
            max_iterations=12,
        )
        assert hydraulic_result.converged
        warm_start = hydraulic_result.solution_vector
        flow_rate = hydraulic_result.unknowns["edge_flow"]

        advected_state = forward_advected if flow_rate >= 0.0 else reverse_advected
        update_result = update_lag_transport_state(
            current_state=state_value,
            advected_state=advected_state,
            dt=0.05,
            lag_time_constant_s=0.7,
            through_molar_flow_rate=flow_rate,
            previous_through_molar_flow_rate=previous_flow,
        )
        previous_flow = flow_rate
        state_value = update_result.state

        assert abs(flow_rate) <= 4.0
        assert 0.0 <= update_result.update_fraction <= 1.0
        assert all(isfinite(value) for value in state_value.composition)
        assert all(0.0 <= value <= 1.0 for value in state_value.composition)
        assert sum(state_value.composition) == pytest.approx(1.0, abs=1.0e-9)
        assert isfinite(state_value.temperature)
        assert 200.0 <= state_value.temperature <= 450.0
        if step > 0:
            assert update_result.flow_sign_changed

        trace.append(
            (
                flow_rate,
                state_value.temperature,
                state_value.composition[0],
                state_value.composition[1],
                update_result.update_fraction,
                update_result.flow_sign_changed,
            )
        )

    return tuple(trace)


def test_causality_contract_preserves_pressure_flow_consistency_without_unilateral_hack() -> None:
    forward_system = _single_pipe_system(upstream_pressure=120000.0, downstream_pressure=100000.0)
    reverse_system = _single_pipe_system(upstream_pressure=100000.0, downstream_pressure=120000.0)

    forward = solve_compiled_hydraulic_graph(
        _compile_single_pipe_network(forward_system),
        initial_unknowns=MappingProxyType(
            {
                "edge_flow": 0.0,
                "pressure_upstream": 0.0,
                "pressure_downstream": 0.0,
            }
        ),
        tolerance=1.0e-12,
        max_iterations=12,
    )
    reverse = solve_compiled_hydraulic_graph(
        _compile_single_pipe_network(reverse_system),
        initial_unknowns=MappingProxyType(
            {
                "edge_flow": 0.0,
                "pressure_upstream": 0.0,
                "pressure_downstream": 0.0,
            }
        ),
        tolerance=1.0e-12,
        max_iterations=12,
    )

    assert forward.converged
    assert reverse.converged
    assert forward.unknowns["pressure_upstream"] == pytest.approx(120000.0)
    assert forward.unknowns["pressure_downstream"] == pytest.approx(100000.0)
    assert reverse.unknowns["pressure_upstream"] == pytest.approx(100000.0)
    assert reverse.unknowns["pressure_downstream"] == pytest.approx(120000.0)
    assert forward.unknowns["edge_flow"] > 0.0
    assert reverse.unknowns["edge_flow"] < 0.0
    assert abs(forward.unknowns["edge_flow"]) == pytest.approx(abs(reverse.unknowns["edge_flow"]))

    forward_residual, _, _ = assemble_residual_vector(forward_system, forward.solution_vector)
    reverse_residual, _, _ = assemble_residual_vector(reverse_system, reverse.solution_vector)
    assert max(abs(value) for value in forward_residual) <= 1.0e-8
    assert max(abs(value) for value in reverse_residual) <= 1.0e-8


def test_reversal_sequence_is_bounded_and_deterministic_across_macro_steps() -> None:
    first_trace = _run_reversal_sequence()
    second_trace = _run_reversal_sequence()

    assert len(first_trace) == len(second_trace)
    sign_change_count = 0
    for first, second in zip(first_trace, second_trace, strict=True):
        assert first[0] == pytest.approx(second[0])
        assert first[1] == pytest.approx(second[1])
        assert first[2] == pytest.approx(second[2])
        assert first[3] == pytest.approx(second[3])
        assert first[4] == pytest.approx(second[4])
        assert first[5] == second[5]
        sign_change_count += int(first[5])

    assert sign_change_count == len(first_trace) - 1


def test_representative_topology_closure_preserves_mass_balance_and_stable_bookkeeping() -> None:
    system = _build_three_branch_topology_system()
    compiled = _compile_three_branch_network(system)
    solved = solve_compiled_hydraulic_graph(
        compiled.hydraulic, tolerance=1.0e-10, max_iterations=20
    )
    warm_solved = solve_compiled_hydraulic_graph(
        compiled.hydraulic,
        warm_start=solved.solution_vector,
        tolerance=1.0e-10,
        max_iterations=20,
    )

    assert solved.residual_norm <= 1.0e-8
    assert warm_solved.residual_norm <= 1.0e-8

    flow_feed_a = solved.unknowns["flow_feed_a"]
    flow_feed_b = solved.unknowns["flow_feed_b"]
    flow_outlet = solved.unknowns["flow_outlet"]
    assert flow_feed_a > 0.0
    assert flow_feed_b > 0.0
    assert flow_outlet > 0.0
    assert flow_outlet == pytest.approx(flow_feed_a + flow_feed_b)

    residual_vector, _, _ = assemble_residual_vector(system, solved.solution_vector)
    assert max(abs(value) for value in residual_vector) <= 1.0e-8

    for key in ("flow_feed_a", "flow_feed_b", "flow_outlet", "mix_head"):
        assert solved.unknowns[key] == pytest.approx(warm_solved.unknowns[key])

    mixed = mix_junction_state(
        incoming_port_conditions={
            "feed_a": PortCondition(
                state=MaterialState(
                    pressure=135000.0,
                    temperature=340.0,
                    mole_fractions=(0.8, 0.2),
                ),
                through_molar_flow_rate=flow_feed_a,
            ),
            "feed_b": PortCondition(
                state=MaterialState(
                    pressure=128000.0,
                    temperature=300.0,
                    mole_fractions=(0.2, 0.8),
                ),
                through_molar_flow_rate=flow_feed_b,
            ),
            "product": PortCondition(
                state=MaterialState(
                    pressure=100000.0,
                    temperature=500.0,
                    mole_fractions=(1.0, 0.0),
                ),
                through_molar_flow_rate=-flow_outlet,
            ),
        },
        previous_state=MaterialState(
            pressure=110000.0,
            temperature=320.0,
            mole_fractions=(0.5, 0.5),
        ),
    )

    assert not mixed.used_fallback
    assert mixed.total_incoming_flow_rate == pytest.approx(flow_feed_a + flow_feed_b)

    expected_temperature = (
        flow_feed_a * 340.0 + flow_feed_b * 300.0
    ) / mixed.total_incoming_flow_rate
    expected_x0 = (flow_feed_a * 0.8 + flow_feed_b * 0.2) / mixed.total_incoming_flow_rate
    assert mixed.state.temperature == pytest.approx(expected_temperature)
    assert mixed.state.mole_fractions[0] == pytest.approx(expected_x0)
    assert mixed.state.mole_fractions[1] == pytest.approx(1.0 - expected_x0)

    binding = compiled.process_bindings["mixer"]
    binding.bind_inlets({"feed_a": flow_feed_a, "feed_b": flow_feed_b})
    binding.bind_outlets(
        {
            "product": {
                "role": "outlet",
                "flow": flow_outlet,
                "composition": mixed.state.mole_fractions,
            },
            "purge": {
                "role": "outlet",
                "flow": 0.0,
                "composition": mixed.state.mole_fractions,
            },
        }
    )

    product_binding = binding.get_outlet("product")
    purge_binding = binding.get_outlet("purge")
    assert product_binding["outlet_flow"] == pytest.approx(flow_outlet)
    assert purge_binding["outlet_flow"] == pytest.approx(0.0)
    total_inlet = flow_feed_a + flow_feed_b
    total_outlet = product_binding["outlet_flow"] + purge_binding["outlet_flow"]
    assert total_inlet == pytest.approx(total_outlet)
