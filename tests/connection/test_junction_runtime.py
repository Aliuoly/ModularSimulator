from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

import importlib

import pytest

junction = importlib.import_module("modular_simulation.connection.junction")
junction_runtime = importlib.import_module("modular_simulation.connection.junction_runtime")
state = importlib.import_module("modular_simulation.connection.state")
topology = importlib.import_module("modular_simulation.connection.topology")

BoundaryNode = topology.BoundaryNode
ConnectionEdge = topology.ConnectionEdge
JunctionNode = topology.JunctionNode
MaterialState = state.MaterialState
PortCondition = state.PortCondition
PortReference = topology.PortReference
TopologyGraph = topology.TopologyGraph
mix_junction_runtime_states = junction_runtime.mix_junction_runtime_states
mix_junction_state = junction.mix_junction_state


def _single_junction_graph() -> TopologyGraph:
    return TopologyGraph(
        process_nodes=(),
        boundary_nodes=(
            BoundaryNode(boundary_id="feed_a", port_name="outlet", direction="source"),
            BoundaryNode(boundary_id="feed_b", port_name="outlet", direction="source"),
            BoundaryNode(boundary_id="product", port_name="inlet", direction="sink"),
        ),
        junction_nodes=(
            JunctionNode(junction_id="mix", port_names=("inlet_a", "inlet_b", "outlet")),
        ),
        edges=(
            ConnectionEdge(
                edge_id="edge_product",
                source=PortReference(node_id="mix", port_name="outlet"),
                target=PortReference(node_id="product", port_name="inlet"),
            ),
            ConnectionEdge(
                edge_id="edge_feed_b",
                source=PortReference(node_id="feed_b", port_name="outlet"),
                target=PortReference(node_id="mix", port_name="inlet_b"),
            ),
            ConnectionEdge(
                edge_id="edge_feed_a",
                source=PortReference(node_id="feed_a", port_name="outlet"),
                target=PortReference(node_id="mix", port_name="inlet_a"),
            ),
        ),
    )


def _two_junction_graph_unsorted() -> TopologyGraph:
    return TopologyGraph(
        process_nodes=(),
        boundary_nodes=(
            BoundaryNode(boundary_id="feed_a", port_name="outlet", direction="source"),
            BoundaryNode(boundary_id="feed_b", port_name="outlet", direction="source"),
            BoundaryNode(boundary_id="sink_a", port_name="inlet", direction="sink"),
            BoundaryNode(boundary_id="sink_b", port_name="inlet", direction="sink"),
        ),
        junction_nodes=(
            JunctionNode(junction_id="junction_b", port_names=("inlet", "outlet")),
            JunctionNode(junction_id="junction_a", port_names=("inlet", "outlet")),
        ),
        edges=(
            ConnectionEdge(
                edge_id="edge_b_out",
                source=PortReference(node_id="junction_b", port_name="outlet"),
                target=PortReference(node_id="sink_b", port_name="inlet"),
            ),
            ConnectionEdge(
                edge_id="edge_a_in",
                source=PortReference(node_id="feed_a", port_name="outlet"),
                target=PortReference(node_id="junction_a", port_name="inlet"),
            ),
            ConnectionEdge(
                edge_id="edge_a_out",
                source=PortReference(node_id="junction_a", port_name="outlet"),
                target=PortReference(node_id="sink_a", port_name="inlet"),
            ),
            ConnectionEdge(
                edge_id="edge_b_in",
                source=PortReference(node_id="feed_b", port_name="outlet"),
                target=PortReference(node_id="junction_b", port_name="inlet"),
            ),
        ),
    )


def test_runtime_maps_fanin_and_fanout_edges_with_mixer_parity() -> None:
    graph = _single_junction_graph()
    previous = MaterialState(
        pressure=110000.0,
        temperature=320.0,
        mole_fractions=(0.5, 0.5),
    )
    edge_states = {
        "edge_feed_a": MaterialState(
            pressure=100000.0,
            temperature=300.0,
            mole_fractions=(0.8, 0.2),
        ),
        "edge_feed_b": MaterialState(
            pressure=200000.0,
            temperature=350.0,
            mole_fractions=(0.2, 0.8),
        ),
        "edge_product": MaterialState(
            pressure=900000.0,
            temperature=1000.0,
            mole_fractions=(1.0, 0.0),
        ),
    }
    edge_flow_rates = {
        "edge_feed_a": 2.0,
        "edge_feed_b": 1.0,
        "edge_product": 3.0,
    }

    runtime_results = mix_junction_runtime_states(
        topology=graph,
        edge_states=edge_states,
        edge_flow_rates=edge_flow_rates,
        previous_junction_states={"mix": previous},
    )

    expected = mix_junction_state(
        incoming_port_conditions={
            "edge_feed_a": PortCondition(
                state=edge_states["edge_feed_a"],
                through_molar_flow_rate=2.0,
            ),
            "edge_feed_b": PortCondition(
                state=edge_states["edge_feed_b"],
                through_molar_flow_rate=1.0,
            ),
            "edge_product": PortCondition(
                state=edge_states["edge_product"],
                through_molar_flow_rate=-3.0,
            ),
        },
        previous_state=previous,
    )

    assert tuple(runtime_results.keys()) == ("mix",)
    assert runtime_results["mix"] == expected
    assert runtime_results["mix"].total_incoming_flow_rate == pytest.approx(3.0)
    assert not runtime_results["mix"].used_fallback


def test_runtime_preserves_near_zero_fallback_and_sorted_junction_keys() -> None:
    graph = _two_junction_graph_unsorted()
    previous_states = {
        "junction_a": MaterialState(
            pressure=111000.0,
            temperature=311.0,
            mole_fractions=(0.6, 0.4),
        ),
        "junction_b": MaterialState(
            pressure=112000.0,
            temperature=312.0,
            mole_fractions=(0.3, 0.7),
        ),
    }
    edge_states_order_a = {
        "edge_b_out": MaterialState(
            pressure=210000.0,
            temperature=350.0,
            mole_fractions=(0.2, 0.8),
        ),
        "edge_a_in": MaterialState(
            pressure=101000.0,
            temperature=301.0,
            mole_fractions=(0.9, 0.1),
        ),
        "edge_a_out": MaterialState(
            pressure=190000.0,
            temperature=330.0,
            mole_fractions=(0.1, 0.9),
        ),
        "edge_b_in": MaterialState(
            pressure=102000.0,
            temperature=302.0,
            mole_fractions=(0.7, 0.3),
        ),
    }
    edge_flow_rates_order_a = {
        "edge_b_out": 5.0e-13,
        "edge_a_in": 3.0e-13,
        "edge_a_out": 4.0e-13,
        "edge_b_in": 7.0e-13,
    }
    edge_states_order_b = {
        edge_id: edge_states_order_a[edge_id]
        for edge_id in ("edge_b_in", "edge_a_out", "edge_a_in", "edge_b_out")
    }
    edge_flow_rates_order_b = {
        edge_id: edge_flow_rates_order_a[edge_id]
        for edge_id in ("edge_b_in", "edge_a_out", "edge_a_in", "edge_b_out")
    }

    result_a = mix_junction_runtime_states(
        topology=graph,
        edge_states=edge_states_order_a,
        edge_flow_rates=edge_flow_rates_order_a,
        previous_junction_states=previous_states,
        near_zero_incoming_flow_epsilon=1.0e-12,
    )
    result_b = mix_junction_runtime_states(
        topology=graph,
        edge_states=edge_states_order_b,
        edge_flow_rates=edge_flow_rates_order_b,
        previous_junction_states=previous_states,
        near_zero_incoming_flow_epsilon=1.0e-12,
    )

    assert tuple(result_a.keys()) == ("junction_a", "junction_b")
    assert tuple(result_b.keys()) == ("junction_a", "junction_b")
    assert result_a == result_b
    assert result_a["junction_a"].used_fallback
    assert result_a["junction_b"].used_fallback
    assert result_a["junction_a"].total_incoming_flow_rate == pytest.approx(3.0e-13)
    assert result_a["junction_b"].total_incoming_flow_rate == pytest.approx(7.0e-13)
    assert result_a["junction_a"].state == previous_states["junction_a"]
    assert result_a["junction_b"].state == previous_states["junction_b"]
