from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownParameterType=false, reportImplicitStringConcatenation=false

from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from modular_simulation.connection.hydraulic_compile import (
    HydraulicCompileLifecycle,
    solve_compiled_hydraulic_graph,
)
from modular_simulation.connection.hydraulic_element import PumpHydraulicElement
from modular_simulation.connection.hydraulic_solver import (
    ElementEquation,
    HydraulicElementLike,
    HydraulicSystemDefinition,
)
from modular_simulation.connection.network import ConnectionNetwork
from modular_simulation.connection.topology import TopologyGraph


@dataclass(frozen=True)
class PumpCase:
    name: str
    upstream_pressure: float
    downstream_pressure: float
    pump_speed: float


def _dp_curve(flow_rate: float, pump_speed: float) -> float:
    return 30000.0 * (pump_speed / 1000.0) ** 2 - 4000.0 * flow_rate


def _d_dp_d_mdot(flow_rate: float, pump_speed: float) -> float:
    del flow_rate, pump_speed
    return -4000.0


def _build_pump_system(*, case: PumpCase, topology: TopologyGraph) -> HydraulicSystemDefinition:
    del topology
    pump_equation = ElementEquation(
        name="pump_head_balance",
        element=cast(
            HydraulicElementLike,
            cast(object, PumpHydraulicElement(dp_curve=_dp_curve, d_dp_d_mdot=_d_dp_d_mdot)),
        ),
        inputs=MappingProxyType(
            {
                "upstream_pressure": case.upstream_pressure,
                "downstream_pressure": case.downstream_pressure,
            }
        ),
        parameters=MappingProxyType({"pump_speed": case.pump_speed}),
        unknown_name_map=MappingProxyType({"flow_rate": "edge_flow"}),
        residual_name_map=MappingProxyType({"head_balance": "pump_head_balance"}),
    )
    return HydraulicSystemDefinition(equations=(pump_equation,))


def _compile_pump_network(case: PumpCase):
    connection_network = ConnectionNetwork(
        compile_lifecycle=HydraulicCompileLifecycle(),
        hydraulic_system_builder=lambda topology: _build_pump_system(case=case, topology=topology),
    )
    connection_network.add_process("pump", inlet_ports=("inlet",), outlet_ports=("outlet",))
    connection_network.add_boundary_source("upstream")
    connection_network.add_boundary_sink("downstream")
    connection_network.connect("upstream", "pump.inlet")
    connection_network.connect("pump.outlet", "downstream")
    compiled = connection_network.compile()
    if compiled.hydraulic is None:
        raise RuntimeError("compiled pump network missing hydraulic graph")
    return compiled.hydraulic


def run_demo() -> list[tuple[PumpCase, float, int, float]]:
    cases = [
        PumpCase(
            "low_speed", upstream_pressure=120000.0, downstream_pressure=140000.0, pump_speed=700.0
        ),
        PumpCase(
            "mid_speed", upstream_pressure=120000.0, downstream_pressure=140000.0, pump_speed=900.0
        ),
        PumpCase(
            "high_speed",
            upstream_pressure=120000.0,
            downstream_pressure=140000.0,
            pump_speed=1100.0,
        ),
        PumpCase(
            "high_backpressure",
            upstream_pressure=120000.0,
            downstream_pressure=155000.0,
            pump_speed=900.0,
        ),
        PumpCase(
            "reverse_flow",
            upstream_pressure=120000.0,
            downstream_pressure=170000.0,
            pump_speed=700.0,
        ),
    ]

    warm_start = None
    rows: list[tuple[PumpCase, float, int, float]] = []
    for case in cases:
        result = solve_compiled_hydraulic_graph(
            _compile_pump_network(case),
            warm_start=warm_start,
            tolerance=1.0e-12,
            max_iterations=25,
        )
        if not result.converged:
            raise RuntimeError(f"hydraulic solve did not converge for case {case.name!r}")
        warm_start = result.solution_vector
        rows.append((case, result.unknowns["edge_flow"], result.iterations, result.residual_norm))

    flow_by_name = {case.name: flow for case, flow, _, _ in rows}
    if not (flow_by_name["low_speed"] < flow_by_name["mid_speed"] < flow_by_name["high_speed"]):
        raise RuntimeError("pump speed causality check failed")
    if not flow_by_name["high_backpressure"] < flow_by_name["mid_speed"]:
        raise RuntimeError("backpressure causality check failed")
    if not flow_by_name["reverse_flow"] < 0.0:
        raise RuntimeError("reverse-flow causality check failed")

    return rows


def main() -> None:
    rows = run_demo()

    print("PUMP CAUSALITY DEMO")
    print("case,upstream_pa,downstream_pa,speed_rpm,flow_rate,iterations,residual_norm")
    for case, flow_rate, iterations, residual_norm in rows:
        print(
            f"{case.name},{case.upstream_pressure:.1f},{case.downstream_pressure:.1f},{case.pump_speed:.1f},{flow_rate:.8f},{iterations},{residual_norm:.3e}"
        )
    print("monotonic checks: PASS")


if __name__ == "__main__":
    main()
