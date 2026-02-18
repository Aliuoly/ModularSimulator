# Work Plan: Connection Layer (Minimal Realistic + Extensible)

> **Generated**: 2026-02-16  
> **Status**: Revised after Momus + user critique  
> **Scope**: Bidirectional process-network coupling with realistic hydraulics and transport

---

## TL;DR

**Objective**: Build a connection layer that is physically coherent and numerically stable without over-engineering MVP.

**What changed in this revision**:
1. `PortCondition` applies to **all** process ports (inlet and outlet) so processes always receive solved pressure/flow for balance closure.
2. Hydraulics remain equation-based (`HydraulicElement`) to preserve causality (pumps are equations, not pressure BC hacks).
3. Transport split into two tiers: **MVP lag model** first, **conservative holdup model** second.
4. Density model is pluggable: incompressible liquid default + ideal-gas option for gas realism.
5. Junctions do algebraic mixing only (no hidden extra holdup).

**Estimated Effort**: Large (14-18 dev days)  
**Parallel Execution**: YES - 6 waves + final verification  
**Critical Path**: PortCondition + HydraulicElement -> Solver -> Coupling loop -> Process adapter -> examples

---

## Context

### Blocking issues addressed

| Issue | Resolution in this plan |
|---|---|
| Causality trap | Pumps/valves/pipes implemented as `HydraulicElement` residual equations |
| Flow reversal discontinuity | Soft upwind blending in transport (`tanh`) + smooth hydraulic absolute value |
| Undefined interface | Explicit `MaterialState`, `PortCondition`, `OutletMaterial`, `ProcessInterface` |

### User-required corrections integrated

- Process must receive solved outlet/inlet flows and pressures each macro step to close ODE mass/energy balances.
- No finite-difference Jacobian entries inside production elements (pump derivative callback required).
- Smooth hydraulic law near zero flow (`m_dot * sqrt(m_dot^2 + delta^2)`) to reduce Newton chatter.
- Pluggable density model with ideal-gas path for gas systems.
- Junction mixing is algebraic unless explicit junction holdup is modeled.
- Macro coupling with warm-started hydraulics solve for speed.

---

## Work Objectives

### Core Objective

Deliver a connection layer that computes pressures/flows robustly, propagates composition/energy realistically, and interoperates with existing `ProcessModel` dynamics through a bidirectional port interface.

### Concrete Deliverables

- `src/modular_simulation/connection/state.py`
- `src/modular_simulation/connection/material_models.py`
- `src/modular_simulation/connection/hydraulic_element.py`
- `src/modular_simulation/connection/hydraulic_solver.py`
- `src/modular_simulation/connection/transport.py`
- `src/modular_simulation/connection/junction.py`
- `src/modular_simulation/connection/connection_layer.py`
- `src/modular_simulation/connection/process_interface.py`
- `tests/connection/` suite for hydraulics, transport, coupling, and examples

### Definition of Done

- [x] Port conditions delivered to all ports every macro step
- [x] Hydraulics converges on representative split/recycle networks
- [x] Reverse flow handled without solver instability
- [x] Junction mixing validated and deterministic under no-incoming case
- [x] MVP lag transport validated; advanced conservative transport spec implemented or explicitly deferred
- [x] Lint + type checks + tests pass

### Must Have

- Bidirectional `PortCondition` for every process port
- `HydraulicElement` abstraction with sparse Jacobian hooks
- Macro coupling interval (`delta_t_c`) with warm start
- Algebraic junction mixing from incoming edge states
- Density model plugin (incompressible + ideal gas)
- Full network solvability checks (pressure references + singularity checks)

### Must NOT Have

- No unilateral outlet pressure ownership by process components
- No hard sign switch in transport flux selection
- No hidden junction holdup when edge holdup already models delay
- No finite-difference Jacobian inside runtime element residuals

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (`pytest`)
- **Automated tests**: Tests-after
- **QA mode**: Agent-executed checks only

### MVP vs Advanced Verification

- **MVP transport**: Delay/lag propagation on composition and temperature, with stable reversal behavior.
- **Advanced transport**: Conservative holdup (component mass + enthalpy) enabled behind explicit model choice.

### Physical invariants to verify

- Hydraulic residual closure on all elements and nodes
- Junction continuity (signed incidence mass continuity)
- No pressure-floating connected components
- Stable behavior near zero flow and through reversals

---

## Execution Strategy

### Parallel Waves

```text
Wave 0 (Interface + properties)
- T0.1 MaterialState + PortCondition + OutletMaterial
- T0.2 DensityModel plugins (incompressible, ideal gas)
- T0.3 Mixture property helpers

Wave 1 (Hydraulic elements)
- T1.1 HydraulicElement ABC with generic inputs
- T1.2 Pipe and valve elements using smooth abs formulation
- T1.3 Pump element with derivative callback (no finite difference)

Wave 2 (Hydraulic solver)
- T2.1 Incidence/topology assembly
- T2.2 Residual assembly (elements + node continuity + constraints)
- T2.3 Sparse Jacobian assembly
- T2.4 Damped Newton + warm start
- T2.5 Solvability checks (reference pressure, singular loops)

Wave 3 (Transport + junction)
- T3.1 MVP lag transport model (composition + temperature)
- T3.2 Soft-upwind reversible flux operator
- T3.3 Algebraic junction mixing from incoming delayed states
- T3.4 No-incoming fallback: hold-last-state + warning
- T3.5 Advanced conservative transport scaffold (optional model)

Wave 4 (Coupling loop)
- T4.1 Macro coupling scheduler (delta_t_c)
- T4.2 Coupling sequence implementation (process -> hydraulics -> transport/mix -> port conditions)
- T4.3 Optional Picard iterations gated by residual threshold

Wave 5 (Process integration)
- T5.1 ProcessInterface ABC
- T5.2 ProcessAdapter for existing ProcessModel
- T5.3 Mapping of PortCondition to process balance terms
- T5.4 Integration tests with existing System/ProcessModel semantics

Wave 6 (Examples + performance)
- T6.1 Pump causality example
- T6.2 Split junction with reversal example
- T6.3 Macro-coupling performance benchmark

Wave FINAL
- F1 Physical coherence review
- F2 Numerical robustness and convergence audit
- F3 Scope fidelity and documentation check
```

---

## TODOs

- [x] 1. Define universal state and port interfaces (`T0.1`)
  - **What to do**: implement `MaterialState`, `PortCondition`, `OutletMaterial` with validation.
  - **Acceptance**: signed flow semantics documented and tested.
  - **QA scenario (happy)**: instantiate valid states and port conditions.
  - **QA scenario (error)**: invalid mole fractions fail validation.

- [x] 2. Implement density model plugin layer (`T0.2`)
  - **What to do**: `IncompressibleDensityModel` and `IdealGasDensityModel`.
  - **Acceptance**: model selection is explicit per edge/system config.
  - **QA scenario (happy)**: ideal-gas density responds to P and T.
  - **QA scenario (error)**: invalid temperature/pressure rejected.

- [x] 3. Implement hydraulic element interface with generic inputs (`T1.1`)
  - **What to do**: residual/Jacobian API uses `inputs: Mapping[str, float]`.
  - **Acceptance**: same API supports pipe/valve/pump without valve-only assumptions.

- [x] 4. Implement smooth pipe and valve laws (`T1.2`)
  - **What to do**: use smooth absolute term to avoid derivative kinks.
  - **Acceptance**: Jacobian remains continuous near zero flow.

- [x] 5. Implement pump element with derivative callback (`T1.3`)
  - **What to do**: require both `dp_curve(mdot, speed)` and `d_dp_d_mdot(mdot, speed)`.
  - **Acceptance**: no finite-difference derivative in runtime residual path.

- [x] 6. Build sparse hydraulic solver (`T2.1-T2.4`)
  - **What to do**: unknown vector, residual vector, sparse Jacobian, damped Newton, warm-start.
  - **Acceptance**: converges on split and recycle test topologies.

- [x] 7. Add solvability and DOF checks (`T2.5`)
  - **What to do**: pressure reference per connected component, singularity detection, disconnected graph handling.
  - **Acceptance**: invalid networks fail fast with actionable errors.

- [x] 8. Implement MVP lag transport (`T3.1`)
  - **What to do**: delay model for composition/temperature propagation under adaptive stepping.
  - **Acceptance**: stable through sign changes when paired with soft-upwind operator.

- [x] 9. Implement junction mixing and fallback (`T3.3-T3.4`)
  - **What to do**: flow-weighted mixing over incoming edges only; hold-last-state when no incoming.
  - **Acceptance**: deterministic behavior for zero-inflow edge cases.

- [x] 10. Add advanced conservative transport scaffold (`T3.5`)
  - **What to do**: pluggable model interface for future mass/enthalpy-holdup transport.
  - **Acceptance**: runtime model selection supports MVP and advanced mode.

- [x] 11. Implement macro coupling sequence (`T4.1-T4.2`)
  - **What to do**: at `t_k` perform process outlet update -> hydraulics solve -> transport/mix -> deliver port conditions -> process advance.
  - **Acceptance**: no hydraulics solve inside every ODE RHS call.

- [x] 12. Add optional Picard iteration gate (`T4.3`)
  - **What to do**: only activate when coupling residual exceeds threshold.
  - **Acceptance**: remains off for weakly coupled cases.

- [x] 13. Implement ProcessInterface and adapter (`T5.1-T5.2`)
  - **What to do**: adapter for current `ProcessModel` patterns while preserving existing behavior.
  - **Acceptance**: existing process models can be connected without rewriting kinetics core.

- [x] 14. Map port conditions into process balances (`T5.3`)
  - **What to do**: ensure process RHS uses solved per-port flow/pressure and incoming state.
  - **Acceptance**: outlet and inlet terms both available each macro step.

- [x] 15. Build integration and physics tests (`T5.4`)
  - **What to do**: validate causality, reversal stability, and closure on representative topologies.
  - **Acceptance**: tests demonstrate no blocker from previous Momus findings.

- [x] 16. Add examples and benchmarks (`T6.1-T6.3`)
  - **What to do**: pump causality demo, split/reversal demo, macro-coupling performance run.
  - **Acceptance**: examples reproduce expected qualitative behavior and benchmark output is recorded.

---

## Final Verification Wave

- [x] F1. **Physical coherence audit**
  - Verify node pressure uniqueness, continuity, and process balance closure with solved port conditions.

- [x] F2. **Numerical robustness audit**
  - Verify convergence with warm start, near-zero flow smoothness, and reversal continuity.

- [x] F3. **Scope fidelity audit**
  - Confirm MVP remains minimal (lag transport) while advanced conservative transport remains pluggable and not mandatory.

---

## Success Criteria

### Verification commands

```bash
uv run pytest tests/connection/ -v
uv run ruff check
uv run basedpyright
```

### Final checklist

- [x] All port conditions provided for all process ports each macro step
- [x] Hydraulics solved with sparse Jacobian and warm start
- [x] Smooth near-zero flow behavior in hydraulic elements
- [x] Pump derivative supplied analytically/by callback (no finite-difference in runtime)
- [x] Junction mixing is algebraic only unless explicit junction holdup is configured
- [x] MVP lag transport operational and stable through reversal
- [x] Advanced conservative transport path scaffolded

---

**Plan Revised**: Ready for final review  
**Next Step**: Optional Momus pass, then `/start-work`
