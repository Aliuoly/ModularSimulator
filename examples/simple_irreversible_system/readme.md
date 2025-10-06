# Simple Irreversible Reaction CSTR

This example models an isothermal CSTR performing the irreversible reaction
A → 2B.  It demonstrates how to express a small DAE system inside the
framework and close the loop with a single feedback controller.

## System definition

* **Differential states (`States`)** – reactor volume `V`, concentrations of A
  `C_A` and B `C_B`.  These are advanced by the solver through the `rhs`
  implementation in `IrreversibleSystem`.【F:examples/simple_irreversible_system/system_definitions.py†L10-L78】
* **Control elements (`ControlElements`)** – a single manipulated variable,
  the inlet flow rate `F_in`.【F:examples/simple_irreversible_system/system_definitions.py†L15-L17】
* **Algebraic states (`AlgebraicStates`)** – the outlet flow `F_out`, computed
  at every solver call from the current volume using the valve correlation
  `F_out = Cv √V`.  The algebraic value is fed back into the RHS when
  calculating the mass balances.【F:examples/simple_irreversible_system/system_definitions.py†L19-L51】
* **Constants** – kinetic and hydraulic parameters `k`, `Cv` and feed
  concentration `CA_in`.  They are provided through the `Constants` Pydantic
  model when the system is created.【F:examples/simple_irreversible_system/system_definitions.py†L21-L25】

The algebraic and differential equations are implemented in
`IrreversibleSystem`.  No additional calculations are required for this
example.

### Governing equations

The discharge flow is enforced algebraically from the current volume,

$$
F_{\text{out}} = C_v \sqrt{V}.
$$

With the reaction rate $r = k V C_A$, the differential states evolve according
to

\begin{align*}
\frac{\mathrm{d}V}{\mathrm{d}t} &= F_{\text{in}} - F_{\text{out}}, \\
\frac{\mathrm{d}C_A}{\mathrm{d}t} &= \frac{1}{V} \left(-r + F_{\text{in}} C_{A,\text{in}} - F_{\text{out}} C_A - C_A \frac{\mathrm{d}V}{\mathrm{d}t}\right), \\
\frac{\mathrm{d}C_B}{\mathrm{d}t} &= \frac{1}{V} \left(2 r - F_{\text{out}} C_B - C_B \frac{\mathrm{d}V}{\mathrm{d}t}\right).
\end{align*}

These expressions match the implementation in `system_definitions.py` and form
the coupled mass balances advanced by the integrator.

## Instrumentation

The simulation uses `SampledDelayedSensor` instances with different settings to
illustrate realistic measurements:

* `F_out` – instantaneous sample of the algebraic discharge flow.
* `F_in` – noisy flow measurement with a 5% coefficient of variance.
* `B` – lab analyser-style sample with a 900 s sampling period and dead time.
* `V` – volume sensor with 1% probability of returning a flagged faulty value.

All sensors produce `TagData` outputs that are historized by
the framework.【F:examples/simple_irreversible_system/run_simulation.py†L33-L60】

No calculated signals are defined in this example.

## Control strategy

A single `PIDController` manipulates `F_in` to track the concentration of B.
The controller uses a constant setpoint trajectory and enforces bounds on the
manipulated flow.【F:examples/simple_irreversible_system/run_simulation.py†L62-L76】

## Simulation settings

* **Update period (`dt`)** – 30 s between measurement/controller updates.
* **Integrator** – defaults to LSODA through the `solver_options` on the
  `System` base class.【F:examples/simple_irreversible_system/run_simulation.py†L86-L109】【F:src/modular_simulation/framework/system.py†L70-L88】
* **Variants** – the script creates both a normal Python implementation and a
  Numba-accelerated version to compare performance.【F:examples/simple_irreversible_system/run_simulation.py†L86-L110】

## Running the example

From the repository root run:

```bash
python examples/simple_irreversible_system/run_simulation.py
```

The script simulates 10,000 controller updates, applies a setpoint change
half-way through, and plots the resulting measurements and controller outputs.
