# Modular Simulation Framework

`modular_simulation` is a lightweight framework for building closed-loop
process simulations.  It separates the physical model, instrumentation and
control logic into composable pieces so that new systems can be assembled from
small, well-typed components.

The framework is designed around Differential-Algebraic Equation (DAE) systems
that are stepped with `scipy.integrate.solve_ivp`.  Algebraic variables are
recomputed at every solver evaluation, which keeps implementations readable
while still supporting systems that mix differential and algebraic states.

## Core Concepts

### Measurable quantities
The full process state is stored in a `MeasurableQuantities` instance, which
contains four Pydantic models:

* `States`: differential states advanced by the solver (e.g., material and
  energy balances).
* `ControlElements`: final control elements that can be manipulated by
  controllers.
* `AlgebraicStates`: values that are recomputed from the current differential
  state and controls each solver step.
* `Constants`: fixed parameters such as equipment sizes, physical properties or
  kinetic coefficients.

Each category provides structured accessors, automatic indexing and helper
methods for working with NumPy arrays, which keeps the right-hand-side (RHS)
implementation concise.【F:src/modular_simulation/quantities/measurable_quantities.py†L1-L41】

### Usable quantities
Sensors and calculations are grouped in a `UsableQuantities` object.  Sensors
sample `MeasurableQuantities` values and return `TagData`
objects that carry the measurement time, value and a quality flag.  Built-in
sensors support realistic effects such as sample-and-hold behaviour, dead time,
noise and fault injection.【F:src/modular_simulation/usables/sensors/sampled_delayed_sensor.py†L1-L146】

Calculations are stateful callables that consume previously measured values to
produce derived signals (e.g., soft sensors or quality checks).  Both sensors
and calculations are historized automatically when the simulation runs, so the
results can be plotted or analysed afterwards.【F:src/modular_simulation/framework/system.py†L103-L136】【F:src/modular_simulation/framework/system.py†L172-L203】

### Controllable quantities
Controllers inherit from the `Controller` base class and operate on
measurements provided by the `UsableQuantities`.  They output commands that are
written back to the `ControlElements`.  The default PID implementation includes
features like setpoint trajectories, output limiting, ramping and cascade
arrangements.【F:src/modular_simulation/control_system/controllers/controller.py†L24-L207】

All controllers, sensors and calculations are validated and linked together
when a `System` instance is created, ensuring that every referenced tag is
available and avoiding mismatched wiring.【F:src/modular_simulation/validation/system_validation.py†L1-L118】

## Defining a system

To model a new process, subclass `System` and implement two static methods:

* `calculate_algebraic_values(...)` computes the algebraic state vector for a
  given set of differential states, controls and constants.
* `rhs(t, y, u, k, algebraic, ...)` returns the derivatives of the differential
  states.

Both methods receive NumPy arrays together with mapping dictionaries that map
field names to array slices, allowing the implementation to remain readable and
self-documenting.【F:src/modular_simulation/framework/system.py†L1-L201】

Additional options on `System` let you control the integration behaviour:

* `dt`: update period for sensors, calculations and controllers.
* `solver_options`: forwarded directly to `solve_ivp` (defaults to LSODA).
* `use_numba` / `numba_options`: enable optional JIT compilation of the RHS and
  algebraic functions for faster simulations.
* `record_history`: toggle internal logging of the raw states when memory is a
  concern.【F:src/modular_simulation/framework/system.py†L44-L125】

## Building and running simulations

The `create_system` helper wires together the Pydantic models, sensors,
calculations and controllers into a complete simulation object.  The helper
duplicates the provided objects to ensure that multiple simulations can run in
parallel without sharing state.【F:src/modular_simulation/framework/utils.py†L1-L81】

A typical workflow is:

1. Define `States`, `ControlElements`, `AlgebraicStates` and `Constants`
   subclasses for your process.
2. Implement a `System` subclass with the appropriate algebraic calculation and
   RHS methods.
3. Instantiate sensors, calculations and controllers that operate on named
   tags.
4. Call `create_system(...)` to obtain a ready-to-run simulation instance.
5. Advance the simulation with `system.step(nsteps=...)` or
   `system.run_until(time=...)`.
6. Use `system.measured_history` and `system.setpoint_history` for plotting or
   data analysis, or access `system.measurable_history` when `record_history`
   is enabled.【F:src/modular_simulation/framework/system.py†L137-L171】【F:src/modular_simulation/framework/system.py†L213-L328】

During a step, the framework executes the following sequence:

1. Synchronise algebraic states with the current differential states.
2. Advance the numerical integrator to the next measurement time.
3. Update sensors and calculations to produce usable measurements.
4. Run each controller (respecting cascade hierarchies) and write back new
   control commands.
5. Record measurement and setpoint histories for later analysis.【F:src/modular_simulation/framework/system.py†L213-L328】【F:src/modular_simulation/control_system/controllers/controller.py†L180-L207】

## Additional utilities

* **Trajectories** – Piecewise setpoint profiles that can be chained together
  for scenarios and setpoint scheduling.【F:src/modular_simulation/control_system/trajectory.py†L1-L143】
* **Plotting helpers** – Convenience functions for plotting
  `TagData` histories.【F:src/modular_simulation/plotting/__init__.py†L1-L34】
* **Validation** – Automatic tag validation prevents runtime errors caused by
  missing sensors or controllers.【F:src/modular_simulation/validation/system_validation.py†L1-L118】

## Examples

The `examples/` directory contains end-to-end systems that demonstrate mass and
energy balances, algebraic constraints, noisy measurements and multiloop
control.  See each example's `readme.md` for the full configuration and setup
details.

## Running the example simulations

Each example directory exposes a `run_simulation.py` script.  Run the desired
scenario with:

```bash
python examples/<example_name>/run_simulation.py
```

Most scripts instantiate both a standard and a Numba-accelerated system to
illustrate the identical usage pattern across different performance targets.
