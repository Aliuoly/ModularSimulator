# Source Organization Proposal

This document captures a high-level proposal for reorganising the package layout in the next release while preserving the separation between simulation primitives and higher-level orchestrators.

## Current layout highlights

- `src/modular_simulation/framework/system.py` holds the main `System` base class with the simulation loop, optional Numba acceleration and history collection logic.【F:src/modular_simulation/framework/system.py†L1-L120】
- `src/modular_simulation/quantities/` defines the Pydantic models for measurable, usable and controllable collections that are passed through the solver and controllers.【F:src/modular_simulation/quantities/measurable_quantities.py†L1-L41】
- `src/modular_simulation/usables/` contains the sensor and calculation implementations that generate `TimeValueQualityTriplet` data for controllers.【F:src/modular_simulation/usables/sensors/sampled_delayed_sensor.py†L1-L146】
- `src/modular_simulation/control_system/` hosts controller abstractions, including PID-style logic and trajectory helpers.【F:src/modular_simulation/control_system/controllers/controller.py†L24-L207】【F:src/modular_simulation/control_system/trajectory.py†L1-L143】

## Proposed folder structure for the next iteration

The goal is to make the package easier to navigate by grouping files by responsibility and adding clear extension points:

1. **`core/`** – foundational models and runtime orchestration
   - `core/system.py` would keep the main simulation loop and orchestration APIs currently found in `framework/system.py`.
   - `core/state.py` could consolidate the measurable/usable/controllable quantity definitions so that downstream users only import from one namespace.
   - `core/runtime.py` can expose helpers like `create_system` and shared dataclasses that coordinate the stepping process (migrated from `framework/utils.py`).

2. **`components/`** – reusable building blocks for simulations
   - `components/controllers/` for controller implementations and mixins (`PID`, cascade helpers, trajectories, bumpless transfer utilities).
   - `components/sensors/` and `components/calculations/` for the measurable pipeline, leaving room for community contributions.
   - `components/actuators/` for logic around control elements that may warrant their own module in future releases.

3. **`domain/`** – plant- or industry-specific assets
   - Group specialised systems, such as any custom `System` subclasses shipped with the package, under `domain/<process_name>/` to avoid polluting `core/`.
   - Encourage examples and templates to import from this namespace instead of reaching directly into `core/` internals.

4. **`io/`** – persistence and visualisation utilities
   - Move plotting helpers (currently in `plotting/`) and any file export utilities into a dedicated namespace to clarify the boundary between runtime logic and presentation.
   - Provide adapters for history serialisation so alternative front-ends can plug in without depending on plotting-specific code paths.

5. **`validation/`** – cross-cutting validation rules
   - Keep validation routines in their own top-level package but break files down by concern (`system`, `wiring`, `configuration`) once they grow.

6. **`examples/`** – usage scenarios
   - Maintain the current example layout but add a short `INDEX.md` outlining which part of the reorganised library each example exercises.

## Migration recommendations

- Introduce transitional re-export modules (e.g., `modular_simulation.framework.system = modular_simulation.core.system`) to avoid breaking downstream imports immediately.
- Update documentation and type hints to prefer the new namespaces, and mark legacy modules with deprecation warnings once the new paths are stable.
- Review dependency footprints (Numba, SciPy, plotting backends) and only import optional packages within the modules that require them to keep `core/` light-weight.
- Capture architectural decisions in `docs/` (ADR-style) whenever a large refactor is made so future contributors understand the rationale.

## Next steps

1. Finalise the target names inside `core/` and `components/` and update the public API exports (`__init__.py`) to guide users to the right entry points.
2. Draft a migration checklist that includes CI verification and example updates so the reorganisation can be completed incrementally.
3. Publish a `CHANGELOG` entry highlighting the new layout and any deprecated import paths to prepare downstream projects for the transition.
