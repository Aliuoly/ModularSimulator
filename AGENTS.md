# Repository Guidance

This repository hosts the Modular Simulation framework. Use the following notes when navigating or extending the code base:

- High-level package architecture and the recommended reorganisation for upcoming versions are documented in `docs/source_organization_proposal.md`. Check that file before moving modules around so new contributions stay consistent.
- Runtime orchestration currently lives under `src/modular_simulation/framework/`, while controllers, sensors and validation logic are grouped under their respective top-level packages (`control_system`, `usables`, `validation`).
- Examples live in `examples/` and mirror the public APIs. When updating APIs ensure the examples continue to import from the supported namespaces.

## Contribution expectations

- Prefer creating additional design notes under `docs/` when you introduce new subsystems. This keeps architectural decisions explicit.
- When modifying code in `src/modular_simulation`, add tests in `tests/` or the relevant example scenario to demonstrate the change where practical.
- Keep optional dependency imports (Numba, plotting backends) scoped to the modules that require them to avoid inflating import time for the rest of the package.
