# CSTR with Energy Balance Example

This example extends the base irreversible reaction system with a coupled energy balance.
It demonstrates how to augment a mass balance model with temperature dynamics, a cooling
jacket, and Arrhenius kinetics inside the modular simulation framework.

## Key Features

* **Mass and Energy Dynamics** – The system tracks reactor volume, species concentrations,
  reactor temperature, and jacket temperature.
* **Arrhenius Kinetics** – The reaction rate follows an Arrhenius expression that depends on
  the reactor temperature.
* **Heat Exchange** – An overall heat transfer coefficient and area govern heat removal to a
  well-mixed cooling jacket that has its own energy balance.
* **Modular Structure** – Sensors, controllers, and constants are configured using the same
  abstractions as other examples, illustrating how additional physics can be incorporated
  without changing the simulation driver.

## Files

* `system_definitions.py` – Declares the data structures and differential equations for
  the readable and numba-accelerated systems.
* `run_simulation.py` – Configures the system, runs a demonstration simulation, and plots
  state and control trajectories.
* `analyze.py` – Provides a basic profiling harness and plotting utility.
* `test_time.py` – Quick timing harness for sensor evaluation (mirrors other examples).

Run `python run_simulation.py` from this directory to reproduce the plots.
