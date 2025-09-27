# Repository Guidance

This repository hosts the Modular Simulation framework. Use the following notes when navigating or extending the code base:


- Runtime orchestration currently lives under `src/modular_simulation/framework/`, while controllers, sensors and are grouped under their respective top-level packages (`control_system`, `usables`).
- Validation is done in the container class for the controllers, sensors, and calculations. These container classes are located in `quantities/`. Most validations use simple custom exceptions found in `validation`. 
- Examples live in `examples/`, showcasing different complexity simulations. The `analyze.py` scripts in each example can be used to benchmark framework performance. 
- Limited testing scripts using pytest live in `tests/` but is not exhaustive.
- A very simple visualization helper is located in `plotting/`. 
- Coding style - focus on simplicity and readability while not compromising on performance for hot loops and major bottlenecks. 