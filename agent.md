# Repository Guidance

This repository hosts the Modular Simulation framework. Use the following notes when navigating or extending the code base:


- Runtime orchestration currently lives under `src/modular_simulation/framework/`
- runtime consist of a "System", found under `src/modular_simulation/framework`, which uses 2 "quantity" classes: MeasurableQuantities and UsableQuantities. MeasurableQuantities hold the underlying system states that are "measurable" in a physical sense, while UsableQuantities hold the components a virtual operator of the System may use to interact with the underlying MeasurableQuantities; UsableQuantities, hence, holds the sensors that measure the underying system states, calculations that process these sensors, and controller that apply control actions to the system. 
- System validation is partly done in the MeasurableQuantities and UsableQuantities themselves, and validation logic that involves the interaction between the 2 quantities is done in the System class itself. 
- Examples live in `examples/`, showcasing different complexity simulations. The `analyze.py` scripts in each example can be used to benchmark framework performance. Currently, only the gas phase example works. 
- Limited testing scripts using pytest live in `tests/` but is not exhaustive.
- A very simple visualization helper is located in `src/modular_simulation/plotting/`. 
- GUI, living under `src/modular_simulation/app/`, will be added using NiceGUI library - though is not ready yet. 
- `plant.py` is a work in progress; same with the simple plant example. Do not worry about these. 
- ALWAYS test the code before submitting work. 