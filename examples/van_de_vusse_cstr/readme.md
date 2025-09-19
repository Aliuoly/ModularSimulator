# Van de Vusse CSTR Example

This example reproduces the classic continuously stirred tank reactor with the Van de Vusse reaction network using the modular
simulation framework. The model tracks the A → B conversion along with an energy balance that includes a cooling jacket. A simple
PI controller adjusts the jacket inlet temperature (with constant jacket flow) to maintain the reactor temperature at a specified
setpoint, while the instantaneous jacket heat duty is reported as a calculated quantity.

Run the simulation with:

```bash
python examples/van_de_vusse_cstr/run_simulation.py
```

The script compares the readable and fast (NumPy-oriented) implementations and plots the resulting trajectories.
