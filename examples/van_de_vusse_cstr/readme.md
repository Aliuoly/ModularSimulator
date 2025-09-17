# Van de Vusse CSTR Example

This example reproduces the classic continuously stirred tank reactor with the Van de Vusse reaction network using the modular
simulation framework. The model tracks four species and an energy balance with a cooling jacket. A simple PI controller adjusts
the jacket heat duty to maintain the reactor temperature at a specified setpoint.

Run the simulation with:

```bash
python examples/van_de_vusse_cstr/run_simulation.py
```

The script compares the readable and fast (NumPy-oriented) implementations and plots the resulting trajectories.
