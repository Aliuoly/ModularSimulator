# Simulation of a CSTR using a Modular Framework

This project presents a dynamic simulation of a Continuous Stirred-Tank Reactor (CSTR) where a simple, irreversible reaction takes place. It is designed as a clear example of how to model and simulate a chemical process using the `modular_simulation` Python package.

The core philosophy of the `modular_simulation` package is to separate the distinct components of a system—its physical state, control elements, sensors, and governing equations—into clean, interchangeable modules. This approach enhances readability, maintainability, and simplifies the process of building complex simulations.

## The CSTR Mathematical Model

The simulation models a liquid-phase, irreversible reaction within a CSTR:

$$A \rightarrow 2B$$

The reaction rate is first-order with respect to the concentration of reactant A. The system is described by a set of Differential-Algebraic Equations (DAEs), which are solved simultaneously.

### Governing Equations

1.  **Volume Balance (Differential Equation):** The change in reactor volume over time is the difference between the inlet and outlet flow rates.
    $$\frac{dV}{dt} = F_{in} - F_{out}$$

2.  **Outlet Flow (Algebraic Equation):** The outlet flow is modeled as gravity-driven, where the flow rate is proportional to the square root of the liquid volume. This algebraic relationship links the outlet flow directly to the current state of the system.
    $$F_{out} = C_v \sqrt{V}$$
    Here, $C\_v$ is the valve coefficient.

3.  **Mole Balances (Differential Equations):** The change in concentration for each component is derived from the general mole balance equation. These equations account for material entering and leaving the reactor, the consumption or generation from the reaction, and the effect of the changing volume. The reaction rate is defined as $r = k[A]V$.

      * **For reactant A:**
        $$\frac{d[A]}{dt} = \frac{1}{V} \left( F_{in}[A]_{in} - F_{out}[A] - r - [A]\frac{dV}{dt} \right)$$

      * **For product B:**
        $$\frac{d[B]}{dt} = \frac{1}{V} \left( -F_{out}[B] + 2r - [B]\frac{dV}{dt} \right)$$

## Implementation with `modular_simulation`

The power of the `modular_simulation` package is in how cleanly the mathematical model translates into code. Each conceptual part of the system has a corresponding class.

### 1\. Defining the System's quantities

First, we define the data structures for all measurable variables using Pydantic models. This provides strong typing and validation.

  * `IrreversableStates`: Holds the differential state variables (`V`, `A`, `B`).
  * `IrreversableControlElements`: Holds the variables that can be externally manipulated (`F_in`).
  * `IrreversableAlgebraicStates`: Holds variables that are calculated from other states (`F_out`).

These are then collected into a single `MeasurableQuantities` object that represents the complete state of the system at any given time.

### 2\. Defining the System Dynamics

The governing equations are implemented in a class that inherits from `System`. This class provides the core mathematical logic to the simulation engine.

  * `_calculate_algebraic_values()`: This method implements the algebraic equation(s). In this case, it calculates `F_out` from the current volume `V`.
  * `rhs()`: This method implements the "right-hand side" of the differential equations, calculating the derivatives (`dV/dt`, `d[A]/dt`, `d[B]/dt`).

### 3\. Defining Instrumentation and Control

Sensors and controllers are defined as separate classes, promoting reusability.

  * **Sensors** (e.g., `VolumeSensor`, `BConcentrationSensor`) define how to "measure" a value from the `MeasurableQuantities` object.
  * **Controllers** (e.g., `PIDController`) take measurements from sensors and calculate an output to manipulate a control element.

These are assembled into `UsableQuantities` (the collection of sensors) and `ControllableQuantities` (the collection of controllers).

### 4\. Assembling the Final Simulation

The final simulation object is created by passing all the modular components—quantities, controllers, constants, and solver options—to the constructor of our `IrreversableSystem` class. This object then handles the entire simulation loop, including the control logic and numerical integration.

## Performance Considerations

This project also includes a `fast_version` which demonstrates the flexibility of the framework. By inheriting from `FastSystem` instead of `System`, the core mathematical functions (`rhs` and `_calculate_algebraic_values`) can be replaced with versions optimized with Numba for high performance. This swap is achieved without altering the structure of the controllers, sensors, or the main simulation loop, showcasing the modularity of the design.

## How to Run the Project

1.  **Run a simulation:**

      * To see the readable version in action: `python readable_version/system_simulation.py`
      * To see the fast version in action: `python fast_version/system_simulation.py`

2.  **Analyze the profiler results:**

    ```
    python readable_version/analyze_performance.py
    python fast_version/analyze_performance.py
    ```

    This utility script loads the `.prof` files and prints a summary of where the program spends its time.