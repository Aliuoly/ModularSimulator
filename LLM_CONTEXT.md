# modular_simulation Context for Agentic LLMs

> [!IMPORTANT]
> This document is designed to provide high-level context and architectural understanding of the `modular_simulation` framework. Use this as a map when exploring the codebase.

## 1. Architectural Overview

`modular_simulation` is a Python framework for simulating closed-loop process control systems. It is designed to solve Differential-Algebraic Equations (DAEs) using standard ODE solvers by decoupling the differential and algebraic states.

### The Simulation Loop
The core loop is managed by the `System` class in `src/modular_simulation/framework/system.py`.

1.  **Time Step (`System.step(duration)`)**:
    *   The user requests a step of a certain duration (e.g., 10 seconds).
    *   The system divides this into logical steps of size `dt`.
    *   Inside the loop, it calls `_update_components()` and then `ProcessModel.step()`.

2.  **Component Update (`System._update_components()`)**:
    *   **Sensors**: Read current process state, add noise/delays.
    *   **Calculations**: Compute derived values from sensors.
    *   **Control Elements**:
        *   Controllers calculate new Manipulated Variable (MV) values.
        *   Control Elements write these MVs to the `ProcessModel` or other targets.

3.  **Process Integration (`ProcessModel.step()`)**:
    *   Uses `scipy.integrate.solve_ivp` to solve the system from `t` to `t + dt`.
    *   **The DAE Trick**: The `ProcessModel` acts as a pure ODE system to the solver (`dy/dt = f(t, y)`). However, strictly algebraic variables are recomputed *inside* the RHS function wrapper (`_rhs_wrapper`) at every internal solver step. This ensures algebraic constraints are satisfied throughout the integration interval.

### Data Flow
`ProcessModel` (Truth) -> `Sensors` (Measured) -> `Calculations` (Derived) -> `Controllers` (Decision) -> `ControlElements` (Action) -> `ProcessModel` (Update)

---

## 2. Core Components

### 2.1 ProcessModel (`src/modular_simulation/measurables/process_model.py`)
The physics/chemistry engine. Users must subclass `ProcessModel`.

*   **State Definition**: All class attributes must be valid state variables.
    *   They are auto-discovered during `model_post_init` using `StateMetadata`.
    *   **`StateMetadata(type, unit, description)`**:
        *   `StateType.DIFFERENTIAL`: Variables with time derivatives ($dx/dt$).
        *   `StateType.ALGEBRAIC`: Variables defined by algebraic equations ($0 = g(x, z)$).
        *   `StateType.CONTROLLED`: External inputs/actuators (MVs).
        *   `StateType.CONSTANT`: Fixed parameters.

*   **Abstract Methods**:
    *   `differential_rhs(...)`: Returns array of $dy/dt$. **Must be stateless.**
    *   `calculate_algebraic_values(...)`: Returns array of algebraic variables. **Must be stateless.**

*   **Views**: `differential_view`, `algebraic_view`, etc., provide efficient array-based access to the underlying scalar attributes.

### 2.2 System (`src/modular_simulation/framework/system.py`)
 The simulation orchestrator.

*   **Responsibility**: Holds the `ProcessModel`, list of `Sensors`, `Calculations`, and `ControlElements`.
*   **Initialization**: `System(process_model=..., sensors=..., ...)`
    *   Recursively validation configuration.
    *   Resolves dependencies (wiring sensors to calculations, calculations to controllers).
    *   Populates the `PointRegistry` (accessed via `system.point_registry`) with all accessible data points.

### 2.3 ControlElement (`src/modular_simulation/components/control_system/control_element.py`)
Represents the physical actuator (valve, pump, heater).

*   **Fields**:
    *   `mv_tag`: The name of the variable it manipulates.
    *   `mv_trajectory`: A `Trajectory` object for manual control.
    *   `controller`: An optional `AbstractController` (e.g., PID) that drives this element in AUTO mode.
*   **Modes**:
    *   `MANUAL`: Follows `mv_trajectory`.
    *   `AUTO`: Follows `controller` output.
*   **Binding**: Can bind directly to a `ProcessModel` attribute (updating the attribute directly while keeping `PointRegistry` in sync) or to a pure `PointRegistry` point.

### 2.4 AbstractSensor (`src/modular_simulation/components/sensors/abstract_sensor.py`)
Reads truth from `ProcessModel` and reports it to the system.

*   **Fields**:
    *   `measurement_tag`: The raw State name in `ProcessModel`.
    *   `alias_tag`: The public name in the simulation (e.g., 'TI-101').
    *   `faulty_probability`, `coefficient_of_variance`: For Simulation realism.

### 2.5 AbstractCalculation (`src/modular_simulation/components/calculations/abstract_calculation.py`)
Performs math on sensors.

*   **Usage**: Subclass and annotate inputs/outputs with `PointMetadata`.
    *   `TagType.INPUT`: Reads from `PointRegistry`.
    *   `TagType.OUTPUT`: Writes to `PointRegistry`.
*   **Wiring**: The keys in your `calculations` dict in the system definition map to these inputs.

## 3. The Type System & Units

The framework is heavily typed.

*   **`StateValue`**: Union types for scalars (float/int) or numpy arrays.
*   **`Seconds`**: Type alias for float, indicating usage.
*   **Units**: Uses `astropy.units`.
    *   `ProcessModel` states define their *intrinsic* units.
    *   `Sensors` can request `unit="deg_C"` even if the process model uses `K`. The framework handles conversion automatically during data access.
    *   **Warning**: The DAE/ODE equations (`differential_rhs`) are currently **unit-naive**. You must perform calculations in consistent units (usually SI) manually within these methods.

## 4. Coding Patterns & Cookbook

### How to Create a New Process Model

```python
from modular_simulation.measurables.process_model import ProcessModel, StateMetadata, StateType
from typing import Annotated

class MyReactor(ProcessModel):
    # Use Annotated[type, StateMetadata] pattern
    
    # 1. Differential Parameter
    T: Annotated[
        float, 
        StateMetadata(StateType.DIFFERENTIAL, unit="K", description="Reactor Temp")
    ] = 300.0

    # 2. Controlled Parameter (MV)
    flow_in: Annotated[
        float,
        StateMetadata(StateType.CONTROLLED, unit="L/s", description="Inlet Flow")
    ] = 10.0

    # ... Implement abstract methods ...
```

### How to Create a Calculation

```python
from modular_simulation.components.calculations.abstract_calculation import AbstractCalculation
from modular_simulation.components.calculations.point_metadata import PointMetadata, TagType
from typing import Annotated

class MyCalculation(AbstractCalculation):
    # Define Inputs
    input_tag: Annotated[
        str, 
        PointMetadata(TagType.INPUT, unit="m", description="Input Level")
    ]
    
    # Define Outputs
    output_tag: Annotated[
        str,
        PointMetadata(TagType.OUTPUT, unit="ft", description="Output Level (Converted)")
    ]

    def _calculation_algorithm(self, t, inputs_dict):
        # inputs_dict keys matches the field names (e.g. 'input_tag')
        val_m = inputs_dict["input_tag"] 
        val_ft = val_m * 3.28084
        return {"output_tag": val_ft}
```

### How to Implement `differential_rhs`
**Crucial**: The function signature receives *flat arrays* and *mapping dictionaries*. Do not access `self.variable` inside this method; it is static/stateless for performance and pickling.

```python
    @override
    @staticmethod
    def differential_rhs(
        t, y, u, k, algebraic, y_map, u_map, k_map, algebraic_map
    ) -> NDArray[np.float64]:
        dydt = np.zeros_like(y)
        
        # Access variables using maps
        Temp = y[y_map["T"]]
        Flow = u[u_map["flow_in"]]
        
        # Physics
        dT_dt = ... 
        
        dydt[y_map["T"]] = dT_dt
        return dydt
```

### Reference vs. Alias in Tags
*   **Raw Tags**: Property names in `ProcessModel` (e.g., `T`, `pressure`, `concentration_A`).
*   **System Tags**: Auto-generated by System (e.g., `(raw, DIFFERENTIAL)_T`).
*   **Sensor/Alias Tags**: User-defined names (e.g., `TI-100`, `PC-101`).
*   **Best Practice**: Always perform lookups in the `PointRegistry` (`system.point_registry.get("tag_name")`) which handles resolution.

## 5. Directory Map

*   `src/modular_simulation/`
    *   `framework/`: `System` loop, time stepping.
    *   `measurables/`: `ProcessModel` base classes.
    *   `components/`:
        *   `control_system/`: `ControlElement`, `PIDController`, `Trajectory`.
        *   `sensors/`: `SampledDelayedSensor`.
        *   `calculations/`: logic blocks.
        *   `point/`: `Point` and `DataValue` (data containers).
    *   `utils/`: Helper functions.

## 6. Testing & Development
*   **`examples/`**: Contains complete, runnable systems. Use these as golden reference patterns.
*   **Unit Tests**: Standard `pytest` suite.
*   **Debugging**: The system relies on `logging`. Use `logging.getLogger(__name__)`.

