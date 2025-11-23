# Modular Simulation Framework

`modular_simulation` is a lightweight framework for building closed-loop process simulations. It separates the physical model, instrumentation, and control logic into composable pieces, allowing new systems to be assembled from small, well-typed components.

The framework is intended to be used for learning about controller design, trying out different controller algorithms, and performing system identification.

## Key Features

*   **Modular Design**: Easily compose systems using pre-built or custom Sensors, Controllers, and Process Models.
*   **DAE Support**: Implements a workaround to allow `scipy.integrate.solve_ivp` (which typically handles ODEs) to solve Differential-Algebraic Equation (DAE) systems by recomputing algebraic variables at every solver step.
*   **Type Safety**: Built on Pydantic and fully typed Python to ensure correctness and provide excellent IDE support.
*   **Unit-Aware**: Utilizes `astropy` for unit management. However, process governing equations are not unit-validated.  

## Available Components

The framework comes with a set of pre-built components to get you started:

*   **Controllers** (`modular_simulation.usables.control_system.controllers`):
    *   `PIDController`: Standard PID algorithm with anti-windup, setpoint weighting, and derivative filtering.
    *   `BangBangController`: Simple on/off control with deadband.
    *   `InternalModelController`: Model-based control using an internal process model.
    *   `FirstOrderTrajectoryController`: Computes MV to drive CV along a desired first-order trajectory.
    *   `MVController`: Pass-through controller that sets MV directly to the setpoint value.
*   **Sensors** (`modular_simulation.usables.sensors`):
    *   `SampledDelayedSensor`: Simulates discrete sampling intervals and measurement delays. Also supports Gaussian noise injection and random fault simulation (drift, freeze, bias).

## Installation

Requires Python 3.13+

```bash
# Using pip
pip install .

# Using uv (recommended)
uv sync
```

## Quick Start

The project includes several example systems in the `examples/` directory. To run the simplified gas phase system:

```bash
python examples/simplified_gas_phase_system/run_simulation.py
```

This will execute the simulation and generate plots and logs demonstrating the system's behavior.

## Architecture & Interactions

The framework orchestrates interactions between the **System** (simulation engine), **Process Model** (physics/chemistry), **Sensors**, **Calculations**, and **Controllers**.

### Interaction Diagrams

We have detailed diagrams illustrating how these components interact:

*   **[Initialization Phase](docs/initialization.mermaid)**: Shows how components are added and wired up before the simulation starts.
*   **[Simulation Step](docs/simulation_step.mermaid)**: Details the order of operations during each time step (Sensors -> Calculations -> Controls -> Process).
*   **[Control Loop Example](docs/control_loop_example.mermaid)**: A specific example of a cascade control loop (Valve -> Flow Controller -> Level Controller).
*   **[Detailed Control Interaction](docs/control_loop_interaction.mermaid)**: In-depth look at the request/response flow between Control Elements, Mode Managers, and Controllers.

### Core Components

1.  **Measurable Quantities**: The state of the world is defined in a single **`ProcessModel`** subclass.
    *   **Definition**: Users must subclass `ProcessModel` and implement two abstract methods:
        *   `differential_rhs`: Defines the system's differential equations (dx/dt).
        *   `calculate_algebraic_values`: Computes algebraic variables based on current state.
    *   **Fields**: Class attributes are annotated using **`StateMetadata`** to define their role and units.
        *   `DIFFERENTIAL`: Differential variables (integrated over time).
        *   `ALGEBRAIC`: Variables computed instantly from States.
        *   `CONTROLLED`: Actuators manipulated by control elements.
        *   `CONSTANT`: Constant parameters, such as dimensions, rate constants, etc.

2.  **Usable Quantities**:
    *   `Sensors`: Read from ProcessModel fields and add noise/delays.
    *   `Calculations`: Derive calculated values from Sensor data or other calculation results. Define inputs and outputs using **`TagMetadata`**.

    ```python
    class MyCalculation(CalculationBase):
        input_tag: Annotated[str, TagMetadata(TagType.INPUT, unit="m")]
        output_tag: Annotated[str, TagMetadata(TagType.OUTPUT, unit="m")]
    ```

3.  **Control System**:
    *   `ControlElements`: Receive actions from Controllers and update the Process Model.
    *   `Controllers`: Used current values of controlled variable (CV) and setpoint (SP) to calculate control action, which is sent to the ControlElement/inner loop controllers.

## Project Structure

*   `src/modular_simulation`: Core framework code.
    *   `framework`: System orchestration and time stepping.
    *   `usables`: Sensors, Calculations, and Control components.
    *   `quantities`: Data structures for process state.
*   `examples`: Ready-to-run example simulations.
*   `docs`: Mermaid diagrams and documentation.
