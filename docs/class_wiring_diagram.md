# Modular Simulation Class Wiring

The diagram below captures how the primary orchestration classes compose and depend on one another. It traces the flow from a top-level `Plant` through `System` orchestration, the measurable/usable quantity containers, and the instrumentation/control primitives that operate on tags.

```mermaid
classDiagram
    direction TB
    class Plant {
        systems: list[System]
        dt: float
        +step(nsteps=1)
    }
    class System {
        dt: Quantity
        +measurable_quantities
        +usable_quantities
        +step(duration)
        +controller_dictionary
        +cv_tag_list
    }
    class MeasurableQuantities {
        +constants: Constants
        +states: States
        +control_elements: ControlElements
        +algebraic_states: AlgebraicStates
        +tag_list()
        +tag_unit_info()
    }
    class MeasurableBase {
        +to_array()
        +update_from_array()
        +tag_list()
        +tag_unit_info()
    }
    class States
    class ControlElements
    class AlgebraicStates
    class Constants

    class UsableQuantities {
        +sensors: list[SensorBase]
        +calculations: list[CalculationBase]
        +controllers: list[ControllerBase]
        +update(t)
        +tag_list()
    }
    class SensorBase {
        measurement_tag: str
        alias_tag?: str
        +measure(t)
        +_initialize()
    }
    class CalculationBase {
        +calculate(t)
        +_initialize(tag_infos)
        +_calculation_algorithm()
    }
    class ControllerBase {
        mv_tag: str
        cv_tag: str
        +update(t)
        +change_control_mode(mode)
        +extend_setpoint()
    }
    class TagInfo {
        tag: str
        unit: UnitBase
        +data: TagData
        +history
    }
    class TagData {
        time: float
        value: float|NDArray
        ok: bool
    }

    Plant --> "1..*" System : aggregates
    Plant ..> System : builds composite

    System --> MeasurableQuantities : owns
    System --> UsableQuantities : owns

    MeasurableQuantities --> States
    MeasurableQuantities --> ControlElements
    MeasurableQuantities --> AlgebraicStates
    MeasurableQuantities --> Constants

    States --|> MeasurableBase
    ControlElements --|> MeasurableBase
    AlgebraicStates --|> MeasurableBase
    Constants --|> MeasurableBase

    UsableQuantities --> SensorBase : orchestrates
    UsableQuantities --> CalculationBase : orchestrates
    UsableQuantities --> ControllerBase : orchestrates
    UsableQuantities ..> MeasurableQuantities : resolves tags

    SensorBase --> TagInfo : publishes
    SensorBase --> TagData : emits samples
    SensorBase ..> MeasurableQuantities : reads values

    CalculationBase --> TagInfo : defines IO
    CalculationBase --> TagData : emits results

    ControllerBase --> TagInfo : SP/CV tags
    ControllerBase --> TagData : MV history
    ControllerBase ..> ControlElements : writes MV

    TagInfo --> TagData : stores latest
```

## Data flow highlights

1. **Plant aggregation** – `Plant` flattens measurable and usable artifacts from each constituent `System`, collecting state/control dictionaries, sensors, calculations, controllers, and solver callbacks before wrapping them in a composite system so the plant can advance every subsystem in lockstep.【F:src/modular_simulation/plant.py†L10-L111】
2. **System orchestration** – `System` owns the measurable/usable containers, validates tag wiring, constructs solver parameters, advances the integration loop, and refreshes algebraic states while invoking `UsableQuantities.update` before each solver step to run sensors, calculations, and controllers.【F:src/modular_simulation/framework/system.py†L28-L296】【F:src/modular_simulation/framework/system.py†L401-L555】
3. **Measurable containers** – `MeasurableQuantities` groups `States`, `ControlElements`, `AlgebraicStates`, and `Constants`, each inheriting array-indexing behavior from `MeasurableBase` to provide tag lists, unit metadata, and vector conversions for solver IO.【F:src/modular_simulation/measurables/measurable_quantities.py†L1-L72】【F:src/modular_simulation/measurables/measurable_base.py†L1-L118】
4. **Usable orchestration** – `UsableQuantities` enforces tag consistency, links sensors/calculations/controllers to tag metadata, initializes them against the system's measurables, and executes their `measure`, `calculate`, and `update` hooks every tick.【F:src/modular_simulation/usables/usable_quantities.py†L1-L185】
5. **Instrumentation primitives** – Sensors resolve measurement getters against measurables and stream `TagData` through `TagInfo`, calculations convert input units and produce outputs the same way, and controllers consume those tag infos to manage setpoints and manipulate control elements.【F:src/modular_simulation/usables/sensors/sensor_base.py†L1-L186】【F:src/modular_simulation/usables/calculations/calculation_base.py†L1-L183】【F:src/modular_simulation/usables/controllers/controller_base.py†L1-L195】【F:src/modular_simulation/usables/tag_info.py†L1-L55】
