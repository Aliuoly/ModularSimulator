# Modular Simulation Class Wiring

The diagram below captures how the primary orchestration classes compose and depend on one another. It traces the flow from a top-level `Plant` down through `System` orchestration, the measurable/usable quantity containers, and the instrumentation/controller primitives that operate on tags.

```mermaid
classDiagram
    direction TB
    class Plant {
        systems: List[System]
        dt: float
        +step(nsteps=1)
    }
    class System {
        dt: Quantity
        +measurable_quantities
        +usable_quantities
        +step(duration)
        +extend_controller_trajectory()
        +controller_dictionary
    }
    class MeasurableQuantities {
        +constants: Constants
        +states: States
        +control_elements: ControlElements
        +algebraic_states: AlgebraicStates
        +tag_list()
        +tag_unit_info()
    }
    class UsableQuantities {
        +sensors: List[Sensor]
        +calculations: List[Calculation]
        +controllers: List[Controller]
        +update(t)
        +tag_list()
    }
    class BaseIndexedModel {
        +to_array()
        +update_from_array()
        +tag_list()
    }
    class States
    class ControlElements
    class AlgebraicStates
    class Constants

    class Sensor {
        measurement_tag: str
        alias_tag?: str
        +measure(t)
    }
    class Calculation {
        +calculate(t)
        +_calculation_algorithm()
    }
    class Controller {
        mv_tag: str
        cv_tag: str
        +update(t)
        +change_control_mode(mode)
    }
    class TagInfo {
        tag: str
        unit: Unit
        +data: TagData
        +history
    }
    class TagData {
        time: float
        value: float|NDArray
        ok: bool
    }

    Plant --> "1..*" System : aggregates
    Plant ..> System : builds CompositeSystem

    System --> MeasurableQuantities : owns
    System --> UsableQuantities : owns
    System --> "controls" Controller : via usable_quantities
    System --> BaseIndexedModel : uses state arrays

    MeasurableQuantities --> States
    MeasurableQuantities --> ControlElements
    MeasurableQuantities --> AlgebraicStates
    MeasurableQuantities --> Constants

    States --|> BaseIndexedModel
    ControlElements --|> BaseIndexedModel
    AlgebraicStates --|> BaseIndexedModel
    Constants --|> BaseIndexedModel

    UsableQuantities --> Sensor : orchestrates
    UsableQuantities --> Calculation : orchestrates
    UsableQuantities --> Controller : orchestrates
    UsableQuantities --> MeasurableQuantities : resolves tags

    Sensor --> TagInfo : publishes
    Sensor --> TagData : emits samples
    Calculation --> TagInfo : defines outputs
    Calculation --> TagData : emits results
    Controller --> TagInfo : SP/CV tags
    Controller --> TagData : MV history
    TagInfo --> TagData : stores latest

    Controller ..> ControlElements : writes MV
    Sensor ..> BaseIndexedModel : reads measurement
    Calculation ..> TagInfo : converts inputs
```

## Data flow highlights

1. **Plant aggregation** – A `Plant` flattens the measurable, usable, and controllable artifacts of each constituent `System`, wrapping them into a composite system so the plant can advance every subsystem in lockstep.【F:src/modular_simulation/plant.py†L10-L111】
2. **System orchestration** – `System` owns the measurable/usable containers, refreshes algebraic states, and advances the solver while letting `UsableQuantities` drive sensor measurements, calculations, and controller updates each step.【F:src/modular_simulation/framework/system.py†L24-L382】
3. **Measurable containers** – `MeasurableQuantities` groups `States`, `ControlElements`, `AlgebraicStates`, and `Constants`, each inheriting array-indexing behaviors from `BaseIndexedModel` to enable efficient solver IO.【F:src/modular_simulation/measurables/measurable_quantities.py†L11-L78】【F:src/modular_simulation/measurables/base_classes.py†L8-L118】
4. **Usable orchestration** – `UsableQuantities` initializes and updates `Sensor`, `Calculation`, and `Controller` instances, ensuring tags resolve to measurable data and propagating updates during each simulation tick.【F:src/modular_simulation/usables/usable_quantities.py†L17-L208】
5. **Instrumentation primitives** – Sensors, calculations, and controllers share the `TagInfo`/`TagData` tagging scheme: sensors and calculations publish `TagData` through their `TagInfo`, while controllers coordinate setpoints, cascade relationships, and manipulate control elements via tag-aware getters and setters.【F:src/modular_simulation/usables/sensors/sensor.py†L18-L164】【F:src/modular_simulation/usables/calculations/calculation.py†L52-L182】【F:src/modular_simulation/usables/controllers/controller.py†L62-L405】【F:src/modular_simulation/usables/tag_info.py†L9-L55】

