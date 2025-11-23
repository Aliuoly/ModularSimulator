# Modular Simulation Class Wiring (Refactored)

The diagrams in this document capture how the refactored framework wires process models, instrumentation, calculations, and controllers. The focus of the framework overview is the reusable infrastructure, while the example sections highlight calculation and control dependencies unique to each application.

## Framework overview

```mermaid
classDiagram
    direction TB
    class System {
        dt: Seconds
        process_model: ProcessModel
        sensors: list[SensorBase]
        calculations: list[CalculationBase]
        controllers: list[ControllerBase]
        +add_component(component)
        +step(duration)
        +tag_info_dict
    }

    class ProcessModel {
        t: Seconds
        +state_metadata_dict
        +differential_view
        +algebraic_view
        +controlled_view
        +constant_view
        +step(dt)
        +make_converted_getter(name, unit)
        +make_converted_setter(name, unit)
    }

    class CategorizedStateView {
        model: ProcessModel
        state_type: StateType
        +to_array()
        +update_from_array()
        +state_list
        +array_size
    }

    class StateMetadata {
        type: StateType
        unit: SerializableUnit
        description: str
    }

    class SensorBase {
        measurement_tag: str
        alias_tag: str
        unit: UnitBase
        +commission(system)
        +measure(t)
    }

    class CalculationBase {
        name: str
        +wire_inputs(system)
        +calculate(t)
        +retrieve_specific_input(tag)
    }

    class ControllerBase {
        mv_tag: str
        cv_tag: str
        mode: ControllerMode
        cascade_controller: ControllerBase?
        +commission(system)
        +update(t)
    }

    class TagInfo {
        tag: str
        unit: UnitBase
        description: str
        +data: TagData
        +history: list[TagData]
    }

    class TagData {
        time: Seconds
        value: StateValue
        ok: bool
    }

    System "1" *-- "1" ProcessModel : orchestrates
    System "1" o-- "*" SensorBase : schedules
    System "1" o-- "*" CalculationBase : wires
    System "1" o-- "*" ControllerBase : coordinates
    System "1" *-- "*" TagInfo : registry

    ProcessModel "1" *-- "4" CategorizedStateView : views
    ProcessModel "1" o-- "*" StateMetadata : describes

    CategorizedStateView "1" --> "1" ProcessModel : references

    SensorBase "1" *-- "1" TagInfo : publishes
    SensorBase ..> ProcessModel : reads states

    CalculationBase "1" *-- "*" TagInfo : outputs
    CalculationBase "1" ..> "*" TagInfo : reads inputs

    ControllerBase "1" *-- "1" TagInfo : setpoint
    ControllerBase "1" ..> "*" TagInfo : uses PV
    ControllerBase "1" o-- "0..1" ControllerBase : cascade
    ControllerBase ..> ProcessModel : writes MV

    TagInfo "1" *-- "1" TagData : current sample
```

### Implementation notes

- `System` validates usable components, registers their tag metadata, and advances the coupled process model while running sensors, calculations, and controllers each step.【F:src/modular_simulation/framework/system.py†L24-L212】【F:src/modular_simulation/framework/system.py†L268-L352】
- `ProcessModel` centralizes state metadata, exposes category-specific views for differential/algebraic/controlled/constant states, and handles solver integration plus unit conversions for getters and setters.【F:src/modular_simulation/measurables/process_model.py†L26-L201】【F:src/modular_simulation/measurables/process_model.py†L320-L409】
- `SensorBase` commissions against the process model, resolves measurement getters, and historizes samples on its dedicated `TagInfo` instance.【F:src/modular_simulation/usables/sensors/sensor_base.py†L38-L189】【F:src/modular_simulation/usables/sensors/sensor_base.py†L221-L304】
- `CalculationBase` extracts annotated tag metadata, wires inputs from the system registry, and emits outputs via managed `TagInfo` records.【F:src/modular_simulation/usables/calculations/calculation_base.py†L37-L126】【F:src/modular_simulation/usables/calculations/calculation_base.py†L150-L240】
- `ControllerBase` wires MV setters and CV getters, establishes cascade relationships, and stores setpoint histories in its own `TagInfo` while updating manipulated variables on the process model.【F:src/modular_simulation/usables/controllers/controller_base.py†L49-L213】【F:src/modular_simulation/usables/controllers/controller_base.py†L260-L359】
- `TagInfo` and `TagData` capture the latest value and historized samples for every usable tag, providing uniform access across sensors, calculations, and controllers.【F:src/modular_simulation/usables/tag_info.py†L1-L71】
