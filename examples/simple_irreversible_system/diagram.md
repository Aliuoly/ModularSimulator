### Simple irreversible system

```mermaid
classDiagram
    direction LR
    class Sensor_F_out {
        <<Sensor>>
        tag: "F_out"
        unit: "L/s"
    }
    class Sensor_F_in {
        <<Sensor>>
        tag: "F_in"
        unit: "L/minute"
    }
    class Sensor_B {
        <<Sensor>>
        tag: "B"
        unit: "mol/L"
    }
    class Sensor_V {
        <<Sensor>>
        tag: "V"
        unit: "L"
    }
    class Controller_PID_B {
        <<PID>>
        mv_tag: "F_in"
        cv_tag: "B"
    }

    Controller_PID_B "1" --> "1" Sensor_B : reads CV
```

- Four delayed sensors expose reactor outflow, inflow, concentration, and volume for downstream use.【F:new_examples/simple_irreversible_system/component_definition.py†L11-L34】
- A single PID loop manipulates `F_in` to regulate concentration `B`, consuming the sensor-provided CV tag.【F:new_examples/simple_irreversible_system/component_definition.py†L38-L49】