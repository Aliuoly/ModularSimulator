### Simple irreversible system with energy balance

```mermaid
classDiagram
    direction TB
    class Sensor_B2 {
        <<Sensor>>
        tag: "B"
        unit: "mol/L"
    }
    class Sensor_V2 {
        <<Sensor>>
        tag: "V"
        unit: "L"
    }
    class Sensor_T {
        <<Sensor>>
        tag: "T"
        unit: "K"
    }
    class Sensor_TJ {
        <<Sensor>>
        tag: "T_J"
        unit: "K"
    }
    class Controller_PID_V {
        <<PID>>
        mv_tag: "F_in"
        cv_tag: "V"
    }
    class Controller_PID_TJ {
        <<PID>>
        mv_tag: "T_J_in"
        cv_tag: "T_J"
    }
    class Controller_PID_T {
        <<PID>>
        mv_tag: "T_J"
        cv_tag: "T"
    }
    class Controller_PID_B2 {
        <<PID>>
        mv_tag: "T"
        cv_tag: "B"
    }

    Controller_PID_V "1" --> "1" Sensor_V2 : reads CV
    Controller_PID_TJ "1" --> "1" Sensor_TJ : reads CV
    Controller_PID_T "1" --> "1" Sensor_T : reads CV
    Controller_PID_B2 "1" --> "1" Sensor_B2 : reads CV

    Controller_PID_TJ "1" o-- "1" Controller_PID_T : cascade
    Controller_PID_T "1" o-- "1" Controller_PID_B2 : cascade
```

- Eight sensors provide flow, concentration, temperature, and jacket measurements, enabling multi-loop control and diagnostics.【F:new_examples/simple_irreversible_energy_balance_system/component_definition.py†L10-L31】
- The volume controller directly regulates `V`, while the jacket temperature loop cascades through reactor temperature and concentration controllers, forming a three-tier cascade tree for thermal management.【F:new_examples/simple_irreversible_energy_balance_system/component_definition.py†L35-L61】

