### Simplified gas-phase system

```mermaid
classDiagram
    direction TB
    class Sensor_mass_rate {
        <<Sensor>>
        tag: "mass_prod_rate"
        unit: "kg/hour"
    }
    class Sensor_pressure {
        <<Sensor>>
        tag: "pressure"
        unit: "kPa"
    }
    class Sensor_yM1 {
        <<Sensor>>
        tag: "yM1"
        unit: ""
    }
    class Sensor_yM2 {
        <<Sensor>>
        tag: "yM2"
        unit: ""
    }
    class Sensor_F_cat {
        <<Sensor>>
        tag: "F_cat"
        unit: "kg/hour"
    }

    class Calc_FilterMassRate {
        <<FirstOrderFilter>>
        output: "filtered_mass_prod_rate"
    }
    class Calc_FilterPressure {
        <<FirstOrderFilter>>
        output: "filtered_pressure"
    }
    class Calc_MoleRatio {
        <<Calculation>>
        output: "rM2"
    }
    class Calc_PartialPressure {
        <<Calculation>>
        output: "pM1"
    }

    class Controller_PID_pM1 {
        <<PID>>
        mv_tag: "F_m1"
        cv_tag: "pM1"
    }
    class Controller_PID_rM2 {
        <<PID>>
        mv_tag: "F_m2"
        cv_tag: "rM2"
    }
    class Controller_PID_massRate {
        <<PID>>
        mv_tag: "F_cat"
        cv_tag: "filtered_mass_prod_rate"
    }
    class Controller_MV_F_cat {
        <<MVController>>
        mv_tag: "F_cat"
        cv_tag: "F_cat"
    }

    Calc_FilterMassRate "1" --> "1" Sensor_mass_rate : reads
    Calc_FilterPressure "1" --> "1" Sensor_pressure : reads
    Calc_MoleRatio "1" --> "1" Sensor_yM1 : reads
    Calc_MoleRatio "1" --> "1" Sensor_yM2 : reads
    Calc_PartialPressure "1" --> "1" Sensor_yM1 : reads
    Calc_PartialPressure "1" --> "1" Calc_FilterPressure : uses

    Controller_PID_pM1 "1" --> "1" Calc_PartialPressure : reads CV
    Controller_PID_rM2 "1" --> "1" Calc_MoleRatio : reads CV
    Controller_PID_massRate "1" --> "1" Calc_FilterMassRate : reads CV
    Controller_MV_F_cat "1" --> "1" Sensor_F_cat : reads CV

    Controller_MV_F_cat "1" o-- "1" Controller_PID_massRate : cascade
```

- Filters smooth the mass-rate and pressure sensors before downstream consumers rely on the derived tags.【F:new_examples/simplified_gas_phase_system/component_definition.py†L21-L45】
- The mole-ratio and partial-pressure calculations depend on analyzer readings and the filtered pressure, providing CVs for the monomer feed controllers.【F:new_examples/simplified_gas_phase_system/calculation_definition.py†L10-L41】【F:new_examples/simplified_gas_phase_system/component_definition.py†L46-L71】
- An MV controller supervises catalyst feed while cascading to a PID that targets filtered production rate, closing the loop around the derived calculation output.【F:new_examples/simplified_gas_phase_system/component_definition.py†L59-L77】