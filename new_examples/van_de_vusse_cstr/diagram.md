### Van de Vusse CSTR

```mermaid
classDiagram
    direction TB
    class Sensor_Ca {
        <<Sensor>>
        tag: "Ca"
        unit: "mol/L"
    }
    class Sensor_Cb {
        <<Sensor>>
        tag: "Cb"
        unit: "mol/L"
    }
    class Sensor_T_reactor {
        <<Sensor>>
        tag: "T"
        unit: "deg_C"
    }
    class Sensor_Tk {
        <<Sensor>>
        tag: "Tk"
        unit: "deg_C"
    }
    class Sensor_Tj_in {
        <<Sensor>>
        tag: "Tj_in"
        unit: "deg_C"
    }

    class Calc_HeatDuty {
        <<Calculation>>
        output: "Qk"
    }

    class Controller_PID_Tj_in {
        <<PID>>
        mv_tag: "Tj_in"
        cv_tag: "T"
    }
    class Controller_PID_T_inner {
        <<PID>>
        mv_tag: "T"
        cv_tag: "Cb"
    }

    Calc_HeatDuty "1" --> "1" Sensor_Tk : reads
    Calc_HeatDuty "1" --> "1" Sensor_T_reactor : reads

    Controller_PID_Tj_in "1" --> "1" Sensor_T_reactor : reads CV
    Controller_PID_T_inner "1" --> "1" Sensor_Cb : reads CV

    Controller_PID_Tj_in "1" o-- "1" Controller_PID_T_inner : cascade
```

- Sampled sensors capture concentrations and temperatures for both the reactor and jacket inlet, supplying the calculation and controller CVs.【F:new_examples/van_de_vusse_cstr/component_definition.py†L14-L37】
- The heat-duty calculation computes jacket energy transfer from reactor and jacket temperatures, while an outer PID on `T` cascades to an inner loop that shapes `Cb` via the reactor temperature manipulation.【F:new_examples/van_de_vusse_cstr/calculation_definition.py†L1-L35】【F:new_examples/van_de_vusse_cstr/component_definition.py†L39-L55】