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

## Example dependency diagrams

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

### Gas-phase polymerization plant

```mermaid
classDiagram
    direction TB
    class Sensor_yM1_full {
        <<Sensor>>
        tag: "yM1"
    }
    class Sensor_yM2_full {
        <<Sensor>>
        tag: "yM2"
    }
    class Sensor_yH2_full {
        <<Sensor>>
        tag: "yH2"
    }
    class Sensor_pressure_full {
        <<Sensor>>
        tag: "pressure"
    }
    class Sensor_mass_rate_full {
        <<Sensor>>
        tag: "mass_prod_rate"
    }
    class Sensor_bed_weight {
        <<Sensor>>
        tag: "bed_weight"
    }
    class Sensor_F_cat_full {
        <<Sensor>>
        tag: "F_cat"
    }
    class Sensor_F_teal {
        <<Sensor>>
        tag: "F_teal"
    }
    class Sensor_lab_MI {
        <<Sensor>>
        tag: "lab_MI"
    }
    class Sensor_lab_density {
        <<Sensor>>
        tag: "lab_density"
    }
    class Sensor_bed_level {
        <<Sensor>>
        tag: "bed_level"
    }

    class Calc_FilterMassRate_full {
        <<FirstOrderFilter>>
        output: "filtered_mass_prod_rate"
    }
    class Calc_FilterPressure_full {
        <<FirstOrderFilter>>
        output: "filtered_pressure"
    }
    class Calc_MoleRatio_full {
        <<Calculation>>
        outputs: "rM2", "rH2"
    }
    class Calc_PartialPressure_full {
        <<Calculation>>
        output: "pM1"
    }
    class Calc_ResidenceTime {
        <<Calculation>>
        output: "residence_time"
    }
    class Calc_CatInventory {
        <<Calculation>>
        output: "cat_inventory"
    }
    class Calc_AlTiRatio {
        <<Calculation>>
        output: "Al_Ti_ratio"
    }
    class Calc_PropertyEstimator {
        <<Calculation>>
        outputs: "inst_MI", "cumm_MI", "inst_density", "cumm_density"
    }

    class Controller_IMC_cat_inventory {
        <<IMC>>
        mv_tag: "F_cat"
        cv_tag: "cat_inventory"
    }
    class Controller_PID_massRate_full {
        <<PID>>
        mv_tag: "cat_inventory"
        cv_tag: "filtered_mass_prod_rate"
    }
    class Controller_PID_pM1_full {
        <<PID>>
        mv_tag: "F_m1"
        cv_tag: "pM1"
    }
    class Controller_PID_rM2_full {
        <<PID>>
        mv_tag: "F_m2"
        cv_tag: "rM2"
    }
    class Controller_IMC_density {
        <<IMC>>
        mv_tag: "rM2"
        cv_tag: "inst_density"
    }
    class Controller_FO_density {
        <<FirstOrderTrajectory>>
        mv_tag: "inst_density"
        cv_tag: "cumm_density"
    }
    class Controller_PID_rH2_full {
        <<PID>>
        mv_tag: "F_h2"
        cv_tag: "rH2"
    }
    class Controller_IMC_MI {
        <<IMC>>
        mv_tag: "rH2"
        cv_tag: "inst_MI"
    }
    class Controller_FO_MI {
        <<FirstOrderTrajectory>>
        mv_tag: "inst_MI"
        cv_tag: "cumm_MI"
    }
    class Controller_IMC_AlTi {
        <<IMC>>
        mv_tag: "F_teal"
        cv_tag: "Al_Ti_ratio"
    }
    class Controller_PID_F_n2 {
        <<PID>>
        mv_tag: "F_n2"
        cv_tag: "filtered_pressure"
    }
    class Controller_PID_F_vent {
        <<PID>>
        mv_tag: "F_vent"
        cv_tag: "filtered_pressure"
    }
    class Controller_BangBang_discharge {
        <<BangBang>>
        mv_tag: "discharge_valve_position"
        cv_tag: "bed_level"
    }

    Calc_FilterMassRate_full "1" --> "1" Sensor_mass_rate_full : reads
    Calc_FilterPressure_full "1" --> "1" Sensor_pressure_full : reads
    Calc_MoleRatio_full "1" --> "1" Sensor_yM1_full : reads
    Calc_MoleRatio_full "1" --> "1" Sensor_yM2_full : reads
    Calc_MoleRatio_full "1" --> "1" Sensor_yH2_full : reads
    Calc_PartialPressure_full "1" --> "1" Sensor_yM1_full : reads
    Calc_PartialPressure_full "1" --> "1" Calc_FilterPressure_full : uses
    Calc_ResidenceTime "1" --> "1" Calc_FilterMassRate_full : uses
    Calc_ResidenceTime "1" --> "1" Sensor_bed_weight : reads
    Calc_CatInventory "1" --> "1" Sensor_F_cat_full : reads
    Calc_CatInventory "1" --> "1" Calc_FilterMassRate_full : uses
    Calc_CatInventory "1" --> "1" Sensor_bed_weight : reads
    Calc_AlTiRatio "1" --> "1" Sensor_F_cat_full : reads
    Calc_AlTiRatio "1" --> "1" Sensor_F_teal : reads
    Calc_PropertyEstimator "1" --> "1" Sensor_mass_rate_full : reads
    Calc_PropertyEstimator "1" --> "1" Sensor_lab_MI : reads
    Calc_PropertyEstimator "1" --> "1" Sensor_lab_density : reads
    Calc_PropertyEstimator "1" --> "1" Calc_ResidenceTime : uses
    Calc_PropertyEstimator "1" --> "1" Calc_MoleRatio_full : uses

    Controller_IMC_cat_inventory "1" --> "1" Calc_CatInventory : reads CV
    Controller_PID_massRate_full "1" --> "1" Calc_FilterMassRate_full : reads CV
    Controller_PID_pM1_full "1" --> "1" Calc_PartialPressure_full : reads CV
    Controller_PID_rM2_full "1" --> "1" Calc_MoleRatio_full : reads CV
    Controller_IMC_density "1" --> "1" Calc_PropertyEstimator : reads CV
    Controller_FO_density "1" --> "1" Calc_PropertyEstimator : reads CV
    Controller_FO_density "1" --> "1" Calc_ResidenceTime : tuning
    Controller_PID_rH2_full "1" --> "1" Calc_MoleRatio_full : reads CV
    Controller_IMC_MI "1" --> "1" Calc_PropertyEstimator : reads CV
    Controller_FO_MI "1" --> "1" Calc_PropertyEstimator : reads CV
    Controller_FO_MI "1" --> "1" Calc_ResidenceTime : tuning
    Controller_IMC_AlTi "1" --> "1" Calc_AlTiRatio : reads CV
    Controller_PID_F_n2 "1" --> "1" Calc_FilterPressure_full : reads CV
    Controller_PID_F_vent "1" --> "1" Calc_FilterPressure_full : reads CV
    Controller_BangBang_discharge "1" --> "1" Sensor_bed_level : reads CV

    Controller_IMC_cat_inventory "1" o-- "1" Controller_PID_massRate_full : cascade
    Controller_PID_rM2_full "1" o-- "1" Controller_IMC_density : cascade
    Controller_IMC_density "1" o-- "1" Controller_FO_density : cascade
    Controller_PID_rH2_full "1" o-- "1" Controller_IMC_MI : cascade
    Controller_IMC_MI "1" o-- "1" Controller_FO_MI : cascade
```

- Analyzer and mass-flow sensors feed multiple derived calculations, including mole ratios, partial pressure, residence time, catalyst inventory, and product property estimators.【F:new_examples/gas_phase_polymerization/component_definition.py†L32-L123】【F:new_examples/gas_phase_polymerization/calculations/misc_calculations.py†L12-L101】【F:new_examples/gas_phase_polymerization/calculations/property_estimator.py†L1-L118】
- Inventory, property, and ratio calculations supply CVs for cascaded IMC, PID, and trajectory controllers that supervise feed flows and quality targets, with inner loops dedicated to product properties and filtered throughput measurements.【F:new_examples/gas_phase_polymerization/component_definition.py†L125-L223】【F:new_examples/gas_phase_polymerization/calculations/misc_calculations.py†L59-L118】【F:new_examples/gas_phase_polymerization/calculations/property_estimator.py†L119-L268】
- Additional loops maintain pressure and solids handling by acting on filtered pressure and bed-level measurements via PID and bang-bang control, respectively.【F:new_examples/gas_phase_polymerization/component_definition.py†L200-L223】

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
