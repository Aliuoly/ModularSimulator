from __future__ import annotations

from functools import partial

from modular_simulation.usables import (
    BangBangController,
    CalculationBase,
    CalculationModelPath,
    InternalModelController,
    PIDController,
    SampledDelayedSensor,
    Trajectory,
    ControlElement,
    FirstOrderTrajectoryController,
)
from modular_simulation.usables.calculations.first_order_filter import FirstOrderFilter
from modular_simulation.utils.wrappers import minute, hour, per_hour
from calculations.misc_calculations import (
    AlTiRatioEstimator,
    CatInventoryEstimator,
    MoleRatioCalculation,
    Monomer1PartialPressure,
    ResidenceTimeCalculation,
)
from calculations.property_estimator import PropertyEstimator

analyzer_partial = partial(
    SampledDelayedSensor,
    deadtime=minute(2),
    sampling_period=minute(2),
    coefficient_of_variance=0.01,
)

sensors = [
    analyzer_partial(measurement_tag="yM1", unit=""),
    analyzer_partial(measurement_tag="yM2", unit=""),
    analyzer_partial(measurement_tag="yH2", unit=""),
    SampledDelayedSensor(
        measurement_tag="pressure",
        unit="kPa",
        coefficient_of_variance=0.00,
    ),
    SampledDelayedSensor(
        measurement_tag="bed_weight",
        unit="tonne",
        coefficient_of_variance=0.00,
    ),
    SampledDelayedSensor(
        measurement_tag="mass_prod_rate",
        unit="tonne/hr",
        deadtime=minute(2),
        sampling_period=minute(2),
        coefficient_of_variance=0.00,
    ),
    SampledDelayedSensor(
        measurement_tag="cumm_MI",
        alias_tag="lab_MI",
        deadtime=hour(2),
        unit="",
        sampling_period=hour(2),
        instrument_range=(0.05, 200.0),
    ),
    SampledDelayedSensor(
        measurement_tag="cumm_density",
        alias_tag="lab_density",
        deadtime=hour(2),
        unit="g/L",
        sampling_period=hour(2),
        instrument_range=(900.0, 970.0),
    ),
    SampledDelayedSensor(
        measurement_tag="bed_level",
        coefficient_of_variance=0.00,
        unit="m",
    ),
    SampledDelayedSensor(measurement_tag="F_cat", unit="kg/hr"),
    SampledDelayedSensor(measurement_tag="F_teal", unit="mol/hr"),
    SampledDelayedSensor(measurement_tag="F_m1", unit="kg/hr"),
    SampledDelayedSensor(measurement_tag="F_m2", unit="kg/hr"),
    SampledDelayedSensor(measurement_tag="F_h2", unit="kg/hr"),
    SampledDelayedSensor(measurement_tag="F_n2", unit="kg/hr"),
    SampledDelayedSensor(measurement_tag="F_vent", unit="L/hr"),
    SampledDelayedSensor(measurement_tag="discharge_valve_position", unit=""),
]

calculations: list[CalculationBase] = [
    FirstOrderFilter(
        name="mass_prod_rate_filter",
        raw_signal_tag="mass_prod_rate",
        filtered_signal_tag="filtered_mass_prod_rate",
        time_constant=minute(10),
    ),
    FirstOrderFilter(
        name="pressure_filter",
        raw_signal_tag="pressure",
        filtered_signal_tag="filtered_pressure",
        time_constant=minute(5),
    ),
    FirstOrderFilter(
        name="pressure_filter_vent",
        raw_signal_tag="pressure",
        filtered_signal_tag="filtered_pressure_vent",
        time_constant=minute(5),
    ),
    MoleRatioCalculation(
        rM2_tag="rM2",
        rH2_tag="rH2",
        yM1_tag="yM1",
        yM2_tag="yM2",
        yH2_tag="yH2",
    ),
    Monomer1PartialPressure(
        pM1_tag="pM1",
        yM1_tag="yM1",
        pressure_tag="filtered_pressure",
    ),
    ResidenceTimeCalculation(
        residence_time_tag="residence_time",
        mass_prod_rate_tag="filtered_mass_prod_rate",
        bed_weight_tag="bed_weight",
    ),
    CatInventoryEstimator(
        cat_inventory_tag="cat_inventory",
        F_cat_tag="F_cat",
        mass_prod_rate_tag="filtered_mass_prod_rate",
        bed_weight_tag="bed_weight",
    ),
    AlTiRatioEstimator(
        AlTi_ratio_tag="Al_Ti_ratio",
        F_teal_tag="F_teal",
        F_cat_tag="F_cat",
    ),
    PropertyEstimator(
        inst_MI_tag="inst_MI",
        inst_density_tag="inst_density",
        cumm_MI_tag="cumm_MI",
        cumm_density_tag="cumm_density",
        mass_prod_rate_tag="mass_prod_rate",
        lab_MI_tag="lab_MI",
        lab_density_tag="lab_density",
        residence_time_tag="residence_time",
        rM2_tag="rM2",
        rH2_tag="rH2",
    ),
]

control_elements: list[ControlElement] = [
    ControlElement(
        mv_tag="F_cat",
        mv_range=(0.0, 20.0),
        controller=InternalModelController(
            cv_tag="cat_inventory",
            cv_range=(0.0, 200.0),
            sp_filter_tc=minute(10),
            model=CalculationModelPath(
                calculation_name=CatInventoryEstimator,
                method_name="model",
            ),
            cascade_controller=PIDController(
                cv_tag="filtered_mass_prod_rate",
                cv_range=(0.0, 100.0),
                sp_trajectory=Trajectory(y0=0.0).ramp(50.0, rate=per_hour(10.0)),
                Kp=0.5,
                Ti=hour(1),
                setpoint_weight=0.0,
            ),
        ),
    ),
    ControlElement(
        mv_tag="F_m1",
        mv_range=(0.0, 70_000.0),
        controller=PIDController(
            cv_tag="pM1",
            cv_range=(0.0, 2000.0),
            sp_trajectory=Trajectory(y0=700.0),
            Kp=50.0,
            Ti=hour(1),
            period=minute(2),
        ),
    ),
    ControlElement(
        mv_tag="F_m2",
        mv_range=(0.0, 7_000.0),
        controller=PIDController(
            cv_tag="rM2",
            cv_range=(0.0, 1.0),
            sp_trajectory=Trajectory(y0=0.3),
            Kp=3500.0,
            Ti=hour(0.5),
            setpoint_weight=0.0,
            period=minute(2),
            cascade_controller=InternalModelController(
                cv_tag="inst_density",
                cv_range=(900.0, 970.0),
                model=CalculationModelPath(
                    calculation_name=PropertyEstimator,
                    method_name="inst_density_model",
                ),
                cascade_controller=FirstOrderTrajectoryController(
                    cv_tag="cumm_density",
                    cv_range=(900.0, 970.0),
                    sp_trajectory=Trajectory(y0=918.0),
                    closed_loop_time_constant_fraction=0.8,
                    open_loop_time_constant="residence_time",
                ),
            ),
        ),
    ),
    ControlElement(
        mv_tag="F_h2",
        mv_range=(0.0, 15.0),
        controller=PIDController(
            cv_tag="rH2",
            cv_range=(0.0, 1.0),
            sp_trajectory=Trajectory(y0=0.0),
            Kp=200.0,
            Ti=hour(1.5),
            setpoint_weight=0.0,
            period=minute(2),
            cascade_controller=InternalModelController(
                cv_tag="inst_MI",
                cv_range=(0.0, 200.0),
                model=CalculationModelPath(
                    calculation_name=PropertyEstimator,
                    method_name="inst_MI_model",
                ),
                cascade_controller=FirstOrderTrajectoryController(
                    cv_tag="cumm_MI",
                    cv_range=(0.0, 200.0),
                    sp_trajectory=Trajectory(y0=2.0),
                    closed_loop_time_constant_fraction=0.8,
                    open_loop_time_constant="residence_time",
                ),
            ),
        ),
    ),
    ControlElement(
        mv_tag="F_teal",
        mv_range=(0.0, 1e5),
        controller=InternalModelController(
            cv_tag="Al_Ti_ratio",
            cv_range=(0.0, 10.0),
            sp_trajectory=Trajectory(y0=0.4),
            model=CalculationModelPath(
                calculation_name=AlTiRatioEstimator,
                method_name="AlTi_model",
            ),
        ),
    ),
    ControlElement(
        mv_tag="F_n2",
        mv_range=(0.0, 500.0),
        controller=PIDController(
            cv_tag="filtered_pressure",
            cv_range=(0.0, 3000.0),
            sp_trajectory=Trajectory(y0=1900.0),
            Kp=500.0 / 200.0,
        ),
    ),
    ControlElement(
        mv_tag="F_vent",
        mv_range=(0.0, 25_000.0),
        controller=PIDController(
            cv_tag="filtered_pressure_vent",
            cv_range=(0.0, 3000.0),
            sp_trajectory=Trajectory(y0=1900.0),
            Kp=25_000.0 / 200.0,
            inverted=True,
        ),
    ),
    ControlElement(
        mv_tag="discharge_valve_position",
        mv_range=(0.0, 1.0),
        controller=BangBangController(
            cv_tag="bed_level",
            cv_range=(0.0, 20.0),
            sp_trajectory=Trajectory(y0=15.0),
            deadband=0.1,
            alpha=0.4,
            inverted=True,
        ),
    ),
]
