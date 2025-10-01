from modular_simulation.usables import SampledDelayedSensor
from calculations.property_estimator import PropertyEstimator
from calculations.misc_calculations import (
    MoleRatioCalculation, 
    ResidenceTimeCalculation, 
    CatInventoryEstimator, 
    AlTiRatioEstimator,
    Monomer1PartialPressure
)

sensors = [
    SampledDelayedSensor(measurement_tag="yM1"),
    SampledDelayedSensor(measurement_tag="yM2"),
    SampledDelayedSensor(measurement_tag="yH2"),
    SampledDelayedSensor(measurement_tag="pressure"),
    SampledDelayedSensor(measurement_tag="bed_weight"),
    SampledDelayedSensor(measurement_tag="mass_prod_rate"),
    SampledDelayedSensor(
        measurement_tag="cumm_MI", alias_tag="lab_MI",
        deadtime = 2 * 3600,
        sampling_period = 2 * 3600,# 2 hours
    ),
    SampledDelayedSensor(
        measurement_tag="cumm_density", alias_tag = "lab_density",
        deadtime = 2 * 3600,
        sampling_period = 2 * 3600,# 2 hours
    ),
    SampledDelayedSensor(measurement_tag="bed_level"),
    SampledDelayedSensor(measurement_tag="F_cat"),
    SampledDelayedSensor(measurement_tag="F_teal"),
    SampledDelayedSensor(measurement_tag="F_m1"),
    SampledDelayedSensor(measurement_tag="F_m2"),
    SampledDelayedSensor(measurement_tag="F_h2"),
    SampledDelayedSensor(measurement_tag="F_n2"),
    SampledDelayedSensor(measurement_tag="F_vent"),
    SampledDelayedSensor(measurement_tag="discharge_valve_position"),
]
calculations = [
    MoleRatioCalculation(
        rM2_tag = "rM2",
        rH2_tag = "rH2",
        yM1_tag = "yM1",
        yM2_tag = "yM2",
        yH2_tag = "yH2",
    ),
    Monomer1PartialPressure(
        pM1_tag = "pM1", 
        yM1_tag = "yM1",
        pressure_tag = "pressure"
    ),
    ResidenceTimeCalculation(
        residence_time_tag = "residence_time", 
        mass_prod_rate_tag = "mass_prod_rate", 
        bed_weight_tag = "bed_weight",  
    ),
    CatInventoryEstimator(
        cat_inventory_tag = "cat_inventory",
        F_cat_tag = "F_cat",
        mass_prod_rate_tag = "mass_prod_rate",
        bed_weight_tag = "bed_weight",  
    ),
    AlTiRatioEstimator(
        AlTi_ratio_tag = "Al_Ti_ratio",
        F_teal_tag = "F_teal",
        F_cat_tag = "F_cat",
    ),
    PropertyEstimator(
        inst_MI_tag = "inst_MI",
        inst_density_tag = "inst_density",
        cumm_MI_tag = "cumm_MI",
        cumm_density_tag = "cumm_density",
        mass_prod_rate_tag = "mass_prod_rate",
        lab_MI_tag = "lab_MI",
        lab_density_tag = "lab_density",
        residence_time_tag = "residence_time",
        rM2_tag = "rM2",
        rH2_tag = "rH2", 
        # rest keep default for now
    )
]
