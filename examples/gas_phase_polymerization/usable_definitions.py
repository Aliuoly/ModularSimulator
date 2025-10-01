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
    SampledDelayedSensor(measurement_tag="cumm_MI", alias_tag="lab_MI"),
    SampledDelayedSensor(measurement_tag="cumm_density", alias_tag = "lab_density"),
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
        output_tags = ["rM2","rH2"], #!!! order DOES matter, check implementation
        measured_input_tags = ["yM1","yM2","yH2"]  # order DOES NOT matter here lol
    ),
    Monomer1PartialPressure(
        output_tags = ["pM1"], 
        measured_input_tags = ["yM1","pressure"]  # order DOES NOT matter here lol
    ),
    ResidenceTimeCalculation(
        output_tags = "residence_time", # auto converts to list if you provide a single string. 
        measured_input_tags = ["mass_prod_rate", "bed_weight"],  
    ),
    CatInventoryEstimator(
        output_tags = "cat_inventory",
        measured_input_tags = ["F_cat","mass_prod_rate","bed_weight"],  
    ),
    AlTiRatioEstimator(
        output_tags = "Al_Ti_ratio",
        measured_input_tags = ["F_teal", "F_cat"],
    ),
    PropertyEstimator(
        output_tags = ["inst_MI","inst_density","cumm_MI","cumm_density"], #!!! order DOES matter, check implementation
        measured_input_tags = ["mass_prod_rate","lab_MI","lab_density"],
        calculated_input_tags = ["residence_time","rM2","rH2"], 
        # rest keep default for now
    )
]

controllers = []