from modular_simulation.usables import SampledDelayedSensor
from calculations.property_estimator import PropertyEstimator
from calculations.misc_calculations import (
    MoleRatioCalculation, 
    ResidenceTimeCalculation, 
    CatInventoryEstimator, 
    AlTiRatioEstimator,
    Monomer1PartialPressure
)
from astropy.units import Unit
from modular_simulation.usables.calculations.first_order_filter import FirstOrderFilter
from functools import partial
analyzer_partial = partial(
    SampledDelayedSensor, 
    deadtime = 120., 
    sampling_period = 120.,
    coefficient_of_variance = 0.01, # 1% rel std
    ) # 2 min delay and sampling period
sensors = [
    analyzer_partial(measurement_tag="yM1", unit=''),
    analyzer_partial(measurement_tag="yM2", unit=''),
    analyzer_partial(measurement_tag="yH2", unit=''),
    SampledDelayedSensor(
        measurement_tag="pressure",
        unit=Unit("kPa"),
        coefficient_of_variance = 0.02,
        ),
    SampledDelayedSensor(
        measurement_tag="bed_weight",
        unit=Unit("tonne"),
        coefficient_of_variance = 0.03,
        ),
    SampledDelayedSensor(
        measurement_tag="mass_prod_rate",
        unit=Unit("tonne/hr"),
        deadtime = 120., 
        sampling_period = 120.,
        coefficient_of_variance = 0.02,
        time_constant = 600.0, # system time units -> seconds
    ),
    SampledDelayedSensor(
        measurement_tag="cumm_MI", alias_tag="lab_MI",
        deadtime = 2 * 3600, unit='',
        sampling_period = 2 * 3600,# 2 hours,
        instrument_range = (0.05, 200.)
    ),
    SampledDelayedSensor(
        measurement_tag="cumm_density", alias_tag = "lab_density",
        deadtime = 2 * 3600,
        unit=Unit("g/L"),
        sampling_period = 2 * 3600,# 2 hours
        instrument_range = (900, 970)
    ),
    SampledDelayedSensor(measurement_tag="bed_level",
        coefficient_of_variance = 0.03,
        unit=Unit("m")
        ),
    SampledDelayedSensor(measurement_tag="F_cat", unit=Unit("kg/hr")),
    SampledDelayedSensor(measurement_tag="F_teal", unit=Unit("mol/hr")),
    SampledDelayedSensor(measurement_tag="F_m1", unit=Unit("kg/hr")),
    SampledDelayedSensor(measurement_tag="F_m2", unit=Unit("kg/hr")),
    SampledDelayedSensor(measurement_tag="F_h2", unit=Unit("kg/hr")),
    SampledDelayedSensor(measurement_tag="F_n2", unit=Unit("kg/hr")),
    SampledDelayedSensor(measurement_tag="F_vent", unit=Unit("L/hr")),
    SampledDelayedSensor(measurement_tag="discharge_valve_position", unit=""),
]
calculations = [
    FirstOrderFilter(
        name = 'mass_prod_rate_filter',
        raw_signal_tag = 'mass_prod_rate',
        filtered_signal_tag = 'filtered_mass_prod_rate',
        time_constant = 600., # system time units again. 
    ),
    FirstOrderFilter(
        name = 'pressure_filter',
        raw_signal_tag = 'pressure',
        filtered_signal_tag = 'filtered_pressure',
        signal_unit = Unit("kPa"),
        time_constant = 300., # system time units again. 
    ),
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
        pressure_tag = "filtered_pressure"
    ),
    ResidenceTimeCalculation(
        residence_time_tag = "residence_time", 
        mass_prod_rate_tag = "filtered_mass_prod_rate", 
        bed_weight_tag = "bed_weight",  
    ),
    CatInventoryEstimator(
        cat_inventory_tag = "cat_inventory",
        F_cat_tag = "F_cat",
        mass_prod_rate_tag = "filtered_mass_prod_rate",
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
