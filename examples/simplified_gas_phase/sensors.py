from modular_simulation.usables import SampledDelayedSensor
from functools import partial

analyzer_partial = partial(
    SampledDelayedSensor, 
    deadtime = 120., 
    sampling_period = 120.,
    coefficient_of_variance = 0.005, # 1% rel std
    ) # 2 min delay and sampling period
sensors = [
    analyzer_partial(measurement_tag="yM1", unit='', random_seed=0),
    analyzer_partial(measurement_tag="yM2", unit='', random_seed=1),
    SampledDelayedSensor(measurement_tag = "pressure", unit="kPa", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "mass_prod_rate", unit="kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "F_m1", unit = "kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "F_m2", unit = "kg/hour", coefficient_of_variance=0.002),
    SampledDelayedSensor(measurement_tag = "F_cat", unit = "kg/hour", coefficient_of_variance=0.002),
]
