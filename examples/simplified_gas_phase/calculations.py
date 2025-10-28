from calculation_definition import (
    MoleRatioCalculation, 
    Monomer1PartialPressure
)
from astropy.units import Unit
from modular_simulation.usables.calculations.first_order_filter import FirstOrderFilter
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
        yM1_tag = "yM1",
        yM2_tag = "yM2",
    ),
    Monomer1PartialPressure(
        pM1_tag = "pM1", 
        yM1_tag = "yM1",
        pressure_tag = "filtered_pressure"
    ),
]