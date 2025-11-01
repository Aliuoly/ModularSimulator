from astropy.units import Unit, Quantity
from modular_simulation.utils.typing import TimeValue

def day(value: float) -> Quantity[Unit("day")]:
    return Quantity(value, unit="day")
def hour(value: float) -> Quantity[Unit("hour")]:
    return Quantity(value, unit="hour")
def minute(value: float) -> Quantity[Unit("minute")]:
    return Quantity(value, unit = "minute")
def second(value: float | TimeValue) -> Quantity[Unit("second")]:
    """
    Convert a float or TimeValue to a Quantity in units of seconds

    :param value: Value to convert to seconds (float or TimeValue)
    :type value: float | TimeValue
    :return: Quantity representing the value in seconds
    :rtype: Quantity
    """
    if isinstance(value, Quantity):
        return value.to("second")
    return Quantity(value, unit="second")

def second_value(value_quantity:Quantity) -> float:
    return value_quantity.to_value("second")