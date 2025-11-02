from astropy.units import Unit, Quantity
from modular_simulation.utils.typing import TimeQuantity, PerTimeQuantity, Seconds, PerSeconds

def day(value: float) -> Quantity[Unit("day")]:
    return Quantity(value, unit="day")
def per_day(value: float) -> Quantity[Unit("1/day")]:
    return Quantity(value, unit="1/day")
def hour(value: float) -> Quantity[Unit("hour")]:
    return Quantity(value, unit="hour")
def per_hour(value: float) -> Quantity[Unit("1/hour")]:
    return Quantity(value, unit="1/hour")
def minute(value: float) -> Quantity[Unit("minute")]:
    return Quantity(value, unit = "minute")
def per_minute(value: float) -> Quantity[Unit("1/minute")]:
    return Quantity(value, unit = "1/minute")
def second(value: Seconds|TimeQuantity) -> Quantity[Unit("second")]:
    """
    Convert a float or Seconds to a Quantity in units of seconds

    :param value: Value to convert to seconds (float or Seconds)
    :type value: float | Seconds
    :return: Quantity representing the value in seconds
    :rtype: Quantity
    """
    if isinstance(value, Quantity):
        return value.to("second")
    return Quantity(value, unit="second")

def second_value(value: Seconds|TimeQuantity) -> float:
    """
    Convert a float or Seconds to its value in seconds as a float
    If is float, just returns as is assuming it is already in seconds.
    :param value: Value to convert to seconds (float or Seconds)
    :type value: float | Seconds
    :return: The value in seconds as a float
    :rtype: float
    """
    if isinstance(value, Quantity):
        return value.to_value("second")
    return value

def per_second(value: PerSeconds|PerTimeQuantity) -> Quantity[Unit("1/second")]:
    """
    Convert a float or Seconds to a Quantity in units of seconds

    :param value: Value to convert to seconds (float or Seconds)
    :type value: float | Seconds
    :return: Quantity representing the value in seconds
    :rtype: Quantity
    """
    if isinstance(value, Quantity):
        return value.to("1/second")
    return Quantity(value, unit="1/second")

def per_second_value(value: PerSeconds|PerTimeQuantity) -> float:
    """
    Convert a float or Seconds to its value in seconds as a float
    If is float, just returns as is assuming it is already in seconds.
    :param value: Value to convert to seconds (float or Seconds)
    :type value: float | Seconds
    :return: The value in seconds as a float
    :rtype: float
    """
    if isinstance(value, Quantity):
        return value.to_value("1/second")
    return value