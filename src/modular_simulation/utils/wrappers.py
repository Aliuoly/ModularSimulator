from astropy.units import Quantity
from modular_simulation.utils.typing import TimeQuantity, PerTimeQuantity, Seconds, PerSeconds


def day(value: float) -> Seconds:
    return value * 86400.0


def per_day(value: float) -> PerSeconds:
    return value / 86400.0


def hour(value: float) -> Seconds:
    return value * 3600.0


def per_hour(value: float) -> PerSeconds:
    return value / 3600.0


def minute(value: float) -> Seconds:
    return value * 60.0


def per_minute(value: float) -> PerSeconds:
    return value / 60.0


def second(value: Seconds | TimeQuantity) -> Seconds:
    """
    Convert a float or Seconds to a Quantity in units of seconds

    :param value: Value to convert to seconds (float or Seconds)
    :type value: float | Seconds
    :return: Quantity representing the value in seconds
    :rtype: Quantity
    """
    if isinstance(value, Quantity):
        if isinstance(value.value, float):
            return float(value.to_value("second"))
        else:
            raise ValueError("Value or value.value is not a float")
    return value


def per_second(value: PerSeconds | PerTimeQuantity) -> PerSeconds:
    """
    Convert a float or Seconds to a Quantity in units of seconds

    :param value: Value to convert to seconds (float or Seconds)
    :type value: float | Seconds
    :return: Quantity representing the value in seconds
    :rtype: Quantity
    """
    if isinstance(value, Quantity):
        if isinstance(value.value, float):
            return float(value.to_value("1/second"))
        else:
            raise ValueError("Value or value.value is not a float")
    return value
