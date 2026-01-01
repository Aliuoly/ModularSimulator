from enum import IntEnum
from astropy.units import UnitBase, Unit


class TagType(IntEnum):
    """Enum for tag types"""

    INPUT = 1
    OUTPUT = 2
    CONSTANT = 3


class PointMetadata:
    """
    Represents information about a model state, including its type, unit, and description

    :var type: The type of the tag (e.g., input, output, constant)
    :vartype type: TagType
    :var unit: The unit associated with the tag's value.
    :vartype unit: UnitBase
    :var description: A brief description of the tag. Use this rather than inline comment where applicable.
    :vartype description: str = ""
    """

    type: TagType
    unit: UnitBase
    description: str = ""

    def __init__(self, type: TagType, unit: UnitBase | str, description: str = ""):
        if not isinstance(type, TagType):
            raise ValueError("type must be an instance of TagType")
        if not isinstance(unit, (UnitBase, str)):
            raise ValueError("unit must be an instance of UnitBase or str")
        self.type = type
        self.unit = Unit(unit) if isinstance(unit, str) else unit
        self.description = str(description)
