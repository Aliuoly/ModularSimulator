from numpy.typing import NDArray
import numpy as np
from typing import TypeAlias, Annotated
from astropy.units import UnitBase, Unit
from pydantic import BeforeValidator, PlainSerializer

StateValue: TypeAlias = float | NDArray[np.float64]

SerializableUnit = Annotated[
    str | UnitBase,
    BeforeValidator(lambda u: u if isinstance(u, UnitBase) else Unit(u)),
    PlainSerializer(lambda u: str(u)),
]