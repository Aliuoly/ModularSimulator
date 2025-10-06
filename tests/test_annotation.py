

from pydantic import BaseModel
from typing import Annotated

class Dummy(BaseModel):
    a: Annotated[float, 'abc']

d = Dummy(a = 100)
for fieldname, fieldinfo in d.__class__.model_fields.items():
    print(fieldinfo.metadata[0])
from astropy.units import Unit, CompositeUnit
molarity = Unit("mol/L")

print(isinstance(molarity, CompositeUnit))