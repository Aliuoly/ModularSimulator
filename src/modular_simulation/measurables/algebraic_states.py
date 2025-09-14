# In modular_simulation/measurables/AlgebraicStates.py
from pydantic import BaseModel, ConfigDict

class AlgebraicStates(BaseModel):
    """
    Base class for quantities that are algebraic functions of states.
    """
    # By adding extra='forbid', Pydantic will error on typos.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')