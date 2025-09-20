from pydantic import BaseModel, ConfigDict

class AlgebraicStates(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    """
    Base class for quantities that are algebraic functions of states. 
    Calculation is defined in the system. It may be calculated from any method
    that does not involve integration.
    """
    # e.g. user defined fields:
    # outlet_flow: float = 0.0
        # e.g. calculated as a pressure driven flow = Cv * f(dP)