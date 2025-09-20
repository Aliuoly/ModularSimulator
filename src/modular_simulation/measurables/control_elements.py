from pydantic import BaseModel, ConfigDict



class ControlElements(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    """
    Representing control elements.
    Directly interacts with the States, but is externally controlled.
    Subclasses should define control variables as fields.
    """
    # e.g. user defined fields:
    # flow_rate: float = 0.0
    # inlet_temperature: float = 273.
    
