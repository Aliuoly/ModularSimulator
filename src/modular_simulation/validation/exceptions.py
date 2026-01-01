class SensorConfigurationError(Exception):
    """
    Raised when a system defines sensors that are not
      resolvable around the system's measurables.
    """


class CalculationConfigurationError(Exception):
    """
    Raised when a system defines calculations that are not resolvable
      around the system's measurements or other calculations.
    """


class ControllerConfigurationError(Exception):
    """
    Raised when a system defines controllers that are dependent on measurements/calculations
      not resolvable around the system's usable definitions, or when the defined manipulated variable,
      which is determine to be a final control element, is not defined in the system's control elements.
    """


class ControlElementConfigurationError(Exception):
    """
    Raised when a system defines control elements that are not resolvable
      around the system's measurables.
    """


class MeasurableConfigurationError(Exception):
    """
    Raise when a system has duplicate measurables defined across its
    States, Constants, AlgebraicStates, and/or ControlElements,
    or when the system has no measurables defined (in which case, there is no system)
    """


class CalculationDefinitionError(Exception):
    """
    Raised when a subclass definition does not satisfy requirements.
    """
