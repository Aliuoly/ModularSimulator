class ConfigurationError(Exception):
    """
    A custom error that arises from mistakes in measurable, usable, or controllable quantity definition. 
    """
    def __init__(self, message = "An validation error occured"):
        self.message = message
        super().__init__(self.message)
