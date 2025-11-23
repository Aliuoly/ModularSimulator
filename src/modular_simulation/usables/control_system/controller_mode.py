from enum import IntEnum

class ControllerMode(IntEnum):
    """
    TRACKING: CV's SP = CV's PV Always. SP cannot be changed - always follows PV.
    AUTO    : CV's SP is provided by sp_trajectory. 
    CASCADE : CV's SP is provided by a cascade controller.
    """
    TRACKING = -1 
    tracking = -1
    track = -1

    MANUAL = 0
    manual = 0
    man = 0

    AUTO = 1
    auto = 1
    automatic = 1
    a = 1

    CASCADE = 2
    cascade = 2
    cas = 2

    @classmethod
    def from_value(cls, value: "ControllerMode | str | int") -> "ControllerMode":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"Invalid controller mode int '{value}'. "
                    f"Valid numeric values: {[m.value for m in cls]}"
                )
        if isinstance(value, str):
            try:
                return cls[value.lower()]
            except Exception:
                raise ValueError(
                    f"Unrecognized controller mode string '{value}'. "
                    f"Accepted strings: {cls.__members__.keys()}"
                )
        raise TypeError(
            f"Expected ControllerMode, int, or str, got {type(value).__name__}"
        )
