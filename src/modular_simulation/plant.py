from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from modular_simulation.system import System


class Plant:
    systems: List["System"]

