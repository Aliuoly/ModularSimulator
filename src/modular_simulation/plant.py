from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from modular_simulation.framework.system import System


class Plant:
    systems: List["System"]

