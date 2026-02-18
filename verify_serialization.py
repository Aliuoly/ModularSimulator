import numpy as np
from modular_simulation.framework.system import System
from modular_simulation.measurables.process_model import ProcessModel, StateMetadata, StateType
from modular_simulation.components import ControlElement, Trajectory
from modular_simulation.components.control_system.controller_mode import ControllerMode
from pydantic import Field
from typing import Annotated


class SimpleModel(ProcessModel):
    temp: Annotated[float, StateMetadata(StateType.DIFFERENTIAL, "K", "temperature")] = Field(300.0)
    heater_power: Annotated[float, StateMetadata(StateType.CONTROLLED, "W", "heater power")] = (
        Field(0.0)
    )

    @staticmethod
    def calculate_algebraic_values(*args, **kwargs):
        return np.array([])

    @staticmethod
    def differential_rhs(t, y, u, k, algebraic, y_map, u_map, k_map, algebraic_map):
        return -0.1 * (y - 290.0) + u[0]


def verify():
    model = SimpleModel()
    traj = Trajectory(y0=10.0)
    ce = ControlElement(
        mv_tag="heater_power", mv_trajectory=traj, mv_range=(0, 100), mode=ControllerMode.MANUAL
    )

    sys = System(dt=1.0, process_model=model, sensors=[], calculations=[], control_elements=[ce])

    sys.step(duration=5.0)
    print(f"Time before save: {sys.time}")
    print(f"Temp before save: {sys.process_model.temp}")

    payload = sys.save()

    restored = System.load(payload)
    print("System restored successfully")

    assert restored.time == sys.time
    assert restored.process_model.temp == sys.process_model.temp
    assert restored.control_elements[0].mode == sys.control_elements[0].mode
    assert isinstance(restored.control_elements[0].mv_trajectory, Trajectory)

    # Test step after restore
    restored.step(duration=1.0)
    print(f"Time after restore step: {restored.time}")
    print(f"Temp after restore step: {restored.process_model.temp}")

    print("Serialization Verify OK")


if __name__ == "__main__":
    verify()
