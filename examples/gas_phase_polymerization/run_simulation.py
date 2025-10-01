from modular_simulation.framework import create_system
from modular_simulation.plotting import plot_triplet_series
from system_definitions import (
    GasPhaseReactorSystem,
    GasPhaseReactorStates,
    GasPhaseReactorAlgebraicStates,
    GasPhaseReactorConstants,
    GasPhaseReactorControlElements
)
from controller_definitions import controllers
from usable_definitions import sensors, calculations
import matplotlib.pyplot as plt



# Assemble the systems
dt = 30.0 # 30 seconds
normal_system = create_system(
    dt=dt,
    system_class=GasPhaseReactorSystem,
    initial_states=GasPhaseReactorStates(),
    initial_controls=GasPhaseReactorControlElements(),
    initial_algebraic=GasPhaseReactorAlgebraicStates(),
    system_constants=GasPhaseReactorConstants(),
    sensors=sensors,
    calculations=calculations,
    controllers=controllers,
)

normal_system.step(10000)

sensor_hist = normal_system.measured_history["sensors"]

fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111)
plot_triplet_series(
    ax,
    sensor_hist["mass_prod_rate"],
)
plt.show()
