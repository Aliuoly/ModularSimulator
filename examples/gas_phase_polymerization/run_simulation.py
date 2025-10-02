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
import logging
#logging.basicConfig(level=logging.DEBUG, format="%(message)s")



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

normal_system.step(2000)
history = normal_system.measured_history
sensor_hist = history["sensors"]
calculations_hist = history["calculations"]
sp_hist = normal_system.setpoint_history

fig = plt.figure(figsize=(15,15))
ax = plt.subplot(331)
plot_triplet_series(
    ax,
    sensor_hist["mass_prod_rate"],
)
plot_triplet_series(
    ax,
    sp_hist["mass_prod_rate"],
    label = 'SP'
)
plt.legend()
ax = plt.subplot(332)
plot_triplet_series(
    ax,
    calculations_hist["cumm_MI"],
    label = 'cumm MI'
)
plot_triplet_series(
    ax,
    sensor_hist["lab_MI"],
    label = 'lab MI'
)
plot_triplet_series(
    ax,
    sp_hist["cumm_MI"],
    label = 'SP'
)
plt.legend()
ax = plt.subplot(333)
plot_triplet_series(
    ax,
    calculations_hist["cumm_density"],
    label = 'cumm density'
)
plot_triplet_series(
    ax,
    sensor_hist["lab_density"],
    label = 'lab density'
)
plot_triplet_series(
    ax,
    sp_hist["cumm_density"],
    label = 'SP'
)
plt.legend()
ax = plt.subplot(6,3,7)
plot_triplet_series(
    ax,
    calculations_hist["cat_inventory"],
    label = 'cat inventory'
)
plot_triplet_series(
    ax,
    sp_hist["cat_inventory"],
    label = 'SP'
)
plt.legend()
ax = plt.subplot(6,3,10)
plot_triplet_series(
    ax,
    calculations_hist["pM1"],
    label = 'pM1'
)
plot_triplet_series(
    ax,
    sp_hist["pM1"],
    label = 'SP'
)
plt.legend()
ax = plt.subplot(335)
plot_triplet_series(
    ax,
    calculations_hist["rH2"],
    label = 'rH2'
)
plot_triplet_series(
    ax,
    sp_hist["rH2"],
    label = 'SP'
)
plt.legend()
ax = plt.subplot(336)
plot_triplet_series(
    ax,
    calculations_hist["rM2"],
    label = 'rM2'
)
plot_triplet_series(
    ax,
    sp_hist["rM2"],
    label = 'SP'
)
plt.legend()
ax = plt.subplot(6,3,13)
plot_triplet_series(
    ax,
    sensor_hist["F_cat"],
    label = 'F_cat'
)
ax = plt.subplot(6,3,16)
plot_triplet_series(
    ax,
    sensor_hist["F_m1"],
    label = 'F_m1'
)
ax = plt.subplot(3,3,8)
plot_triplet_series(
    ax,
    sensor_hist["F_h2"],
    label = 'F_h2'
)
ax = plt.subplot(3,3,9)
plot_triplet_series(
    ax,
    sensor_hist["F_m2"],
    label = 'F_m2'
)
plt.legend()
plt.show()