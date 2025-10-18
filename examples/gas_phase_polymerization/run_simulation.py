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
import matplotlib as mpl
from astropy.units import Unit
import logging
#logging.basicConfig(level=logging.DEBUG, format="%(message)s")
#mpl.set_loglevel("warning") # silence matplotlib debug messages


# Assemble the systems
dt = 30.0 * Unit("second") # 30 seconds
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

normal_system.step(24 * Unit("hour"))
history = normal_system.measured_history
sensor_hist = history["sensors"]
calculations_hist = history["calculations"]
sp_hist = normal_system.setpoint_history

fig, axes = plt.subplots(4, 3, figsize=(20, 18), sharex=True)
axes = axes.flatten()

# -------- Row 1: Flow–Property Pairs --------
# (F_m1, pM1)
ax = axes[0]
plot_triplet_series(ax, sensor_hist["F_m1"], label="F_m1",
                    line_kwargs={"color": "tab:blue"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax2 = ax.twinx()
plot_triplet_series(ax2, calculations_hist["pM1"], label="pM1",
                    line_kwargs={"color": "tab:orange"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax2, sp_hist["pM1"], label="SP pM1",
                    line_kwargs={"color": "tab:red", "linestyle": "--"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax.set_title("F_m1 vs pM1")
ax.set_ylabel("F_m1")
ax2.set_ylabel("pM1")
ax.grid(True, alpha=0.3)
lns1, labs1 = ax.get_legend_handles_labels()
lns2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lns1 + lns2, labs1 + labs2, loc="best")

# (F_m2, rM2)
ax = axes[1]
plot_triplet_series(ax, sensor_hist["F_m2"], label="F_m2",
                    line_kwargs={"color": "tab:blue"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax2 = ax.twinx()
plot_triplet_series(ax2, calculations_hist["rM2"], label="rM2",
                    line_kwargs={"color": "tab:orange"}, style='step', 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax2, sp_hist["rM2"], label="SP rM2",
                    line_kwargs={"color": "tab:red", "linestyle": "--"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax.set_title("F_m2 vs rM2")
ax.set_ylabel("F_m2")
ax2.set_ylabel("rM2")
ax.grid(True, alpha=0.3)
lns1, labs1 = ax.get_legend_handles_labels()
lns2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lns1 + lns2, labs1 + labs2, loc="best")

# (F_h2, rH2)
ax = axes[2]
plot_triplet_series(ax, sensor_hist["F_h2"], label="F_h2",
                    line_kwargs={"color": "tab:blue"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax2 = ax.twinx()
plot_triplet_series(ax2, calculations_hist["rH2"], label="rH2",
                    line_kwargs={"color": "tab:orange"}, style='step', 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax2, sp_hist["rH2"], label="SP rH2",
                    line_kwargs={"color": "tab:red", "linestyle": "--"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax.set_title("F_h2 vs rH2")
ax.set_ylabel("F_h2")
ax2.set_ylabel("rH2")
ax.grid(True, alpha=0.3)
lns1, labs1 = ax.get_legend_handles_labels()
lns2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lns1 + lns2, labs1 + labs2, loc="best")

# -------- Row 2 --------
# (Prod Rate, F_cat)
ax = axes[3]
plot_triplet_series(ax, sensor_hist["mass_prod_rate"], label="Prod Rate",
                    line_kwargs={"color": "royalblue",'alpha':0.5}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax, calculations_hist["filtered_mass_prod_rate"], label="filtered Prod Rate",
                    line_kwargs={"color": "green"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax, sp_hist["filtered_mass_prod_rate"], label="SP Prod Rate",
                    line_kwargs={"color": "tab:green", "linestyle": "--"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours

ax.set_title("Prod Rate vs F_cat")
ax.set_ylabel("Prod Rate")
ax.grid(True, alpha=0.3)
lns1, labs1 = ax.get_legend_handles_labels()
ax.legend(lns1, labs1, loc="best")

# Cumulative MI
ax = axes[4]
plot_triplet_series(ax, calculations_hist["cumm_MI"], label="cumm MI",
                    line_kwargs={"color": "tab:orange"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax, calculations_hist["inst_MI"], label="inst MI",
                    line_kwargs={"color": "tab:red", "linestyle": ":"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax, sensor_hist["lab_MI"], label="lab MI",
                    line_kwargs={"color": "tab:blue"}, style = 'step', 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax, sp_hist["cumm_MI"], label="SP MI",
                    line_kwargs={"color": "tab:red", "linestyle": "--"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax, sp_hist["inst_MI"], label="SP inst",
                    line_kwargs={"color": "tab:green", "linestyle": "--"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax.set_title("Melt Index")
ax.set_ylabel("MI")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

# Cumulative Density
ax = axes[5]
plot_triplet_series(ax, calculations_hist["cumm_density"], label="cumm density",
                    line_kwargs={"color": "tab:orange"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax, calculations_hist["inst_density"], label="inst density",
                    line_kwargs={"color": "tab:red", "linestyle": ":"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax, sensor_hist["lab_density"], label="lab density",
                    line_kwargs={"color": "tab:blue"}, style = 'step', 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax, sp_hist["cumm_density"], label="SP cumm",
                    line_kwargs={"color": "tab:red", "linestyle": "--"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax, sp_hist["inst_density"], label="SP inst",
                    line_kwargs={"color": "tab:green", "linestyle": "--"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax.set_title("Density")
ax.set_ylabel("Density")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

# -------- Row 3 --------
# Catalyst Inventory
ax = axes[6]
plot_triplet_series(ax, calculations_hist["cat_inventory"], label="cat inventory",
                    line_kwargs={"color": "tab:orange"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax, sp_hist["cat_inventory"], label="SP",
                    line_kwargs={"color": "tab:red", "linestyle": "--"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax.set_title("Catalyst Inventory")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

# Pressure + N2 + Vent
ax = axes[7]
plot_triplet_series(ax, calculations_hist["filtered_pressure"], label="filtered Pressure",
                    line_kwargs={"color": "tab:blue"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax2 = ax.twinx()
plot_triplet_series(ax2, sensor_hist["F_n2"], label="F_n2",
                    line_kwargs={"color": "tab:orange"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax2, sensor_hist["F_vent"], label="F_vent",
                    line_kwargs={"color": "tab:red"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax.set_title("Pressure, N₂, Vent")
ax.set_ylabel("Pressure")
ax2.set_ylabel("Flows")
ax.grid(True, alpha=0.3)
lns1, labs1 = ax.get_legend_handles_labels()
lns2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lns1 + lns2, labs1 + labs2, loc="best")

# Extra slots (can leave empty or plot more feeds if wanted)
ax = axes[8]
plot_triplet_series(ax, sensor_hist["bed_weight"], label="bed_weight",
                    line_kwargs={"color": "tab:blue"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax2 = ax.twinx()
plot_triplet_series(ax2, sensor_hist["bed_level"], label="bed_level",
                    line_kwargs={"color": "tab:orange"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
plot_triplet_series(ax2, sp_hist["bed_level"], label="bed_level SP",
                    line_kwargs={"color": "tab:red", 'linestyle': '--'}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax.set_title("Bed weight vs Bed level")
ax.set_ylabel("Bed weight")
ax2.set_ylabel("Bed level")
ax.grid(True, alpha=0.3)
lns1, labs1 = ax.get_legend_handles_labels()
lns2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lns1 + lns2, labs1 + labs2, loc="best")

ax = axes[9]
plot_triplet_series(ax, sensor_hist["F_cat"], label="F_cat",
                    line_kwargs={"color": "tab:orange"}, 
                    time_converter=lambda t: t/3600.) # seconds to hours
ax.set_title("Fcat")
ax.set_ylabel("Fcat")
lns1, labs1 = ax.get_legend_handles_labels()
ax.legend(lns1, labs1, loc="best")

axes[10].axis("off")
axes[11].axis("off")

# -------- Final touches --------
for ax in (axes[6], axes[7], axes[5]):  # col0, col1, col2
    ax.tick_params(labelbottom=True)
    ax.set_xlabel("Time [h]")

plt.tight_layout()
plt.show()
