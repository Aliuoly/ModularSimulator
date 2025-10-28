from modular_simulation.framework import create_system
from modular_simulation.plotting import plot_triplet_series
from system_definition import (
    GasPhaseReactorSystem,
    GasPhaseReactorStates,
    GasPhaseReactorAlgebraicStates,
    GasPhaseReactorConstants,
    GasPhaseReactorControlElements
)
from sensors import sensors
from calculations import calculations
from controllers import controllers
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.units import Unit
import logging
from functools import partial
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

if __name__ == '__main__':
    normal_system.step(12 * Unit("hour"))
    # extend controller still uses dt units
    normal_system.extend_controller_trajectory("rM2").hold(3600.).hold(3600., 0.25)
    normal_system.extend_controller_trajectory("pM1").hold(3600.*6).hold(3600., 680)
    normal_system.step(12 * Unit("hour"))
    history = normal_system.measured_history
    sensor_hist = history["sensors"]
    calculations_hist = history["calculations"]

    sp_hist = normal_system.setpoint_history

    fig, axes = plt.subplots(4, 2, figsize=(18, 18), sharex=True)
    axes = axes.flatten()

    ax = axes[0]
    ploter = partial(plot_triplet_series, t_start = 12, t_end = 24)
    ploter(ax, sensor_hist["F_m1"], label="F_m1",
                        line_kwargs={"color": "tab:blue"}, 
                        time_converter=lambda t: t/3600.) # seconds to hours
    ax.set_title("F_m1")
    ax.set_ylabel("F_m1")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    ax.legend(lns1, labs1, loc="best")

    ax = axes[2]
    ploter(ax, calculations_hist["pM1"], label="pM1",
                        line_kwargs={"color": "tab:orange"}, 
                        time_converter=lambda t: t/3600.) # seconds to hours
    ploter(ax, sp_hist["pM1.sp"], label="SP pM1",
                        line_kwargs={"color": "tab:red", "linestyle": "--"}, 
                        time_converter=lambda t: t/3600.) # seconds to hours
    ax.set_ylabel("pM1")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    ax.legend(lns1, labs1, loc="best")

        # (F_m2, rM2)
    ax = axes[1]
    ploter(ax, sensor_hist["F_m2"], label="F_m2",
                        line_kwargs={"color": "tab:blue"}, 
                        time_converter=lambda t: t/3600.) # seconds to hours
    ax.set_title("F_m2")
    ax.set_ylabel("F_m2")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    ax.legend(lns1, labs1, loc="best")

    ax = axes[3]
    ploter(ax, calculations_hist["rM2"], label="rM2",
                        line_kwargs={"color": "tab:orange"}, style='step', 
                        time_converter=lambda t: t/3600.) # seconds to hours
    ploter(ax, sp_hist["rM2.sp"], label="SP rM2",
                        line_kwargs={"color": "tab:red", "linestyle": "--"}, 
                        time_converter=lambda t: t/3600.) # seconds to hours
    ax.set_title("rM2")
    ax.set_ylabel("rM2")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    ax.legend(lns1, labs1, loc="best")

    ax = axes[4]
    ploter(ax, sensor_hist["mass_prod_rate"], label="Prod Rate",
                        line_kwargs={"color": "royalblue"}, 
                        time_converter=lambda t: t/3600.) # seconds to hours
    ax.set_title("Prod Rate")
    ax.set_ylabel("Prod Rate")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    ax.legend(lns1, labs1, loc="best")
    ax = axes[6]
    ploter(ax, sensor_hist["F_cat"], label="Fcat",
                        line_kwargs={"color": "royalblue"}, style='step', 
                        time_converter=lambda t: t/3600.) # seconds to hours
    
    ax.set_title("F_cat")
    ax.set_ylabel("Fcat")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    ax.legend(lns1, labs1, loc="best")

    # Pressure
    ax = axes[5]
    ploter(ax, sensor_hist["pressure"], label="Pressure",
                        line_kwargs={"color": "tab:blue", 'alpha': 0.5}, 
                        time_converter=lambda t: t/3600.) # seconds to hours
    ax.set_title("Pressure")
    ax.set_ylabel("Pressure")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    ax.legend(lns1, labs1, loc="best")

    plt.tight_layout()
    plt.show()