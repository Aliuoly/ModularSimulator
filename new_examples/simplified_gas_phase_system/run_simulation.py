from modular_simulation.framework import System
from modular_simulation.plotting import plot_triplet_series
from process_definition import GasPhaseReactorProcessModel
from component_definition import sensors, calculations, controllers
from modular_simulation.utils.wrappers import second, hour, second_value, per_hour
from functools import partial
import logging

#logging.basicConfig(level=logging.DEBUG, format="%(message)s")
#mpl.set_loglevel("warning") # silence matplotlib debug messages


# Assemble the systems
dt = second(30)
system = System(
    dt = dt, 
    process_model = GasPhaseReactorProcessModel(),
    sensors = sensors,
    calculations = calculations,
    controllers = controllers,
)

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    # run till steady state
    system.extend_controller_trajectory("F_cat").hold(100,9.0) 
    system.step(hour(12))

    # cycle 1
    system.extend_controller_trajectory("F_cat").ramp(-2.0, rate = per_hour(1)) # ramp dow 2 kg over 2 hours
    system.step(hour(12))
    # reset to steady state at new cat rate
    system.extend_controller_trajectory("F_cat").ramp(2.0, rate = per_hour(1)) # ramp dow 2 kg over 2 hours
    system.step(hour(12))

    history = system.measured_history
    sensor_hist = history["sensors"]
    calculations_hist = history["calculations"]

    sp_hist = system.setpoint_history

    fig, axes = plt.subplots(4, 2, figsize=(18, 18), sharex=True)
    axes = axes.flatten()

    ax = axes[0]
    ploter = partial(plot_triplet_series, t_start = 10)
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
    ploter(ax, sensor_hist["F_cat"], label="Fcat",style='step', 
                        time_converter=lambda t: t/3600.) # seconds to hours
    ploter(ax, sensor_hist["effective_cat"], label="Fcat", style='step', 
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

        # Pressure
    ax = axes[7]
    ploter(ax, sensor_hist["monomer_rates"], label="monomer rate",
                        time_converter=lambda t: t/3600., array_index=0) # seconds to hours
    ploter(ax, sensor_hist["monomer_rates"], label="comonomer rate",
                        time_converter=lambda t: t/3600., array_index=1) # seconds to hours
    ax.set_title("monomer_rates")
    ax.set_ylabel("monomer_rates")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    ax.legend(lns1, labs1, loc="best")

    plt.tight_layout()
    plt.show()
