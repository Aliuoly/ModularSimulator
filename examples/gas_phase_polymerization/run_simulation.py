from __future__ import annotations

import logging
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.units import Quantity, Unit

from modular_simulation.core import create_system
from modular_simulation.plotting import plot_triplet_series

from .controller_definitions import create_controllers
from .system_definitions import GasPhaseReactorModel
from .usable_definitions import create_calculations, create_sensors


def make_system(**overrides: Any):
    """Instantiate the gas-phase reactor simulation."""

    model = GasPhaseReactorModel()
    dt = 30.0 * Unit("s")

    system = create_system(
        dt=dt,
        dynamic_model=model,
        sensors=create_sensors(),
        calculations=create_calculations(),
        controllers=create_controllers(),
        use_numba=overrides.get("use_numba", False),
    )
    return system


def plot_results(system) -> None:
    """Visualize the instrumentation and set-point history."""

    history = system.measured_history
    sensor_hist = history["sensors"]
    calculations_hist = history["calculations"]
    sp_hist = system.setpoint_history

    fig, axes = plt.subplots(4, 3, figsize=(20, 18), sharex=True)
    axes = axes.flatten()

    # -------- Row 1: Flow–Property Pairs --------
    ax = axes[0]
    plot_triplet_series(
        ax,
        sensor_hist["F_m1"],
        label="F_m1",
        line_kwargs={"color": "tab:blue"},
        time_converter=lambda t: t / 3600.0,
    )
    ax2 = ax.twinx()
    plot_triplet_series(
        ax2,
        calculations_hist["pM1"],
        label="pM1",
        line_kwargs={"color": "tab:orange"},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax2,
        sp_hist["pM1.sp"],
        label="SP pM1",
        line_kwargs={"color": "tab:red", "linestyle": "--"},
        time_converter=lambda t: t / 3600.0,
    )
    ax.set_title("F_m1 vs pM1")
    ax.set_ylabel("F_m1")
    ax2.set_ylabel("pM1")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    lns2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lns1 + lns2, labs1 + labs2, loc="best")

    ax = axes[1]
    plot_triplet_series(
        ax,
        sensor_hist["F_m2"],
        label="F_m2",
        line_kwargs={"color": "tab:blue"},
        time_converter=lambda t: t / 3600.0,
    )
    ax2 = ax.twinx()
    plot_triplet_series(
        ax2,
        calculations_hist["rM2"],
        label="rM2",
        line_kwargs={"color": "tab:orange"},
        style="step",
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax2,
        sp_hist["rM2.sp"],
        label="SP rM2",
        line_kwargs={"color": "tab:red", "linestyle": "--"},
        time_converter=lambda t: t / 3600.0,
    )
    ax.set_title("F_m2 vs rM2")
    ax.set_ylabel("F_m2")
    ax2.set_ylabel("rM2")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    lns2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lns1 + lns2, labs1 + labs2, loc="best")

    ax = axes[2]
    plot_triplet_series(
        ax,
        sensor_hist["F_h2"],
        label="F_h2",
        line_kwargs={"color": "tab:blue"},
        time_converter=lambda t: t / 3600.0,
    )
    ax2 = ax.twinx()
    plot_triplet_series(
        ax2,
        calculations_hist["rH2"],
        label="rH2",
        line_kwargs={"color": "tab:orange"},
        style="step",
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax2,
        sp_hist["rH2.sp"],
        label="SP rH2",
        line_kwargs={"color": "tab:red", "linestyle": "--"},
        time_converter=lambda t: t / 3600.0,
    )
    ax.set_title("F_h2 vs rH2")
    ax.set_ylabel("F_h2")
    ax2.set_ylabel("rH2")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    lns2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lns1 + lns2, labs1 + labs2, loc="best")

    ax = axes[3]
    plot_triplet_series(
        ax,
        sensor_hist["mass_prod_rate"],
        label="Prod Rate",
        line_kwargs={"color": "royalblue", "alpha": 0.5},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        calculations_hist["filtered_mass_prod_rate"],
        label="filtered Prod Rate",
        line_kwargs={"color": "green"},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        sp_hist["filtered_mass_prod_rate.sp"],
        label="SP Prod Rate",
        line_kwargs={"color": "tab:green", "linestyle": "--"},
        time_converter=lambda t: t / 3600.0,
    )
    ax.set_title("Prod Rate vs F_cat")
    ax.set_ylabel("Prod Rate")
    ax.grid(True, alpha=0.3)
    lns1, labs1 = ax.get_legend_handles_labels()
    ax.legend(lns1, labs1, loc="best")

    ax = axes[4]
    plot_triplet_series(
        ax,
        calculations_hist["cumm_MI"],
        label="cumm MI",
        line_kwargs={"color": "tab:orange"},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        calculations_hist["inst_MI"],
        label="inst MI",
        line_kwargs={"color": "tab:red", "linestyle": ":"},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        sensor_hist["lab_MI"],
        label="lab MI",
        line_kwargs={"color": "tab:blue"},
        style="step",
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        sp_hist["cumm_MI.sp"],
        label="SP MI",
        line_kwargs={"color": "tab:red", "linestyle": "--"},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        sp_hist["inst_MI.sp"],
        label="SP inst",
        line_kwargs={"color": "tab:green", "linestyle": "--"},
        time_converter=lambda t: t / 3600.0,
    )
    ax.set_title("Melt Index")
    ax.set_ylabel("MI")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[5]
    plot_triplet_series(
        ax,
        calculations_hist["cumm_density"],
        label="cumm density",
        line_kwargs={"color": "tab:orange"},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        calculations_hist["inst_density"],
        label="inst density",
        line_kwargs={"color": "tab:red", "linestyle": ":"},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        sensor_hist["lab_density"],
        label="lab density",
        line_kwargs={"color": "tab:blue"},
        style="step",
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        sp_hist["cumm_density.sp"],
        label="SP cumm",
        line_kwargs={"color": "tab:red", "linestyle": "--"},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        sp_hist["inst_density.sp"],
        label="SP inst",
        line_kwargs={"color": "tab:green", "linestyle": "--"},
        time_converter=lambda t: t / 3600.0,
    )
    ax.set_title("Density")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[6]
    plot_triplet_series(
        ax,
        calculations_hist["cat_inventory"],
        label="cat inventory",
        line_kwargs={"color": "tab:orange"},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        sp_hist["cat_inventory.sp"],
        label="SP",
        line_kwargs={"color": "tab:red", "linestyle": "--"},
        time_converter=lambda t: t / 3600.0,
    )
    ax.set_title("Catalyst Inventory")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[7]
    plot_triplet_series(
        ax,
        calculations_hist["filtered_pressure"],
        label="filtered Pressure",
        line_kwargs={"color": "green"},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax,
        sensor_hist["pressure"],
        label="Pressure",
        line_kwargs={"color": "tab:blue", "alpha": 0.5},
        time_converter=lambda t: t / 3600.0,
    )
    ax2 = ax.twinx()
    plot_triplet_series(
        ax2,
        sensor_hist["F_n2"],
        label="F_n2",
        line_kwargs={"color": "tab:orange"},
        time_converter=lambda t: t / 3600.0,
    )
    plot_triplet_series(
        ax2,
        sensor_hist["F_vent"],
        label="F_vent",
        line_kwargs={"color": "tab:red"},
        time_converter=lambda t: t / 3600.0,
    )
    ax.set_title("Pressure, N₂, Vent")
    ax.set_ylabel("Pressure")
    ax2.set_ylabel("Flows")
    ax.grid(True, alpha=0.3)

    for idx in range(8, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mpl.set_loglevel("warning")

    system = make_system()
    system.step(1 * Unit("hour"))
    logging.info("System snapshot: %s", system)

    system.extend_controller_trajectory(cv_tag="pM1", value=700.0 * Unit("kPa"))
    system.step(Quantity(8.0, Unit("hour")))
    plot_results(system)

    plt.show()
