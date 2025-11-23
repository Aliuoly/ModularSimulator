"""Simulation entry-point for the gas-phase polymerization example."""

from __future__ import annotations

import logging
from functools import partial

import matplotlib.pyplot as plt
import matplotlib as mpl

from modular_simulation.framework import System
from modular_simulation.plotting import plot_triplet_series
from modular_simulation.utils.wrappers import hour, second

from process_definition import GasPhaseReactorProcessModel
from component_definition import sensors, calculations, control_elements


def create_system(use_numba: bool = False) -> System:
    return System(
        dt=second(30),
        process_model=GasPhaseReactorProcessModel(),
        sensors=sensors,
        calculations=calculations,
        control_elements=control_elements,
        use_numba=use_numba,
    )


system = create_system()


def plot(system: System, t_start: float) -> None:
    history = system.history

    fig, axes = plt.subplots(4, 3, figsize=(20, 18), sharex=True)
    axes = axes.flatten()
    to_hours = partial(
        plot_triplet_series, time_converter=lambda t: t / second(hour(1)), t_start=t_start
    )

    ax = axes[0]
    to_hours(ax, history["F_m1"], label="F_m1", line_kwargs={"color": "tab:blue"})
    ax2 = ax.twinx()
    to_hours(ax2, history["pM1"], label="pM1", line_kwargs={"color": "tab:orange"})
    to_hours(
        ax2, history["pM1.sp"], label="SP pM1", line_kwargs={"color": "tab:red", "linestyle": "--"}
    )
    ax.set_title("F_m1 vs pM1")
    ax.set_ylabel("F_m1")
    ax2.set_ylabel("pM1")
    ax.grid(True, alpha=0.3)
    ax.legend(
        ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0],
        ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1],
        loc="best",
    )

    ax = axes[1]
    to_hours(ax, history["F_m2"], label="F_m2", line_kwargs={"color": "tab:blue"})
    ax2 = ax.twinx()
    to_hours(ax2, history["rM2"], label="rM2", line_kwargs={"color": "tab:orange"}, style="step")
    to_hours(
        ax2, history["rM2.sp"], label="SP rM2", line_kwargs={"color": "tab:red", "linestyle": "--"}
    )
    ax.set_title("F_m2 vs rM2")
    ax.set_ylabel("F_m2")
    ax2.set_ylabel("rM2")
    ax.grid(True, alpha=0.3)
    ax.legend(
        ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0],
        ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1],
        loc="best",
    )

    ax = axes[2]
    to_hours(ax, history["F_h2"], label="F_h2", line_kwargs={"color": "tab:blue"})
    ax2 = ax.twinx()
    to_hours(ax2, history["rH2"], label="rH2", line_kwargs={"color": "tab:orange"}, style="step")
    to_hours(
        ax2, history["rH2.sp"], label="SP rH2", line_kwargs={"color": "tab:red", "linestyle": "--"}
    )
    ax.set_title("F_h2 vs rH2")
    ax.set_ylabel("F_h2")
    ax2.set_ylabel("rH2")
    ax.grid(True, alpha=0.3)
    ax.legend(
        ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0],
        ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1],
        loc="best",
    )

    ax = axes[3]
    to_hours(
        ax,
        history["mass_prod_rate"],
        label="Prod Rate",
        line_kwargs={"color": "royalblue", "alpha": 0.5},
    )
    to_hours(
        ax,
        history["filtered_mass_prod_rate"],
        label="filtered Prod Rate",
        line_kwargs={"color": "green"},
    )
    to_hours(
        ax,
        history["filtered_mass_prod_rate.sp"],
        label="SP Prod Rate",
        line_kwargs={"color": "tab:green", "linestyle": "--"},
    )
    ax.set_title("Prod Rate vs F_cat")
    ax.set_ylabel("Prod Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[4]
    to_hours(ax, history["cumm_MI"], label="cumm MI", line_kwargs={"color": "tab:orange"})
    to_hours(
        ax, history["inst_MI"], label="inst MI", line_kwargs={"color": "tab:red", "linestyle": ":"}
    )
    to_hours(
        ax,
        history["cumm_MI.sp"],
        label="SP MI",
        line_kwargs={"color": "tab:red", "linestyle": "--"},
    )
    to_hours(
        ax,
        history["inst_MI.sp"],
        label="SP inst",
        line_kwargs={"color": "tab:green", "linestyle": "--"},
    )
    to_hours(
        ax,
        history["lab_MI"],
        label="lab MI",
        line_kwargs={"color": "tab:blue", "scaley": False},
        style="step",
    )
    ax.set_title("Melt Index")
    ax.set_ylabel("MI")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[5]
    to_hours(ax, history["cumm_density"], label="cumm density", line_kwargs={"color": "tab:orange"})
    to_hours(
        ax,
        history["inst_density"],
        label="inst density",
        line_kwargs={"color": "tab:red", "linestyle": ":"},
    )
    to_hours(
        ax,
        history["cumm_density.sp"],
        label="SP cumm",
        line_kwargs={"color": "tab:red", "linestyle": "--"},
    )
    to_hours(
        ax,
        history["inst_density.sp"],
        label="SP inst",
        line_kwargs={"color": "tab:green", "linestyle": "--"},
    )
    to_hours(
        ax,
        history["lab_density"],
        label="lab density",
        line_kwargs={"color": "tab:blue"},
        style="step",
    )
    ax.set_title("Density")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[6]
    to_hours(
        ax, history["cat_inventory"], label="cat inventory", line_kwargs={"color": "tab:orange"}
    )
    to_hours(
        ax,
        history["cat_inventory.sp"],
        label="SP",
        line_kwargs={"color": "tab:red", "linestyle": "--"},
    )
    ax.set_title("Catalyst Inventory")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[7]
    to_hours(
        ax, history["filtered_pressure"], label="filtered Pressure", line_kwargs={"color": "green"}
    )
    to_hours(
        ax, history["pressure"], label="Pressure", line_kwargs={"color": "tab:blue", "alpha": 0.5}
    )
    ax2 = ax.twinx()
    to_hours(ax2, history["F_n2"], label="F_n2", line_kwargs={"color": "tab:orange"})
    to_hours(ax2, history["F_vent"], label="F_vent", line_kwargs={"color": "tab:red"})
    ax.set_title("Pressure, N₂, Vent")
    ax.set_ylabel("Pressure")
    ax2.set_ylabel("Flows")
    ax.grid(True, alpha=0.3)
    ax.legend(
        ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0],
        ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1],
        loc="best",
    )

    ax = axes[8]
    to_hours(ax, history["bed_weight"], label="bed_weight", line_kwargs={"color": "tab:blue"})
    ax2 = ax.twinx()
    to_hours(ax2, history["bed_level"], label="bed_level", line_kwargs={"color": "tab:orange"})
    to_hours(
        ax2,
        history["bed_level.sp"],
        label="bed_level SP",
        line_kwargs={"color": "tab:red", "linestyle": "--"},
    )
    ax.set_title("Bed weight vs Bed level")
    ax.set_ylabel("Bed weight")
    ax2.set_ylabel("Bed level")
    ax.grid(True, alpha=0.3)
    ax.legend(
        ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0],
        ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1],
        loc="best",
    )

    ax = axes[9]
    to_hours(ax, history["F_cat"], label="F_cat", line_kwargs={"color": "tab:orange"})
    ax.set_title("Fcat")
    ax.set_ylabel("Fcat")
    ax.legend(loc="best")

    axes[10].axis("off")
    axes[11].axis("off")

    for ax in (axes[6], axes[7], axes[8]):
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Time [h]")

    plt.tight_layout()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mpl.set_loglevel("warning")
    system.step(duration=hour(48))
    system.extend_control_element_mv_trajectory("F_cat").hold(hour(3)).step(-2)
    system.step(duration=hour(24))
    plot(system, t_start=24)
    plt.show()
