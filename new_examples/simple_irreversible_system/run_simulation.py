# pyright: reportImplicitRelativeImport=false
from modular_simulation.framework import System
from modular_simulation.plotting import plot_triplet_series
from process_definition import IrreversibleProcessModel
from component_definition import sensors, calculations, control_elements
from modular_simulation.utils.wrappers import second, day
from functools import partial
import logging


# --- 2. Assemble and Initialize the System ---
dt = second(30)
system = System(
    dt=dt,
    process_model=IrreversibleProcessModel(),
    sensors=sensors,
    calculations=calculations,
    control_elements=control_elements,
)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mpl.set_loglevel("warning")  # silence matplotlib debug messages

    pv_kwargs = {"linestyle": "-"}
    sp_kwargs = {"linestyle": "--", "alpha": 0.5}
    plt.figure(figsize=(12, 8))
    # --- First simulation run ---
    system.step(day(1))
    # # --- Change the setpoint and continue the simulation ---
    system.extend_controller_sp_trajectory(cv_tag="B", value=0.2)
    system.step(day(1))
    system.extend_controller_sp_trajectory(cv_tag="B").ramp(0.3, rate=0.1 / day(1))
    system.step(day(4))

    # 3. Plot the results.
    # =====================

    # The simulation history is stored in the system's `_history` attribute.
    history = system.history
    # Plot Concentration of B
    plot_in_days = partial(plot_triplet_series, time_converter=lambda t: t / second(day(1)))
    ax = plt.subplot(2, 2, 1)
    plot_in_days(
        ax,
        history["B"],
        style="step",
        line_kwargs=pv_kwargs,
    )
    plot_in_days(
        ax,
        history["B.sp"],
        style="step",
        line_kwargs=sp_kwargs,
    )
    plt.title("Concentration of B")
    plt.xlabel("Time ")
    plt.ylabel("[B] (mol/L)")
    plt.grid(True)

    # Plot Inlet Flow Rate (F_in)
    ax = plt.subplot(2, 2, 2)
    plot_in_days(
        ax,
        history["F_in"],
        style="step",
        line_kwargs=pv_kwargs,
    )
    plt.title("Inlet Flow Rate (F_in)")
    plt.xlabel("Time ")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    # Plot Reactor Volume (V)
    ax = plt.subplot(2, 2, 3)
    plot_in_days(
        ax,
        history["V"],
        style="step",
        line_kwargs=pv_kwargs,
    )
    plt.title("Reactor Volume (V)")
    plt.xlabel("Time ")
    plt.ylabel("Volume (L)")
    plt.grid(True)

    # Plot Outlet Flow Rate (F_out)
    ax = plt.subplot(2, 2, 4)
    plot_in_days(
        ax,
        history["F_out"],
        style="step",
        line_kwargs=pv_kwargs,
    )
    plt.title("Outlet Flow Rate (F_out)")
    plt.xlabel("Time ")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()
