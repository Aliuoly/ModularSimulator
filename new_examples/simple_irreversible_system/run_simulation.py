from modular_simulation.framework import System
from modular_simulation.plotting import plot_triplet_series
from process_definition import IrreversibleProcessModel
from component_definition import sensors, calculations, controllers
from modular_simulation.utils.wrappers import second, day
import logging


# --- 2. Assemble and Initialize the System ---
dt = second(30)
system = System(
    dt = dt, 
    process_model = IrreversibleProcessModel(),
    sensors = sensors,
    calculations = calculations,
    controllers = controllers,
)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
    )
    mpl.set_loglevel("warning") # silence matplotlib debug messages
    
    
    pv_kwargs = {'linestyle': '-'}
    sp_kwargs = {'linestyle': '--', "alpha": 0.5}
    plt.figure(figsize=(12, 8))
    # --- First simulation run ---
    system.step(day(1))
    # --- Change the setpoint and continue the simulation ---
    system.extend_controller_trajectory(cv_tag="B", value=0.2)
    system.step(day(1))
            

    # 3. Plot the results.
    # =====================

    # The simulation history is stored in the system's `_history` attribute.
    history = system.measured_history  #type: ignore
    sensor_hist = history["sensors"]
    calc_hist = history["calculations"]
    sp_hist= system.setpoint_history
    # Plot Concentration of B
    ax = plt.subplot(2, 2, 1)
    plot_triplet_series(
        ax,
        sensor_hist["B"],
        style="step",
        line_kwargs=pv_kwargs,
    )
    plot_triplet_series(
        ax,
        sp_hist["B.sp"],
        style="step",
        line_kwargs=sp_kwargs,
    )
    plt.title("Concentration of B")
    plt.xlabel("Time ")
    plt.ylabel("[B] (mol/L)")
    plt.grid(True)

    # Plot Inlet Flow Rate (F_in)
    ax = plt.subplot(2, 2, 2)
    plot_triplet_series(
        ax,
        sensor_hist["F_in"],
        style="step",
        line_kwargs=pv_kwargs,
    )
    plt.title("Inlet Flow Rate (F_in)")
    plt.xlabel("Time ")
    plt.ylabel("Flow (L/s)")
    plt.grid(True)

    # Plot Reactor Volume (V)
    ax = plt.subplot(2, 2, 3)
    plot_triplet_series(
        ax,
        sensor_hist["V"],
        style="step",
        line_kwargs=pv_kwargs,
    )
    plt.title("Reactor Volume (V)")
    plt.xlabel("Time ")
    plt.ylabel("Volume (L)")
    plt.grid(True)

    # Plot Outlet Flow Rate (F_out)
    ax = plt.subplot(2, 2, 4)
    plot_triplet_series(
        ax,
        sensor_hist["F_out"],
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
