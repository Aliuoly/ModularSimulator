from modular_simulation.plant import Plant
from modular_simulation.framework import create_system
from modular_simulation.usables import SampledDelayedSensor
from modular_simulation.control_system import PIDController, Trajectory
from modular_simulation.plotting import plot_triplet_series
import matplotlib.pyplot as plt
from plant_definitions import (
    TankAStates,
    TankAConstants,
    TankAAlgebraicStates,
    TankAControlElements,
    TankASystem,
    TankBAlgebraicStates,
    TankBConstants,
    TankBControlElements,
    TankBStates,
    TankBSystem,
)
import logging


tank_A_states = TankAStates(V_A = 10.)
tank_B_states = TankBStates(V_B = 5.)
tank_A_algebraic_states = TankAAlgebraicStates(F_out_A = 0.1)
tank_B_algebraic_states = TankBAlgebraicStates(F_in_B = 0.1, F_out_B = 0.1)
tank_A_control_elements = TankAControlElements(F_in_A = 1.0)
tank_B_control_elements = TankBControlElements()
tank_A_constants = TankAConstants(Cv_F_out_A = 0.1)
tank_B_constants = TankBConstants(Cv_F_out_B = 0.15)

tank_A_sensors = [
    SampledDelayedSensor(measurement_tag = "V_A"),
    SampledDelayedSensor(measurement_tag = "F_in_A"),
    SampledDelayedSensor(measurement_tag = "F_out_A"),
]
tank_B_sensors = [
    SampledDelayedSensor(measurement_tag = "V_B"),
    SampledDelayedSensor(measurement_tag = "F_in_B"),
    SampledDelayedSensor(measurement_tag = "F_out_B")
]

tank_A_controllers = [
    PIDController(
        cv_tag = "V_A",
        mv_tag = "F_in_A",
        sp_trajectory = Trajectory(10.0).hold(150).step(10.0),
        mv_range = (0, 10.),
        Kp = 0.1,
        Ti = 10.,
    )
]
tank_B_controllers = []

tank_A_system = create_system(
    system_class = TankASystem,
    dt = 1.0,
    initial_states = tank_A_states,
    initial_algebraic = tank_A_algebraic_states,
    initial_controls = tank_A_control_elements,
    system_constants = tank_A_constants,
    sensors = tank_A_sensors,
    calculations = [],
    controllers = tank_A_controllers,
)

tank_B_system = create_system(
    system_class = TankBSystem,
    dt = 1.0,
    initial_states = tank_B_states,
    initial_algebraic = tank_B_algebraic_states,
    initial_controls = tank_B_control_elements,
    system_constants = tank_B_constants,
    sensors = tank_B_sensors,
    calculations = [],
    controllers = tank_B_controllers,
)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s"
    )
plant = Plant(dt = 1.0, systems = [tank_A_system, tank_B_system])

    
plant.step(300)

hist = plant._composite_system.measured_history
measured_hist = hist["sensors"]
sp_hist = plant._composite_system.setpoint_history
plt.figure(figsize=(10,10))
ax = plt.subplot(2,2,1)
plot_triplet_series(
    ax,
    measured_hist["V_A"],
    label='V_A'
)
plt.legend()
plot_triplet_series(
    ax,
    sp_hist["V_A"],
    line_kwargs = {'c':'r'},
    label='SP'
)
plt.legend()
ax = plt.subplot(2,2,2)
plot_triplet_series(
    ax,
    measured_hist["F_in_A"],
    label='F_in_A',
)
plt.legend()
ax = plt.subplot(2,2,3)
plot_triplet_series(
    ax, 
    measured_hist["V_B"],
    label='V_B',
)
plt.legend()
ax = plt.subplot(2,2,4)
plot_triplet_series(
    ax,
    measured_hist["F_in_B"],
    label='F_in_B',
)
plt.legend()
plt.tight_layout()
plt.show()