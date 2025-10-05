from typing import List

import numpy as np
from modular_simulation.control_system import (
    PIDController, 
    Trajectory, 
    InternalModelController, 
    CalculationModelPath,
    FirstOrderTrajectoryController,
    BangBangController
)
from modular_simulation.control_system.controller import Controller
from calculations.misc_calculations import CatInventoryEstimator, AlTiRatioEstimator
from calculations.property_estimator import PropertyEstimator

prod_rate_controller = InternalModelController(
    mv_tag="F_cat",
    cv_tag="cat_inventory",
    sp_trajectory=Trajectory(0.0),
    mv_range=(0.0, 20.0),
    model = CalculationModelPath(
        calculation_name = CatInventoryEstimator,
        method_name = "model"
    ),
    cascade_controller = PIDController(
        mv_tag = "cat_inventory",
        cv_tag = "mass_prod_rate",
        sp_trajectory = Trajectory(0.0).ramp(50, ramprate=10/3600.),
        mv_range = (0.0, 1e9),
        Kp = 250.,
        Ti = 4500.,
        Td = 0.,
    )
)
pM1_controller = PIDController(
    mv_tag="F_m1",
    cv_tag="pM1",
    sp_trajectory=Trajectory(700.0),
    mv_range=(0.0, 70_000), # kg/h
    Kp=3.454*3600*28/1000,
    Ti=270.0*3600*28/1000,
    Td=0,
)
density_controller = PIDController(
    mv_tag="F_m2",
    cv_tag="rM2",
    sp_trajectory=Trajectory(0.3),
    mv_range=(0.0, 7_000), #kg/h
    Kp=55.0*3600*56/1000,
    Ti=35.0*3600*56/1000,
    Td=0,
    cascade_controller=InternalModelController(
        mv_tag = "rM2",
        cv_tag = "inst_density",
        sp_trajectory=Trajectory(918.0),
        mv_range = (0.0, 0.6),
        model = CalculationModelPath(
            calculation_name = PropertyEstimator,
            method_name = "inst_density_model",
        ),
        cascade_controller = FirstOrderTrajectoryController(
            mv_tag = "inst_density",
            cv_tag = "cumm_density",
            sp_trajectory = Trajectory(918.0),
            mv_range = (905.0, 965.0),
            closed_loop_time_constant_fraction=0.8,
            open_loop_time_constant="residence_time",
        ),
    )
)
MI_controller = PIDController(
    mv_tag="F_h2",
    cv_tag="rH2",
    sp_trajectory=Trajectory(0.0),
    mv_range=(0., 15), #kg/h
    Kp=26.0*3600*2/1000,
    Ti=748.0*3600*2/1000,
    Td=0,
    cascade_controller=InternalModelController(
        mv_tag = "rH2",
        cv_tag = "inst_MI",
        sp_trajectory=Trajectory(2.0),
        mv_range = (0, 0.9),
        model = CalculationModelPath(
            calculation_name = PropertyEstimator,
            method_name = "inst_MI_model",
        ),
        cascade_controller = FirstOrderTrajectoryController(
            mv_tag = "inst_MI",
            cv_tag = "cumm_MI",
            sp_trajectory = Trajectory(2.0),
            mv_range = (0.2, 50.0),
            closed_loop_time_constant_fraction=0.8,
            open_loop_time_constant="residence_time",
        ),
    )
)
Al_Ti_ratio_controller = InternalModelController(
    mv_tag="F_teal",
    cv_tag="Al_Ti_ratio",
    sp_trajectory=Trajectory(0.4),
    mv_range=(0.0, 1e5),
    model=CalculationModelPath(
        calculation_name = AlTiRatioEstimator,
        method_name = "AlTi_model"
    )
)
under_pressure_controller = PIDController(
    mv_tag="F_n2",
    cv_tag="pressure",
    sp_trajectory=Trajectory(2300.0),
    mv_range=(0.0, 500.0), # kg/h
    Kp= 500 / 200.0,
    Ti=np.inf,
    Td=0.0,
)
over_pressure_controller = PIDController(
    mv_tag="F_vent",
    cv_tag="pressure",
    sp_trajectory=Trajectory(2300.0),
    mv_range=(0.0, 250_000), # L/s
    Kp= 250_000.0 / 200.0,
    Ti=np.inf,
    Td=0.0,
    inverted=True,
)
discharge_level_controller = BangBangController( 
    mv_tag="discharge_valve_position",
    cv_tag="bed_level",
    sp_trajectory=Trajectory(15.0), #m
    deadband = 0.1,
    mv_range = (0.0, 1.0), 
    alpha = 0.4,
    inverted = True,
)


controllers: List[Controller] = [
    Al_Ti_ratio_controller,
    prod_rate_controller,
    pM1_controller,
    density_controller,
    MI_controller,
    under_pressure_controller,
    over_pressure_controller,
    discharge_level_controller,
]