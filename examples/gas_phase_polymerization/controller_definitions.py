from typing import List
import numpy as np
from astropy.units import Unit, Quantity #type:ignore
from modular_simulation.control_system import (
    PIDController,
    Trajectory,
    InternalModelController,
    CalculationModelPath,
    FirstOrderTrajectoryController,
    BangBangController,
)
from modular_simulation.control_system.controller import Controller
from calculations.misc_calculations import CatInventoryEstimator, AlTiRatioEstimator
from calculations.property_estimator import PropertyEstimator


controllers: List[Controller] = [

    # --- Catalyst Inventory / Production Rate Control ---
    InternalModelController(
        mv_tag="F_cat",
        cv_tag="cat_inventory",
        sp_trajectory=Trajectory(0.0, unit=Unit("kg")),
        mv_range=(0.0 * Unit("kg/h"), 20.0 * Unit("kg/h")),
        model=CalculationModelPath(
            calculation_name=CatInventoryEstimator,
            method_name="model",
        ),
        cascade_controller=PIDController(
            mv_tag="cat_inventory",
            cv_tag="filtered_mass_prod_rate",
            sp_trajectory=Trajectory(0.0, unit=Unit("tonne/hour")).ramp(
                50.0, ramprate=25.0 / 3600.0 
                #TODO: ramp magnitude and ramp rate are always assumed rate is currently always unit / system time unit. Newed to make this unit aware as well.
            ), 
            mv_range=(0.0 * Unit("kg"), 80 * Unit("kg")), # 20kg/h, 4 hour res time -> 0 = F - inv/tau -> max inv = F * tau = 80kg
            Kp=0.5, # expect about 2 ton pr per kg of cat inv -> for error = 1, want 0.5kg of inv
            Ti=3600.0, 
            Td=0.0,
            setpoint_weight = 0.0
        ),
    ),

    # --- Monomer 1 Partial Pressure Control ---
    PIDController(
        mv_tag="F_m1",
        cv_tag="pM1",
        sp_trajectory=Trajectory(700.0, unit=Unit("kPa")),
        mv_range=(0.0 * Unit("kg/h"), 70_000.0 * Unit("kg/h")),
        Kp=50, #kg/h per kPa of error in pM1. Typicaly error < 20kPa, and expected mv to move maybe 1 ton there.
        Ti=3600, # in system units, which is seconds
        Td=0.0,
    ),

    # --- Density / Comonomer Ratio Cascade ---
    PIDController(
        mv_tag="F_m2",
        cv_tag="rM2",
        sp_trajectory=Trajectory(0.3, unit=Unit()),
        mv_range=(0.0 * Unit("kg/h"), 7_000.0 * Unit("kg/h")),
        Kp=3500, #kg/h / mol ratio of error. Typical error < 0.1, so halve the range seems reasonable
        Ti=1800, #Kp/Ti kg/h per 1 unit of error accumulated. Just gonna pick like expected tc / 2 or something
        Td=0.0,
        setpoint_weight = 0.0,
        cascade_controller=InternalModelController(
            mv_tag="rM2",
            cv_tag="inst_density",
            sp_trajectory=Trajectory(918.0, unit=Unit("kg/m^3")),
            mv_range=(0.0 * Unit(), 0.6 * Unit()),
            model=CalculationModelPath(
                calculation_name=PropertyEstimator,
                method_name="inst_density_model",
            ),
            cascade_controller=FirstOrderTrajectoryController(
                mv_tag="inst_density",
                cv_tag="cumm_density",
                sp_trajectory=Trajectory(918.0, unit=Unit("kg/m^3")),
                mv_range=(905.0 * Unit("kg/m^3"), 965.0 * Unit("kg/m^3")),
                closed_loop_time_constant_fraction=0.8,
                open_loop_time_constant="residence_time",
            ),
        ),
    ),

    # --- Melt Index / Hydrogen Ratio Cascade ---
    PIDController(
        mv_tag="F_h2",
        cv_tag="rH2",
        sp_trajectory=Trajectory(0.0, unit=Unit()),
        mv_range=(0.0 * Unit("kg/h"), 15.0 * Unit("kg/h")),
        Kp=200,
        Ti=5400,
        Td=0.0,
        setpoint_weight = 0.0,
        cascade_controller=InternalModelController(
            mv_tag="rH2",
            cv_tag="inst_MI",
            sp_trajectory=Trajectory(2.0, unit=Unit()),
            mv_range=(0.0 * Unit(), 0.9 * Unit()),
            model=CalculationModelPath(
                calculation_name=PropertyEstimator,
                method_name="inst_MI_model",
            ),
            cascade_controller=FirstOrderTrajectoryController(
                mv_tag="inst_MI",
                cv_tag="cumm_MI",
                sp_trajectory=Trajectory(2.0, unit=Unit()),
                mv_range=(0.2 * Unit(), 50.0 * Unit()),
                closed_loop_time_constant_fraction=0.8,
                open_loop_time_constant="residence_time",
            ),
        ),
    ),

    # --- Al/Ti Ratio Control ---
    InternalModelController(
        mv_tag="F_teal",
        cv_tag="Al_Ti_ratio",
        sp_trajectory=Trajectory(0.4, unit=Unit()),
        mv_range=(0.0 * Unit("mol/s"), 1e5 * Unit("mol/s")),
        model=CalculationModelPath(
            calculation_name=AlTiRatioEstimator,
            method_name="AlTi_model",
        ),
    ),

    # --- Underpressure Nitrogen Flow Control ---
    PIDController(
        mv_tag="F_n2",
        cv_tag="filtered_pressure",
        sp_trajectory=Trajectory(2300.0, unit=Unit("kPa")),
        mv_range=(0.0 * Unit("kg/h"), 500.0 * Unit("kg/h")),
        Kp=500.0 / 200.0,
        Ti=np.inf,
        Td=0.0,
    ),

    # --- Overpressure Vent Flow Control ---
    PIDController(
        mv_tag="F_vent",
        cv_tag="filtered_pressure",
        sp_trajectory=Trajectory(2300.0, unit=Unit("kPa")),
        mv_range=(0.0 * Unit("L/hr"), 250_00.0 * Unit("L/hr")),
        Kp=250_00.0 / 200.0,
        Ti=np.inf,
        Td=0.0,
        inverted=True,
    ),

    # --- Discharge Level Bang-Bang Control ---
    BangBangController(
        mv_tag="discharge_valve_position",
        cv_tag="bed_level",
        sp_trajectory=Trajectory(15.0, unit=Unit("m")),
        deadband=0.1,
        mv_range=(0 * Unit(), 1 * Unit()),
        alpha=0.4,
        inverted=True,
    ),
]
