from __future__ import annotations

from astropy.units import Unit

from modular_simulation.interfaces import (
    BangBangController,
    CalculationModelPath,
    ControllerBase,
    FirstOrderTrajectoryController,
    InternalModelController,
    PIDController,
    Trajectory,
)

from .calculations.misc_calculations import CatInventoryEstimator, AlTiRatioEstimator
from .calculations.property_estimator import PropertyEstimator


def create_controllers() -> list[ControllerBase]:
    """Return the control hierarchy for the gas-phase reactor."""

    return [
        InternalModelController(
            mv_tag="F_cat",
            cv_tag="cat_inventory",
            sp_trajectory=Trajectory(y0=0.0, unit=Unit("kg")),
            mv_range=(0.0, 20.0),
            model=CalculationModelPath(
                calculation_name=CatInventoryEstimator,
                method_name="model",
            ),
            cascade_controller=PIDController(
                mv_tag="cat_inventory",
                cv_tag="filtered_mass_prod_rate",
                sp_trajectory=Trajectory(y0=0.0, unit=Unit("tonne/hour")).ramp(
                    50.0, ramprate=25.0 / 3600.0
                ),
                mv_range=(0.0, 80.0),
                Kp=0.5,
                Ti=3600.0,
                Td=0.0,
                setpoint_weight=0.0,
            ),
        ),
        PIDController(
            mv_tag="F_m1",
            cv_tag="pM1",
            sp_trajectory=Trajectory(y0=700.0, unit=Unit("kPa")),
            mv_range=(0.0, 70_000.0),
            Kp=50.0,
            Ti=3600.0,
            Td=0.0,
        ),
        PIDController(
            mv_tag="F_m2",
            cv_tag="rM2",
            sp_trajectory=Trajectory(y0=0.3, unit=Unit()),
            mv_range=(0.0, 7_000.0),
            Kp=3500.0,
            Ti=1800.0,
            Td=0.0,
            setpoint_weight=0.0,
            cascade_controller=InternalModelController(
                mv_tag="rM2",
                cv_tag="inst_density",
                sp_trajectory=Trajectory(y0=918.0, unit=Unit("kg/m^3")),
                mv_range=(0.0, 0.6),
                model=CalculationModelPath(
                    calculation_name=PropertyEstimator,
                    method_name="inst_density_model",
                ),
                cascade_controller=FirstOrderTrajectoryController(
                    mv_tag="inst_density",
                    cv_tag="cumm_density",
                    sp_trajectory=Trajectory(y0=918.0, unit=Unit("kg/m^3")),
                    mv_range=(905.0, 965.0),
                    closed_loop_time_constant_fraction=0.8,
                    open_loop_time_constant="residence_time",
                ),
            ),
        ),
        PIDController(
            mv_tag="F_h2",
            cv_tag="rH2",
            sp_trajectory=Trajectory(y0=0.0, unit=Unit()),
            mv_range=(0.0, 15.0),
            Kp=200.0,
            Ti=5400.0,
            Td=0.0,
            setpoint_weight=0.0,
            cascade_controller=InternalModelController(
                mv_tag="rH2",
                cv_tag="inst_MI",
                sp_trajectory=Trajectory(y0=2.0, unit=Unit()),
                mv_range=(0.0, 0.9),
                model=CalculationModelPath(
                    calculation_name=PropertyEstimator,
                    method_name="inst_MI_model",
                ),
                cascade_controller=FirstOrderTrajectoryController(
                    mv_tag="inst_MI",
                    cv_tag="cumm_MI",
                    sp_trajectory=Trajectory(y0=2.0, unit=Unit()),
                    mv_range=(0.2, 50.0),
                    closed_loop_time_constant_fraction=0.8,
                    open_loop_time_constant="residence_time",
                ),
            ),
        ),
        InternalModelController(
            mv_tag="F_teal",
            cv_tag="Al_Ti_ratio",
            sp_trajectory=Trajectory(y0=0.4, unit=Unit()),
            mv_range=(0.0, 1e5),
            model=CalculationModelPath(
                calculation_name=AlTiRatioEstimator,
                method_name="AlTi_model",
            ),
        ),
        PIDController(
            mv_tag="F_n2",
            cv_tag="filtered_pressure",
            sp_trajectory=Trajectory(y0=1900.0, unit=Unit("kPa")),
            mv_range=(0.0, 500.0),
            Kp=500.0 / 200.0,
            Ti=float("inf"),
            Td=0.0,
        ),
        PIDController(
            mv_tag="F_vent",
            cv_tag="filtered_pressure",
            sp_trajectory=Trajectory(y0=1900.0, unit=Unit("kPa")),
            mv_range=(0.0, 25_000.0),
            Kp=25_000.0 / 200.0,
            Ti=float("inf"),
            Td=0.0,
            inverted=True,
        ),
        BangBangController(
            mv_tag="discharge_valve_position",
            cv_tag="bed_level",
            sp_trajectory=Trajectory(y0=15.0, unit=Unit("m")),
            deadband=0.1,
            mv_range=(0.0, 1.0),
            alpha=0.4,
            inverted=True,
        ),
    ]


__all__ = ["create_controllers"]
