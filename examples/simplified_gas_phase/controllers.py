from astropy.units import Unit #type:ignore
from modular_simulation.usables import (
    PIDController,
    Trajectory,
    ControllerMode,
    MVController,
)
controllers = [
    PIDController(
        mv_tag = "F_m1",
        cv_tag = "pM1",
        sp_trajectory = Trajectory(y0 = 700.0, unit = Unit("kPa")),
        Kp = 1000/2000*28,
        Ti = 5*60., # 12 minutes
        Td = 0.0,
        mv_range = (0, 80000.),
        setpoint_weight = 0.0,
        velocity_form=True
    ),
    PIDController(
        mv_tag = "F_m2",
        cv_tag = "rM2",
        sp_trajectory = Trajectory(y0 = 0.3, unit = Unit("")),
        Kp = 300 * 56,
        Ti = 30*60., # 30 minutes
        Td = 0.0,
        mv_range = (0, 8000.),
        setpoint_weight = 0.0,
        velocity_form=True
    ),
    MVController(
        mv_tag = "F_cat",
        cv_tag = "F_cat",
        sp_trajectory = Trajectory(y0=6.0, unit = Unit("kg/hr")),
        mv_range = (0, 20.),
        mode = ControllerMode.AUTO,
        cascade_controller = PIDController(
            mv_tag = "F_cat",
            cv_tag = "filtered_mass_prod_rate",
            sp_trajectory = Trajectory(y0=50.0, unit = Unit("tonne/hr")),
            Kp = 1,
            Ti = 60*60., # 30 minutes
            Td = 0.0,
            mv_range = (0, 20.),
            mode = ControllerMode.TRACKING,
        )
    )
]