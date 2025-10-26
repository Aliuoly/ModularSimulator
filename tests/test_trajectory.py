import numpy as np
import pytest
from astropy.units import Quantity, Unit

from modular_simulation.interfaces import Trajectory


def test_set_now_prunes_segments_and_records_history():
    traj = Trajectory(y0=Quantity(0.0, Unit("K")))
    times = [float(t) for t in range(5)]

    for t in times:
        traj.set_now(t, Quantity(t, Unit("K")))
        assert len(traj.segments) <= 2
        assert traj(t).to_value(Unit("K")) == pytest.approx(t)

    history = traj.history()
    assert history["time"].tolist() == times
    assert history["value"].tolist() == times


def test_hold_and_ramp_segments():
    traj = Trajectory(y0=Quantity(10.0, Unit("K")))
    traj.hold(2.0, value=Quantity(10.0, Unit("K"))).ramp(magnitude=Quantity(10.0, Unit("K")), duration=2.0)
    traj.step(Quantity(-2.0, Unit("K")))

    assert traj(1.0).to_value(Unit("K")) == pytest.approx(10.0)
    assert traj(2.0).to_value(Unit("K")) == pytest.approx(10.0)
    assert traj(3.0).to_value(Unit("K")) == pytest.approx(15.0)
    assert traj(4.0).to_value(Unit("K")) == pytest.approx(18.0)


def test_random_walk_is_deterministic_with_seed():
    traj = Trajectory(y0=Quantity(0.0, Unit("K")))
    traj.random_walk(std=Quantity(0.5, Unit("K")), duration=5.0, dt=1.0, seed=42)

    values = np.array([traj(t).to_value(Unit("K")) for t in [0.0, 1.0, 2.5, 5.0]])

    traj2 = Trajectory(y0=Quantity(0.0, Unit("K")))
    traj2.random_walk(std=Quantity(0.5, Unit("K")), duration=5.0, dt=1.0, seed=42)
    values_2 = np.array([traj2(t).to_value(Unit("K")) for t in [0.0, 1.0, 2.5, 5.0]])

    assert values == pytest.approx(values_2)


def test_writer_updates_using_time_callback():
    current_time = 0.0

    def time_getter():
        return current_time

    traj = Trajectory(y0=Quantity(1.0, Unit("K")))
    writer = traj.writer(time_getter)

    current_time = 1.0
    writer(Quantity(5.0, Unit("K")))
    current_time = 2.0
    writer(Quantity(6.0, Unit("K")))

    history = traj.history()
    assert history["time"].tolist()[-2:] == [1.0, 2.0]
    assert history["value"].tolist()[-2:] == [5.0, 6.0]
    assert traj(2.0).to_value(Unit("K")) == pytest.approx(6.0)
