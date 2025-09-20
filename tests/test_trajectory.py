import pytest

pytest.importorskip("numpy")

from modular_simulation.control_system.trajectory import Trajectory


def test_set_now_prunes_segments_and_records_history():
    traj = Trajectory(y0=0.0, t0=0.0)
    times = [float(t) for t in range(5)]

    for t in times:
        traj.set_now(t, t)
        assert len(traj.segments) <= 2
        assert traj(t) == pytest.approx(t)

    history = traj.history()
    assert history["time"].tolist() == times
    assert history["value"].tolist() == times
