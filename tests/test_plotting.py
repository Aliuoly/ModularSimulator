import pytest
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from modular_simulation.plotting import plot_triplet_series, triplets_from_history, triplets_to_arrays
from modular_simulation.usables import TimeValueQualityTriplet



def _sample_series() -> list[TimeValueQualityTriplet]:
    return [
        TimeValueQualityTriplet(t=0.0, value=1.0, ok=True),
        TimeValueQualityTriplet(t=1.0, value=2.0, ok=False),
        TimeValueQualityTriplet(t=2.0, value=3.0, ok=True),
    ]


def test_triplets_to_arrays_splits_fields():
    samples = _sample_series()
    times, values, ok = triplets_to_arrays(samples)

    assert np.allclose(times, [0.0, 1.0, 2.0])
    assert np.allclose(values, [1.0, 2.0, 3.0])
    assert np.array_equal(ok, [True, False, True])


def test_plot_triplet_series_marks_bad_values():
    fig, ax = plt.subplots()
    samples = _sample_series()

    plot_triplet_series(ax, samples, label="series")

    assert len(ax.lines) == 2  # one segment for each contiguous True run
    assert len(ax.collections) == 1
    scatter = ax.collections[0]
    offsets = scatter.get_offsets()
    assert offsets.shape[0] == 1
    np.testing.assert_allclose(offsets, [[1.0, 2.0]])


def test_triplets_from_history_builds_triplets():
    history = {
        "time": np.array([0.0, 1.0]),
        "value": np.array([5.0, 6.0]),
        "ok": np.array([True, False]),
    }

    triplets = triplets_from_history(history)

    assert triplets == [
        TimeValueQualityTriplet(t=0.0, value=5.0, ok=True),
        TimeValueQualityTriplet(t=1.0, value=6.0, ok=False),
    ]
