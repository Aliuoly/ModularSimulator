"""Utilities for plotting :class:`TimeValueQualityTriplet` series with Matplotlib."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Iterable, List, Tuple, TYPE_CHECKING

import numpy as np

from modular_simulation.usables import TimeValueQualityTriplet

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from matplotlib.axes import Axes
else:  # pragma: no cover - fall back when matplotlib is absent
    Axes = Any  # type: ignore

SeriesInput = Sequence[TimeValueQualityTriplet] | Mapping[str, Sequence[Any]]


def _coerce_scalar(value: Any) -> float:
    arr = np.asarray(value)
    if arr.shape != () and arr.shape != (1, ):
        raise ValueError(
            "Plotting only supports scalar values; received array with shape " 
            f"{arr.shape}."
        )
    return float(arr.item())


def triplets_to_arrays(samples: Sequence[TimeValueQualityTriplet]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a sequence of triplets into NumPy arrays of time, value, and quality flags."""

    times = np.empty(len(samples), dtype=float)
    values = np.empty(len(samples), dtype=float)
    ok = np.empty(len(samples), dtype=bool)

    for idx, sample in enumerate(samples):
        times[idx] = float(sample.t)
        values[idx] = _coerce_scalar(sample.value)
        ok[idx] = bool(sample.ok)
    return times, values, ok


def _extract_series(samples: SeriesInput) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(samples, Mapping):
        try:
            times_like = samples["time"]
            values_like = samples["value"]
            ok_like = samples["ok"]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError("History mapping must include 'time', 'value', and 'ok' entries.") from exc

        times_arr = np.asarray(times_like, dtype=float)
        ok_arr = np.asarray(ok_like, dtype=bool)
        if len(times_arr) != len(values_like) or len(times_arr) != len(ok_arr):
            raise ValueError("History arrays must share the same length.")

        values_arr = np.empty(len(times_arr), dtype=float)
        for idx, raw_value in enumerate(values_like):
            values_arr[idx] = _coerce_scalar(raw_value)
        return times_arr, values_arr, ok_arr

    return triplets_to_arrays(samples)


def _iter_true_segments(mask: np.ndarray) -> Iterable[slice]:
    start: int | None = None
    for idx, flag in enumerate(mask):
        if flag:
            if start is None:
                start = idx
        elif start is not None:
            yield slice(start, idx)
            start = None
    if start is not None:
        yield slice(start, mask.shape[0])


def plot_triplet_series(
    ax: Axes,
    samples: SeriesInput,
    *,
    label: str | None = None,
    style: str = "line",
    line_kwargs: Mapping[str, Any] | None = None,
    bad_kwargs: Mapping[str, Any] | None = None,
) -> List[Any]:
    """Plot a :class:`TimeValueQualityTriplet` series on ``ax``.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    samples:
        Either a sequence of triplets or a mapping containing ``time``, ``value`` and ``ok`` arrays.
    label:
        Legend label applied to the first continuous segment.
    style:
        ``"line"`` for ``ax.plot`` or ``"step"`` for ``ax.step``.
    step_where:
        Passed to :meth:`matplotlib.axes.Axes.step` when ``style`` is ``"step"``.
    connect_gaps:
        When ``True`` missing samples (``ok == False``) are omitted but the surrounding points
        remain connected. When ``False`` the line is broken at each missing sample.
    line_kwargs:
        Additional keyword arguments forwarded to the plotting call.
    bad_kwargs:
        Keyword arguments for the scatter plot highlighting bad samples. Defaults to red ``"x"`` markers.

    Returns
    -------
    list of Matplotlib artists added to the axes.
    """

    times, values, ok = _extract_series(samples)
    artists: List[Any] = []

    if times.size == 0:
        return artists

    plot_func = ax.plot
    if style == "step":
        plot_func = ax.step
    elif style != "line":
        raise ValueError(f"Unsupported style '{style}'. Expected 'line' or 'step'.")

    line_opts = dict(line_kwargs or {})
    line_opts.update({"label": label})
    plot_func(times, values, **line_opts)

    bad_indices = np.flatnonzero(~ok.astype(bool))
    if bad_indices.size:
        bad_opts = {"marker": "x", "color": "black", "label": None, "zorder": 3}
        if bad_kwargs:
            bad_opts.update(bad_kwargs)
        scatter = ax.scatter(times[bad_indices], values[bad_indices], **bad_opts)
        artists.append(scatter)

    return artists


def triplets_from_history(history_entry: Mapping[str, Sequence[Any]]) -> List[TimeValueQualityTriplet]:
    """Convert a history mapping (time/value/ok) to triplet objects."""

    times, values, ok = _extract_series(history_entry)
    return [
        TimeValueQualityTriplet(t=float(t), value=float(v), ok=bool(flag))
        for t, v, flag in zip(times, values, ok, strict=True)
    ]


__all__ = ["plot_triplet_series", "triplets_from_history", "triplets_to_arrays"]
