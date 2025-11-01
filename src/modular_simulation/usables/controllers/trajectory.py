from __future__ import annotations
import math
from bisect import bisect_right
from pydantic.dataclasses import dataclass
from pydantic import PlainSerializer, BeforeValidator
from dataclasses import field
from typing import overload, Annotated
from modular_simulation.utils.typing import TimeValue, StateValue, PerTimeValue
from modular_simulation.utils.wrappers import second, second_value

# ---- Stateless hashy noise for deterministic RW ----
def _noise_unit(seed: int, k: int) -> float:
    a, c, m = 1664525, 1013904223, 2**32
    x = (seed + k) & 0xFFFFFFFF
    x = (a * x + c) % m
    return (float(x) / (m - 1)) * 2.0 - 1.0  # (-1, 1)


def _noise_interp(seed: int, x: float) -> float:
    k0 = math.floor(x)
    r0 = _noise_unit(seed, k0)
    r1 = _noise_unit(seed, k0 + 1)
    f = x - k0
    return (1 - f) * r0 + f * r1


@dataclass(config={"arbitrary_types_allowed": True})
class Trajectory:
    """
    Minimal piecewise setpoint trajectory with trimming.

    - Values: unitless floats (caller handles value units if desired).
    - Time: astropy Quantity for all public time/duration parameters.
    - Internals: seconds (float) in two primitive lists: _knots_t, _knots_y.
    - Ops: set, step, ramp, hold, random_walk
    - Trimming: an internal clock t0; all ops call _gc() to discard history.

    Semantics:
      - set(value, t=None): from time t onward hold 'value'. If t is None, uses current end time.
      - step(mag): instantaneous jump at current end time.
      - ramp(mag, duration): linear change over 'duration' starting at end time.
      - hold(duration, value=None): keep value (or set to 'value' first) and append a flat segment.
      - random_walk(std, duration, dt, clamp=(min,max), seed): cumulative RW knots, deterministic.
    """
    y0: float
    t0: Annotated[TimeValue, BeforeValidator(second), PlainSerializer(second_value),]

    _knots_t: list[float] = field(init=False)
    _knots_y: list[float] = field(init=False)

    def __post_init__(self):
        self._knots_t = [self.t0] # already in seconds due to validation in SerializableTimeValue
        self._knots_y = [float(self.y0)]

    # --------------------------
    # Public API
    # --------------------------

    def __call__(self, t: TimeValue) -> float:
        """Evaluate trajectory at time t (Quantity)."""
        ts = second(t)
        ti, yi = self._knots_t, self._knots_y

        # fast bounds
        j = bisect_right(ti, ts)
        if j == 0:
            return yi[0]
        if j == len(ti):
            return yi[-1]

        # linear interp (right-hand at steps because of bisect_right)
        tL, tR = ti[j - 1], ti[j]
        yL, yR = yi[j - 1], yi[j]
        if tR == tL:
            return yR
        f = (ts - tL) / (tR - tL)
        return yL + f * (yR - yL)

    def set(self, value: float, t: TimeValue|None = None) -> Trajectory:
        """
        Redefine future from time t to a constant 'value'.
        If t is None, use the current end time.
        """
        ts = self.end_time_s() if t is None else second(t)
        self._clip_at(ts)
        self._append_knot(ts, float(value))
        self._gc()
        return self

    def step(self, magnitude: float) -> Trajectory:
        """Instantaneous jump at current end time by 'magnitude'."""
        t_end, y_end = self.end()
        self._append_knot(t_end, y_end + float(magnitude))
        self._gc()
        return self

    @overload
    def ramp(self, magnitude: float, *, duration: TimeValue, rate: None = None) -> Trajectory: ...
    @overload
    def ramp(self, magnitude: float, *, duration: None = None, rate: "PerTimeValue") -> Trajectory: ...

    def ramp(
        self,
        magnitude: float,
        *,
        duration: TimeValue | None = None,
        rate: PerTimeValue | None = None
    ) -> Trajectory:
        """
        Create a trajectory segment that linearly interpolates to a new setpoint.

        Exactly one of ``duration`` or ``rate`` must be provided.

        If ``duration`` is specified, the segment linearly changes by ``magnitude``
        over the given duration. If ``rate`` is specified instead, the duration
        is computed as ``abs(magnitude / rate)``. Negative magnitudes produce
        downward ramps, while negative rates are treated as positive speeds with
        direction implied by ``magnitude``.

        Parameters
        ----------
        magnitude : float
            Total change in the setpoint value over the ramp.
        duration : TimeValue, optional
            Duration of the ramp segment (e.g., ``10 * u.s``). Mutually exclusive
            with ``rate``.
        rate : PerTimeValue, optional
            Ramp rate in units of 1/time (e.g., ``0.2 / u.s``). Mutually exclusive
            with ``duration``.

        Returns
        -------
        Trajectory
            The updated trajectory with the new ramp segment appended.
        """
        if duration is not None:
            dur_s = second(duration)
            if dur_s < 0:
                raise ValueError("duration must be non-negative")
        elif rate is not None:
            dur_s = abs(float(magnitude) / rate.to("1/second").value)
        else:
            raise ValueError("Provide exactly one of 'duration' or 'rate'.")

        t0, y0 = self.end()
        if dur_s == 0.0:
            self._append_knot(t0, y0 + float(magnitude))
        else:
            self._append_knot(t0 + dur_s, y0 + float(magnitude))

        self._gc()
        return self

    def hold(self, duration: TimeValue, value: float | None = None) -> Trajectory:
        """
        Keep the (optionally provided) value for 'duration' time.
        Equivalent to: (set(value) if provided). then append flat for duration.
        """
        if value is not None:
            # Set at end time without changing time
            self.set(value, None)
        # Just extend in time by duration, value stays at last knot
        dur_s = second(duration)
        if dur_s < 0:
            raise ValueError("duration must be non-negative")
        t_end, y_end = self.end()
        if dur_s == 0.0:
            # no-op in time, but explicit flat knot keeps semantics clear
            self._append_knot(t_end, y_end)
        else:
            self._append_knot(t_end + dur_s, y_end)
        self._gc()
        return self

    def random_walk(
        self,
        std: float = 0.1,
        *,
        duration: TimeValue,
        dt: TimeValue = second(1.0),
        clamp: tuple[float | None, float | None] = (None, None),
        seed: int = 0,
    ) -> Trajectory:
        """
        Append a bounded random-walk segment.

        - Deterministic, state-free noise via hash-based LCG + linear interp.
        - Increments ~ std * sqrt(dt) * noise_interp(...)
        - Adds knots every 'dt' up to 'duration'.
        """
        D = second(duration)
        h = second(dt)
        if D < 0 or h <= 0:
            raise ValueError("duration must be >= 0 and dt must be > 0")

        t, y = self.end()
        steps = max(1, int(math.ceil(D / h)))
        lo, hi = clamp

        for k in range(1, steps + 1):
            x = k  # “bin” index
            inc = std * math.sqrt(h) * _noise_interp(seed, x)
            y = y + inc
            if lo is not None and y < lo:
                y = lo
            if hi is not None and y > hi:
                y = hi
            self._append_knot(t + min(k * h, D), y)

        self._gc()
        return self

    def clone(self) -> Trajectory:
        t = Trajectory(self._knots_y[0], second(self._knots_t[0]))
        t._knots_t = self._knots_t.copy()
        t._knots_y = self._knots_y.copy()
        t.t0 = self.t0
        return t

    # --------------------------
    # Trimming / clock
    # --------------------------

    def advance_clock(self, t: TimeValue) -> None:
        """
        Move internal trimming time forward to t, and drop history.
        """
        if t > self.t0:
            self.t0 = t
            self._gc()

    # --------------------------
    # Introspection
    # --------------------------

    def start(self) -> tuple[float, float]:
        return self._knots_t[0], self._knots_y[0]

    def end(self) -> tuple[float, float]:
        return self._knots_t[-1], self._knots_y[-1]

    def end_time_s(self) -> float:
        return self._knots_t[-1]

    def times_seconds(self) -> list[float]:
        return self._knots_t.copy()

    def values(self) -> list[float]:
        return self._knots_y.copy()

    # --------------------------
    # Internals
    # --------------------------

    def _append_knot(self, t_s: float, y: float) -> None:
        if t_s < self._knots_t[-1]:
            raise RuntimeError("Internal: non-monotonic append; call _clip_at first.")
        self._knots_t.append(t_s)
        self._knots_y.append(float(y))

    def _clip_at(self, t_s: float) -> None:
        """
        Ensure a knot at t_s (interpolated if inside a segment),
        then drop future knots past t_s.
        """
        ti, yi = self._knots_t, self._knots_y
        j = bisect_right(ti, t_s)

        if j == 0:
            # Insert new front knot at t_s with y(t_s) (which equals first y)
            y = self.__call__(second(t_s))
            self._knots_t = [t_s]
            self._knots_y = [float(y)]
            return

        if j == len(ti):
            if t_s > ti[-1]:
                # extend with a hold up to t_s, then keep that knot
                self._append_knot(t_s, yi[-1])
            # else equal to last—nothing to do
            return

        tL, tR = ti[j - 1], ti[j]
        if t_s == tR:
            self._knots_t = ti[: j + 1]
            self._knots_y = yi[: j + 1]
            return

        # between knots: insert interpolated knot and truncate
        y_clip = self.__call__(second(t_s))
        self._knots_t = ti[:j] + [t_s]
        self._knots_y = yi[:j] + [float(y_clip)]

    def _gc(self) -> None:
        """
        Trim all knots earlier than t0. Keep trajectory valid by
        re-anchoring at (t0, y(t0)) as the first knot.
        """
        tmin = self.t0
        ti, yi = self._knots_t, self._knots_y

        if tmin <= ti[0]:
            return  # nothing to drop

        j = bisect_right(ti, tmin)
        # Evaluate value at tmin for a clean new start
        y0 = self.__call__(second(tmin))
        # Keep all knots with t >= tmin, but ensure first knot is exactly at tmin
        tail_t = [t for t in ti[j:] if t >= tmin]
        tail_y = yi[j:]

        self._knots_t = [tmin] + tail_t
        self._knots_y = [float(y0)] + tail_y
