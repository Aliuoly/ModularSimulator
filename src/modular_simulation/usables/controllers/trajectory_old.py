from collections.abc import Callable
import bisect
import math
import numpy as np
from astropy.units import UnitBase, Unit #type: ignore
from dataclasses import dataclass
from pydantic import BaseModel, Field, PrivateAttr, field_serializer, field_validator, ConfigDict
from dataclasses import replace

Number = int | float

# ------------------------------
# Utilities
# ------------------------------

def _clamp(x: float, lo: float | None = None, hi: float | None = None) -> float:
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x

# Hash-based, grid-sampled noise so evaluation is O(1) without state.
# We use a seeded LCG keyed by integer time bins and optionally interpolate.

def _noise_unit(seed: int, k: int) -> float:
    # simple 32-bit LCG for reproducibility
    a = 1664525
    c = 1013904223
    m = 2 ** 32
    x = (seed + k) & 0xFFFFFFFF
    x = (a * x + c) % m
    # map to (-1, 1)
    return (float(x) / (m - 1)) * 2.0 - 1.0


def _noise_interp(seed: int, x: float) -> float:
    # x is in bins; we linear-interp between integer bins for continuity
    k0 = math.floor(x)
    k1 = k0 + 1
    r0 = _noise_unit(seed, k0)
    r1 = _noise_unit(seed, k1)
    frac = x - k0
    return (1 - frac) * r0 + frac * r1


# ------------------------------
# Segment definitions
# ------------------------------

@dataclass(frozen=True)
class Segment:
    """Base segment describing a portion of a setpoint trajectory timeline.

    Subclasses implement :meth:`eval` to compute the setpoint value at time
    ``t`` given the setpoint at the beginning of the segment.
    """
    t0: float  # start time of segment (absolute)
    duration: float  # duration of the segment; float('inf') for open-ended

    def eval(self, t: float, y0: float) -> float:
        raise NotImplementedError

    def t1(self) -> float:
        return self.t0 + self.duration


@dataclass(frozen=True)
class Hold(Segment):
    """Trajectory segment that maintains a constant output value."""
    value: float

    def eval(self, t: float, y0: float) -> float:
        return self.value


@dataclass(frozen=True)
class Step(Segment):
    """Trajectory segment that applies an instantaneous offset to ``y0``."""
    magnitude: float

    def eval(self, t: float, y0: float) -> float:
        return y0 + self.magnitude


@dataclass(frozen=True)
class Ramp(Segment):
    """Trajectory segment that linearly interpolates to a new setpoint."""
    magnitude: float  # total change over duration (can be +/-)

    def eval(self, t: float, y0: float) -> float:
        # linear interpolation from y0 to y0 + magnitude over [t0, t1]
        if self.duration <= 0:
            return y0 + self.magnitude
        frac = (t - self.t0) / self.duration
        frac = 0.0 if frac < 0 else (1.0 if frac > 1.0 else frac)
        return y0 + frac * self.magnitude


@dataclass(frozen=True)
class RandomWalk(Segment):
    """Trajectory segment that emulates a bounded random walk.

    The segment samples a deterministic pseudo-random Field to produce
    repeatable noise without maintaining state between evaluations.  The
    resulting value can be clamped to an optional min/max.
    """
    std: float = 0.1
    dt: float = 1.0  # grid period for the underlying noise Field
    clamp_min: float | None = None
    clamp_max: float | None = None
    seed: int = 0

    def eval(self, t: float, y0: float) -> float:
        # cumulative sum of zero-mean increments; here we emulate a RW
        # by integrating a continuous noise Field over [t0, t]
        if t <= self.t0:
            return y0
        # integrate via trapezoid over one step for O(1) closed form approx:
        # y(t) = y0 + std * sqrt(dt) * W where W is scaled noise integral.
        # We approximate the integral as (n steps)*mean(noise) * dt.
        # For responsiveness, just use the noise value at (t - t0)/dt.
        x = (t - self.t0) / self.dt
        inc = _noise_interp(self.seed, x)
        y = y0 + self.std * math.sqrt(self.dt) * inc
        return _clamp(y, self.clamp_min, self.clamp_max)


# ------------------------------
# Trajectory class
# ------------------------------

class Trajectory(BaseModel):
    """Composable setpoint profile made of piecewise time segments.

    Trajectories expose fluent builders (``hold``/``step``/``ramp``/``random_walk``)
    and can be modified on-line via :meth:`set_now`.  History arrays are
    maintained alongside cached breakpoints and starting values to keep
    evaluation amortized :math:`O(1)` for monotonic time progression.
    """
    y0: float
    t0: float = 0.0
    segments: list[Segment] = Field(default_factory=list)

    # fast-path caches
    _breaks: list[float] = PrivateAttr(default_factory=list)
    _cursor: int = PrivateAttr(default=0)
    _last_value: float | None = PrivateAttr(default=None)
    _start_vals: list[float] = PrivateAttr(default_factory=list)  # y at each segment start (O(1) eval)

    # historization and bookkeeping for fast set_now
    _history_times: list[float] = PrivateAttr(default_factory=list)
    _history_values: list[Number] = PrivateAttr(default_factory=list)
    _last_set_time: float | None = PrivateAttr(default=None)
    _last_set_value: Number | None = PrivateAttr(default=None)
    _max_retained_segments: int = PrivateAttr(default=1)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # --------------------------
    # Builder / mutators (chainable)
    # --------------------------
    @field_validator("unit", mode = 'before')
    @classmethod
    def convert_unit(cls, unit: str|UnitBase) -> UnitBase:
        if isinstance(unit, str):
            return Unit(unit)
        return unit
    
    def clone(self) -> "Trajectory":
        out = Trajectory(y0 = self.y0, unit = self.unit, t0 = self.t0, segments = list(self.segments))
        out._breaks = list(self._breaks)
        out._start_vals = list(self._start_vals)
        out._history_times = list(self._history_times)
        out._history_values = list(self._history_values)
        out._last_set_time = self._last_set_time
        out._last_set_value = self._last_set_value
        return out

    # --------------------------
    # Historization helpers
    # --------------------------

    def _record_history_entry(self, t: float, value: Number) -> None:
        """Append or update a historized setpoint sample at time ``t``."""
        if self._history_times:
            if t < self._history_times[-1] - 1e-12:
                # keep history sorted even if times go slightly backwards
                idx = bisect.bisect_left(self._history_times, t)
                self._history_times.insert(idx, t)
                self._history_values.insert(idx, value)
                return
            if abs(t - self._history_times[-1]) <= 1e-12:
                self._history_values[-1] = value
                return
        self._history_times.append(t)
        self._history_values.append(value)

    def _history_value_at(self, t: float) -> Number:
        if not self._history_times:
            return self.y0
        idx = bisect.bisect_right(self._history_times, t) - 1
        if idx < 0:
            return self.y0
        return self._history_values[idx]

    def history(self) -> dict[str, np.ndarray]:
        """Return historized setpoint samples as NumPy arrays."""
        return {
            "time": np.asarray(self._history_times, dtype=float),
            "value": np.asarray(self._history_values, dtype=float),
        }
    def writer(self, time_getter: Callable[[], float]) -> Callable[[Number], None]:
        """Return a callable that writes setpoints at the time provided by ``time_getter``."""

        def _write(value: Number) -> None:
            t = time_getter()
            if t is None:
                raise RuntimeError("Cannot write to trajectory without a valid time reference.")
            self.set_now(float(t), float(value))

        return _write
    def last_setpoint(self) -> Number:
        if self._last_set_value is not None:
            return self._last_set_value
        if self._history_values:
            return self._history_values[-1]
        if self.segments:
            idx = len(self.segments) - 1
            seg = self.segments[idx]
            return seg.eval(seg.t0, self._start_vals[idx])
        return self.y0

    def current_value(self, t: float) -> float:
        if self._last_set_time is not None and t >= self._last_set_time - 1e-12:
            if self._last_set_value is not None:
                return float(self._last_set_value)
        return float(self.eval(t))

    def _can_append_at_end(self, t: float) -> bool:
        end_t = self._end_time()
        if t < end_t - 1e-12:
            return False
        return True

    def hold(self, duration: float, value: float | None = None) -> "Trajectory":
        if value is None:
            value = self.peek_end_value()
        t = self._end_time()
        self._append(Hold(t, duration, value))
        return self

    def step(self, magnitude: float) -> "Trajectory":
        t = self._end_time()
        self._append(Step(t, 0.0, magnitude))
        return self

    def ramp(self, magnitude: float | None = None, *, duration: float | None = None, ramprate: float | None = None) -> "Trajectory":
        if magnitude is None and duration is None and ramprate is None:
            raise ValueError("Provide magnitude+duration, or magnitude+ramprate, or duration+ramprate")
        if magnitude is None:
            magnitude = ramprate * duration  # type: ignore
        if duration is None:
            duration = abs(magnitude / ramprate)  # type: ignore
        t = self._end_time()
        self._append(Ramp(t, duration, magnitude))
        return self

    def random_walk(self, std: float = 0.1, *, duration: float = 1.0, dt: float = 1.0, min: float | None = None, max: float | None = None, seed: int = 0) -> "Trajectory":
        t = self._end_time()
        self._append(RandomWalk(t, duration, std, dt, min, max, seed))
        return self

    # O(1) inner-loop retarget: no deletion, just append a Hold at t
    def set_now(self, t: float, value: float) -> "Trajectory":
        """Append a hold segment starting at ``t`` with the provided value.

        When ``t`` is at or beyond the current end time the operation is O(1).
        Otherwise the future of the trajectory is trimmed before appending the
        new hold segment, which is O(n) in the number of retained segments.
        """
        if not self._can_append_at_end(t):
            self.trim_future(t)

        if self.segments:
            last_seg = self.segments[-1]
            prev_start = last_seg.t0
            prev_value = last_seg.eval(min(t, last_seg.t1()), self._start_vals[-1])
            # remove segments of 0 duration at the end where we are about to append to 

        else:
            prev_start = self.t0
            prev_value = self.y0

        self._record_history_entry(prev_start, prev_value)

        if len(self.segments) > self._max_retained_segments:
            drop = len(self.segments) - self._max_retained_segments
            del self.segments[:drop]
            del self._start_vals[:drop]
            self._rebuild_breaks()
        
        
        self._append(Hold(t0 = t, duration = 0, value = value), start_value=prev_value)
        self._record_history_entry(t, value)
        self._last_set_time = t
        self._last_set_value = value
        return self
    


    def trim_future(self, t: float) -> "Trajectory":
        """Drop segments starting after time ``t`` and clip any active segment."""
        if not self.segments:
            return self
        new_segments: list[Segment] = []
        for seg in self.segments:
            if seg.t0 < t < seg.t1():
                # clip ongoing segment
                new_dur = t - seg.t0
                seg = replace(seg, duration=new_dur)  # type: ignore
                new_segments.append(seg)
            elif seg.t1() <= t:
                new_segments.append(seg)
        self.segments = new_segments
        self._rebuild_breaks()
        return self

    # --------------------------
    # Evaluation
    # --------------------------

    def __call__(self, t: float) -> float:
        return self.eval(t)

    def eval(self, t: float) -> float:
        if not self.segments:
            return self.y0
        # fast path: monotone time forward
        if self._last_value is not None and self._cursor < len(self.segments):
            seg = self.segments[self._cursor]
            if seg.t0 <= t < seg.t1():
                y = seg.eval(t, self._start_vals[self._cursor])
                self._last_value = y
                return y
            if t >= seg.t1():
                while self._cursor + 1 < len(self.segments) and t >= self.segments[self._cursor].t1():
                    self._cursor += 1
                seg = self.segments[min(self._cursor, len(self.segments)-1)]
                y = seg.eval(t, self._start_vals[self._cursor])
                self._last_value = y
                return y
        if self._breaks and t < self._breaks[0]:
            y_hist = float(self._history_value_at(t))
            self._last_value = y_hist
            self._cursor = 0
            return y_hist
        # general path: binary search
        idx = bisect.bisect_right(self._breaks, t) - 1
        idx = max(0, min(idx, len(self.segments)-1))
        y = self.segments[idx].eval(t, self._start_vals[idx])
        self._cursor = idx
        self._last_value = y
        return y

    # --------------------------
    # Introspection & internals
    # --------------------------

    def _end_time(self) -> float:
        return self.t0 if not self.segments else self.segments[-1].t1()

    def peek_end_value(self) -> float:
        if not self.segments:
            return self.y0
        idx = len(self.segments) - 1
        t = math.nextafter(self.segments[-1].t1(), -math.inf)
        return self.segments[idx].eval(t, self._start_vals[idx])

    def _append(self, seg: Segment, *, start_value: float | None = None) -> None:
        # append must be at the current end time to keep cached starts valid
        if not self._can_append_at_end(seg.t0):
            self.trim_future(seg.t0)
        # compute start value using previous tail
        if start_value is None:
            if self.segments:
                prev_idx = len(self.segments) - 1
                y_prev_end = self.segments[prev_idx].eval(self.segments[prev_idx].t1(), self._start_vals[prev_idx])
            else:
                y_prev_end = self.y0
        else:
            y_prev_end = start_value
        self.segments.append(seg)
        self._start_vals.append(y_prev_end)
        if self._breaks:
            if seg.t0 < self._breaks[-1] - 1e-12:
                self._rebuild_breaks()
            else:
                self._breaks.append(seg.t1())
        else:
            self._breaks = [seg.t0, seg.t1()]
        self._cursor = len(self.segments) - 1
        self._last_value = None

    def _rebuild_breaks(self, recompute_starts: bool = False) -> None:
        self._breaks = [s.t0 for s in self.segments] + ([self.segments[-1].t1()] if self.segments else [self.t0])
        self._cursor = 0
        self._last_value = None
        if recompute_starts:
            self._start_vals = []
            y = self.y0
            for s in self.segments:
                self._start_vals.append(y)
                y = s.eval(s.t1(), y)
    @field_serializer("unit", mode="plain")
    def _serialize_unit(self, unit: UnitBase) -> str:
        return str(unit)
    @field_serializer("segments", mode="plain")
    def _serialize_segments(self, segments: list[Segment]) -> list:
        # subclass of segments all get interpreted as base 'Segment',
        # which will throw unimplemented error if used as is. 
        # therefore, serialization must lose info on future
        # segments:
        return []
    @field_serializer("y0", mode="plain")
    def _serialize_y0(self, y0: float) -> float:
        # y0 is to take on the last applicable setpoint
        # of the existing trajectory. 
        # unless ofcourse no last value exists yet
        if self._last_value is None:
            return self.y0
        return self._last_value
    @field_serializer("t0", mode="plain")
    def _serialize_t0(self, t0:float) -> float:
        # t0 will be forced to 0 in case it wasnt before
        # which, I don't think is allowed... not that I've tried. 
        return 0.0

# ------------------------------
# Multi-channel convenience wrapper
# ------------------------------

@dataclass
class MultiTrajectory:
    tracks: list[Trajectory]

    def __call__(self, t: float) -> np.ndarray:
        return np.array([tr(t) for tr in self.tracks])

    def track(self, i: int) -> Trajectory:
        return self.tracks[i]


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    # Build a trajectory
    T = (Trajectory(y0=0.0, unit = "m3", t0=0.0)
         .hold(1.0, value=0.0)
         .step(5.0)
         .ramp(magnitude=5.0, ramprate=1.0)  # duration = 5s
         .random_walk(std=0.1, duration=10.0, dt=0.5, min=0.0, max=12.0, seed=42)
         .hold(3.0))

    ts = np.linspace(0, 25, 101, dtype = np.float64)
    ys = np.array([T(t) for t in ts])
    print(f"y(0..25): mean={ys.mean():.3f}, min={ys.min():.3f}, max={ys.max():.3f}")

    # Cascade update: outer loop updates inner-loop SP every 0.1s
    inner = Trajectory(y0=ys[0], unit = "m2", t0=0.0)
    for k, t in enumerate(ts):
        sp = T(t)
        inner.set_now(t, sp)  # O(1) append + clear future
        _ = inner(t)  # controller pulls inner(t)

    # Vector example (e.g., pressure & temperature)
    M = MultiTrajectory([T.clone(), Trajectory(y0=2.0, unit = "m3").ramp(magnitude=-1.0, duration=10.0)])
    yv = M(3.3)
    print("vector at t=3.3:", yv)
