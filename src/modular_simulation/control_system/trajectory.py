from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import List, Optional, Union
import bisect
import math
import numpy as np

Number = Union[int, float]

# ------------------------------
# Utilities
# ------------------------------

def _clamp(x: float, lo: Optional[float], hi: Optional[float]) -> float:
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
    t0: float  # start time of segment (absolute)
    duration: float  # duration of the segment; np.inf for open-ended

    def eval(self, t: float, y0: float) -> float:
        raise NotImplementedError

    def t1(self) -> float:
        return self.t0 + self.duration


@dataclass(frozen=True)
class Hold(Segment):
    value: float

    def eval(self, t: float, y0: float) -> float:
        return self.value


@dataclass(frozen=True)
class Step(Segment):
    magnitude: float

    def eval(self, t: float, y0: float) -> float:
        return y0 + self.magnitude


@dataclass(frozen=True)
class Ramp(Segment):
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
    std: float = 0.1
    dt: float = 1.0  # grid period for the underlying noise field
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    seed: int = 0

    def eval(self, t: float, y0: float) -> float:
        # cumulative sum of zero-mean increments; here we emulate a RW
        # by integrating a continuous noise field over [t0, t]
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

@dataclass
class Trajectory:
    y0: float = 0.0
    t0: float = 0.0
    segments: List[Segment] = field(default_factory=list)

    # fast-path caches
    _breaks: List[float] = field(default_factory=list, init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)
    _last_value: float|None = field(default=None, init=False, repr=False)
    _start_vals: List[float] = field(default_factory=list, init=False, repr=False)  # y at each segment start (O(1) eval)

    # --------------------------
    # Builder / mutators (chainable)
    # --------------------------

    def clone(self) -> "Trajectory":
        out = Trajectory(self.y0, self.t0, list(self.segments))
        out._breaks = list(self._breaks)
        out._start_vals = list(self._start_vals)
        return out

    def _can_append_at_end(self, t: float) -> bool:
        end_t = self._end_time()
        if t < end_t - 1e-12:
            return False
        return True

    def hold(self, duration: float, value: Optional[float] = None) -> "Trajectory":
        if value is None:
            value = self.peek_end_value()
        t = self._end_time()
        self._append(Hold(t, duration, value))
        return self

    def step(self, magnitude: float) -> "Trajectory":
        t = self._end_time()
        self._append(Step(t, 0.0, magnitude))
        return self

    def ramp(self, magnitude: Optional[float] = None, *, duration: Optional[float] = None, ramprate: Optional[float] = None) -> "Trajectory":
        if magnitude is None and duration is None and ramprate is None:
            raise ValueError("Provide magnitude+duration, or magnitude+ramprate, or duration+ramprate")
        if magnitude is None:
            magnitude = ramprate * duration  # type: ignore
        if duration is None:
            duration = abs(magnitude / ramprate)  # type: ignore
        t = self._end_time()
        self._append(Ramp(t, duration, magnitude))
        return self

    def random_walk(self, std: float = 0.1, *, duration: float = 1.0, dt: float = 1.0, min: Optional[float] = None, max: Optional[float] = None, seed: int = 0) -> "Trajectory":
        t = self._end_time()
        self._append(RandomWalk(t, duration, std, dt, min, max, seed))
        return self

    # O(1) inner-loop retarget: no deletion, just append a Hold at t
    def set_now(self, t: float, value: float) -> "Trajectory":
        """Hard-set SP from time t onward. This is O(1): we append a Hold at t.
        Requires t >= current end time to keep evaluation O(1).
        falls back to trim_future O(n)"""
        if self._can_append_at_end(t):
            self._append(Hold(t, float('inf'), value))
        else:
            self.trim_future(t)
            self._append(Hold(t, float('inf'), value))
        return self
    


    def trim_future(self, t: float) -> "Trajectory":
        """Drop segments starting after time t; clip the active one."""
        if not self.segments:
            return self
        new_segments: List[Segment] = []
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

    def _append(self, seg: Segment) -> None:
        # append must be at the current end time to keep cached starts valid
        if not self._can_append_at_end(seg.t0):
            self.trim_future(seg.t0)
        # compute start value using previous tail
        if self.segments:
            prev_idx = len(self.segments) - 1
            y_prev_end = self.segments[prev_idx].eval(self.segments[prev_idx].t1(), self._start_vals[prev_idx])
        else:
            y_prev_end = self.y0
        self.segments.append(seg)
        self._start_vals.append(y_prev_end)
        self._breaks = [s.t0 for s in self.segments] + [self.segments[-1].t1()]
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

# ------------------------------
# Multi-channel convenience wrapper
# ------------------------------

@dataclass
class MultiTrajectory:
    tracks: List[Trajectory]

    def __call__(self, t: float) -> np.ndarray:
        return np.array([tr(t) for tr in self.tracks])

    def track(self, i: int) -> Trajectory:
        return self.tracks[i]


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    # Build a trajectory
    T = (Trajectory(y0=0.0, t0=0.0)
         .hold(1.0, value=0.0)
         .step(5.0)
         .ramp(magnitude=5.0, ramprate=1.0)  # duration = 5s
         .random_walk(std=0.1, duration=10.0, dt=0.5, min=0.0, max=12.0, seed=42)
         .hold(3.0))

    ts = np.linspace(0, 25, 101, dtype = np.float64)
    ys = np.array([T(t) for t in ts])
    print(f"y(0..25): mean={ys.mean():.3f}, min={ys.min():.3f}, max={ys.max():.3f}")

    # Cascade update: outer loop updates inner-loop SP every 0.1s
    inner = Trajectory(y0=ys[0], t0=0.0)
    for k, t in enumerate(ts):
        sp = T(t)
        inner.set_now(t, sp)  # O(1) append + clear future
        _ = inner(t)  # controller pulls inner(t)

    # Vector example (e.g., pressure & temperature)
    M = MultiTrajectory([T.clone(), Trajectory(y0=2.0).ramp(magnitude=-1.0, duration=10.0)])
    yv = M(3.3)
    print("vector at t=3.3:", yv)
