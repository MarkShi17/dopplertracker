"""Retarded-time acoustics in a homogeneous, optionally moving atmosphere.

Stratified media (``atmosphere.LayeredAtmosphere``) supply their own travel time;
the emission-time solve and Doppler identity below are unchanged.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class Atmosphere:
    c: float = 343.0
    wind: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if (
            not np.isfinite(self.c)
            or self.c <= 0
            or np.shape(self.wind) != (3,)
            or not np.isfinite(self.wind).all()
            or np.linalg.norm(self.wind) >= self.c
        ):
            raise ValueError("sound speed must be positive and greater than wind speed")


class Trajectory:
    def __init__(self, id, times, positions, max_gap=30.0, metadata=None):
        self.id = str(id)
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        if (
            self.times.ndim != 1
            or len(self.times) < 2
            or self.positions.shape != (len(self.times), 3)
            or not np.isfinite(self.times).all()
            or not np.isfinite(self.positions).all()
            or np.any(np.diff(self.times) <= 0)
            or not np.isfinite(max_gap)
            or max_gap <= 0
        ):
            raise ValueError("trajectory requires finite positions and strictly increasing times")
        self.max_gap = max_gap
        self.metadata = metadata or {}
        # Separate interpolators prevent samples beyond a gap from altering endpoint velocities.
        self.segments = [
            idx
            for idx in np.split(np.arange(len(self.times)), np.flatnonzero(np.diff(self.times) > max_gap) + 1)
            if len(idx) >= 2
        ]
        self.pieces = []
        velocity_samples = []
        for idx in self.segments:
            start, end = self.times[idx[0]], self.times[idx[-1]]
            relative_times = self.times[idx] - start
            # Derivatives estimated from vector samples commute with rotations.  In
            # contrast, independent component-wise PCHIP slopes can change when a
            # physically identical trajectory is expressed in a rotated frame.
            derivatives = np.gradient(self.positions[idx], relative_times, axis=0)
            spline = CubicHermiteSpline(
                relative_times, self.positions[idx], derivatives, axis=0, extrapolate=False
            )
            self.pieces.append((start, end, spline, spline.derivative()))
            sample_times = np.r_[self.times[idx], (self.times[idx[:-1]] + self.times[idx[1:]]) / 2] - start
            velocity_samples.append(spline.derivative()(sample_times))
        self.velocity_samples = np.concatenate(velocity_samples) if velocity_samples else np.empty((0, 3))
        self.max_speed = np.max(np.linalg.norm(self.velocity_samples, axis=1), initial=0.0)

    def valid(self, t):
        t = np.asarray(t)
        valid = np.zeros(t.shape, dtype=bool)
        for start, end, _, _ in self.pieces:
            valid |= (t >= start) & (t <= end)
        return valid

    def at(self, t):
        t = np.asarray(t)
        if len(self.pieces) == 1:
            start, _, spline, derivative = self.pieces[0]
            return spline(t - start), derivative(t - start)
        position = np.full(t.shape + (3,), np.nan)
        velocity = np.full_like(position, np.nan)
        for start, end, spline, derivative in self.pieces:
            mask = (t >= start) & (t <= end)
            position[mask], velocity[mask] = spline(t[mask] - start), derivative(t[mask] - start)
        return position, velocity


def _medium(atmosphere, track):
    """Per-flight atmospheres (``atmosphere.FlightAtmospheres``) resolve by track id."""
    return atmosphere.for_flight(track.id) if hasattr(atmosphere, "for_flight") else atmosphere


def travel_time(source_minus_observer, atmosphere=Atmosphere(), observer=None):
    """Travel time and its gradient with respect to source position.

    A stratified medium (``atmosphere.LayeredAtmosphere``) also needs the
    absolute ``observer`` position, because its profile depends on height.
    """
    if not isinstance(atmosphere, Atmosphere):
        if observer is None:
            raise ValueError("a layered atmosphere needs the observer position")
        return atmosphere.travel_time(source_minus_observer, np.asarray(observer, dtype=float)[2])
    r = np.asarray(source_minus_observer, dtype=float)
    w = np.asarray(atmosphere.wind)
    a = atmosphere.c**2 - np.dot(w, w)
    rw = r @ w
    tau = (rw + np.sqrt(rw**2 + a * np.sum(r * r, axis=-1))) / a
    ray = r + tau[..., None] * w
    denominator = atmosphere.c**2 * tau - ray @ w
    gradient = np.divide(
        ray, denominator[..., None], out=np.zeros_like(ray), where=denominator[..., None] > 1e-12
    )
    return tau, gradient


def predict(track, observer, receive_times, f0=1.0, atmosphere=Atmosphere(), drift=0.0, reference=None):
    """Return observed Hz and emission time; NaN outside measured track coverage.

    drift is source Hz per emission second relative to reference (absolute UTC).
    atmosphere may be homogeneous, layered, or per-flight layered.
    """
    atmosphere = _medium(atmosphere, track)
    if isinstance(atmosphere, Atmosphere):
        if track.max_speed + np.linalg.norm(atmosphere.wind) >= atmosphere.c and np.any(
            np.linalg.norm(track.velocity_samples - np.array(atmosphere.wind), axis=1) >= atmosphere.c
        ):
            raise ValueError("propagator requires subsonic aircraft motion relative to the medium")
    elif track.max_speed + atmosphere.max_wind_speed >= atmosphere.min_sound_speed:
        raise ValueError("propagator requires subsonic aircraft motion relative to the medium")
    receive = np.asarray(receive_times, dtype=float)
    low = np.full(receive.shape, track.times[0])
    high = low.copy()
    eligible = np.zeros(receive.shape, dtype=bool)
    for start, end, spline, _ in track.pieces:
        first = start + travel_time(spline(0) - observer, atmosphere, observer)[0]
        last = end + travel_time(spline(end - start) - observer, atmosphere, observer)[0]
        mask = (receive >= first) & (receive <= last)
        low, high = np.where(mask, start, low), np.where(mask, end, high)
        eligible |= mask
    emit = np.clip(receive, low, high)
    for _ in range(16):
        pos, vel = track.at(emit)
        tau, grad = travel_time(pos - observer, atmosphere, observer)
        denom = 1 + np.sum(grad * vel, axis=-1)
        step = (emit + tau - receive) / np.maximum(denom, 0.1)
        emit = np.clip(emit - step, low, high)
        if np.max(np.abs(step), initial=0) < 1e-7:
            break
    pos, vel = track.at(emit)
    tau, grad = travel_time(pos - observer, atmosphere, observer)
    denom = 1 + np.sum(grad * vel, axis=-1)
    ref = track.times[0] if reference is None else reference
    source_hz = f0 + drift * (emit - ref)
    valid = (
        eligible
        & track.valid(emit)
        & (np.abs(emit + tau - receive) < 1e-4)
        & (denom > 0.05)
        & (source_hz > 0)
    )
    freq = source_hz / np.maximum(denom, 0.05)
    return np.where(valid, freq, np.nan), np.where(valid, emit, np.nan)


def closest_approaches(track, observer, atmosphere=Atmosphere()):
    """Acoustic CPA minimizes propagation duration, with finite segment endpoints retained.

    In still air this is geometric CPA. With wind, the interior acoustic minimum
    is where the source frequency is unshifted. Reception time itself is monotone
    for a subsonic source and must NOT be minimized.
    """
    atmosphere = _medium(atmosphere, track)
    results = []
    for idx in track.segments:
        delays, _ = travel_time(track.positions[idx] - observer, atmosphere, observer)
        minima = list(np.flatnonzero((delays[1:-1] <= delays[:-2]) & (delays[1:-1] <= delays[2:])) + 1)
        if delays[0] <= delays[1]:
            minima.insert(0, 0)
        if delays[-1] <= delays[-2]:
            minima.append(len(idx) - 1)
        for i in minima:
            base = track.times[idx[i]]
            if i == 0 or i == len(idx) - 1:
                emission = base
            else:
                fit = minimize_scalar(
                    lambda dt: float(travel_time(track.at(base + dt)[0] - observer, atmosphere, observer)[0]),
                    bounds=(track.times[idx[i - 1]] - base, track.times[idx[i + 1]] - base),
                    method="bounded",
                )
                emission = base + fit.x
            pos, vel = track.at(emission)
            delay, _ = travel_time(pos - observer, atmosphere, observer)
            hit = (float(emission + delay), float(np.linalg.norm(pos - observer)), float(np.linalg.norm(vel)))
            if not results or abs(hit[0] - results[-1][0]) > 0.01:
                results.append(hit)
    return results


def ideal_curve(times, f0, speed, distance, heard_cpa, c=343.0):
    """Analytic straight, constant-speed pass; times and heard_cpa share a clock."""
    if not (0 < speed < c and distance > 0):
        raise ValueError("ideal pass needs 0 < speed < sound speed and distance > 0")
    reception = np.asarray(times) - heard_cpa + distance / c
    a = c * c - speed * speed
    emit = (c * c * reception - np.sqrt(c * c * speed * speed * reception**2 + a * distance * distance)) / a
    return f0 / (1 + speed * speed * emit / (c * np.sqrt(distance * distance + speed * speed * emit * emit)))
