"""Experimental separable inversion and nuisance-model selection.

For fixed geometry, source frequency and its polynomial drift enter linearly.
Solve those small regressions inside the geometry optimizer (variable projection)
instead of making every nuisance coefficient another nonlinear search dimension.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares, lsq_linear

from .physics import Atmosphere, Trajectory, predict
from .refine import extract_ridge
from .scoring import score_curve


@dataclass(frozen=True)
class ModelOptions:
    degree: int = 0
    ridge: str = "peak"
    loss: str = "soft_l1"
    smoothing: str = "none"
    fit_sound_speed: bool = False
    fit_wind: bool = False
    fit_altitude: bool = False
    fit_clock_rate: bool = False
    robust_source: bool = True
    radius: float = 1200.0
    max_nfev: int = 130
    # Aircraft velocity used for the Hermite derivatives: "differenced" positions
    # (the original model), the aircraft's "reported" ADS-B velocity, or "fused"
    # positions and reported velocities from a constant-acceleration smoother.
    velocity: str = "differenced"
    # Per-flight ADS-B time offsets (s) for flights 2..N; the shared recording
    # clock absorbs the first.  Bounded +/-1.5 s with a 0.5 s Gaussian prior.
    fit_track_offsets: bool = False

    def __post_init__(self):
        if self.degree not in (0, 1, 2) or self.ridge not in ("peak", "viterbi"):
            raise ValueError("degree must be 0, 1 or 2 and ridge must be peak or viterbi")
        if self.smoothing not in ("none", "spline", "kalman") or self.loss not in (
            "linear",
            "soft_l1",
            "huber",
            "cauchy",
        ):
            raise ValueError("unsupported smoothing or loss")
        if not np.isfinite(self.radius) or self.radius <= 0 or self.max_nfev < 1:
            raise ValueError("radius and evaluation budget must be positive")
        if self.velocity not in ("differenced", "reported", "fused"):
            raise ValueError("velocity must be differenced, reported or fused")
        if self.velocity == "fused" and self.smoothing != "none":
            raise ValueError("fused velocity smoothing replaces position smoothing")


def _noise_scale(positions, times):
    if len(positions) < 3:
        return np.full(3, 0.1)
    intervals = np.diff(times)
    left = intervals[1:] / (intervals[:-1] + intervals[1:])
    right = 1 - left
    interpolated = left[:, None] * positions[:-2] + right[:, None] * positions[2:]
    # A constant-velocity trajectory has zero residual even with uneven reports.
    # Independent equal-variance position noise contributes variance
    # sigma² * (1 + left² + right²) to each interpolation residual.
    residual = (positions[1:-1] - interpolated) / np.sqrt(1 + left * left + right * right)[:, None]
    center = np.median(residual, axis=0)
    sigma = np.median(np.abs(residual - center), axis=0) / 0.67448975
    return np.clip(sigma, 0.1, 100.0)


def _kalman_axis(t, y, sigma, process=0.3):
    """Constant-velocity Kalman filter followed by Rauch–Tung–Striebel smoothing."""
    states, covs, predicted, predicted_cov, transitions = [], [], [], [], []
    state = np.array([y[0], (y[1] - y[0]) / (t[1] - t[0])])
    covariance = np.diag([sigma * sigma, 10000.0])
    for i in range(len(t)):
        dt = 0.0 if i == 0 else t[i] - t[i - 1]
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = process * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        state = f @ state
        covariance = f @ covariance @ f.T + q
        predicted.append(state.copy())
        predicted_cov.append(covariance.copy())
        transitions.append(f)
        gain = covariance[:, 0] / (covariance[0, 0] + sigma * sigma)
        state = state + gain * (y[i] - state[0])
        covariance = covariance - np.outer(gain, covariance[0, :])
        states.append(state.copy())
        covs.append(covariance.copy())
    smooth = np.array(states)
    for i in range(len(t) - 2, -1, -1):
        gain = np.linalg.solve(predicted_cov[i + 1].T, (covs[i] @ transitions[i + 1].T).T).T
        smooth[i] = states[i] + gain @ (smooth[i + 1] - predicted[i + 1])
    return smooth[:, 0]


def smooth_trajectory(track, method="spline"):
    if method not in ("spline", "kalman"):
        raise ValueError("smoothing method must be spline or kalman")
    positions = track.positions.copy()
    estimates = []
    for idx in track.segments:
        if len(idx) < 5:
            continue
        t = track.times[idx] - track.times[idx[0]]
        observed = track.positions[idx]
        sigma = _noise_scale(observed, t)
        estimates.append(sigma.tolist())
        for axis in range(3):
            if method == "kalman":
                positions[idx, axis] = _kalman_axis(t, observed[:, axis], sigma[axis])
            else:
                spline = UnivariateSpline(t, observed[:, axis], k=3, s=len(idx) * sigma[axis] ** 2)
                positions[idx, axis] = spline(t)
    return Trajectory(
        track.id,
        track.times,
        positions,
        track.max_gap,
        {**track.metadata, "smoothing": method, "estimated_position_noise_m": estimates},
        velocities=track.velocities,
    )


# Reported-velocity measurement noise, m/s (1 sigma).  Horizontal: 1 kt
# quantisation per component (0.149 m/s uniform rms) plus GNSS velocity error.
# Vertical: 64 fpm steps (0.094 m/s rms) plus GNSS rate noise; barometric rate
# reports additionally jitter by +/-128 fpm in level flight.
HORIZONTAL_VELOCITY_SIGMA = 0.2
GEOMETRIC_RATE_SIGMA = 0.25
BARO_RATE_SIGMA = 0.6
# White-jerk spectral density, m^2/s^5.  Straight flight in light turbulence is
# about 1e-3 (0.1 m/s^2 of acceleration change in 10 s); a turn entry reaching
# 3-4 m/s^2 within about 6 s is about 1.  One fixed value cannot serve both, so
# by default each segment picks the maximum-likelihood value from this
# physically bounded grid (horizontal axes jointly, vertical separately).
JERK_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _velocity_noise_scale(positions, velocities, times, velocity_sigma):
    """Per-axis position noise from trapezoidal position/velocity consistency.

    p[i+1] - p[i] - (v[i] + v[i+1]) dt / 2 is exact for constant acceleration
    with uneven spacing; its variance is 2 sigma_p^2 + dt^2 sigma_v^2 / 2.
    Falls back to the constant-velocity estimate when velocities are sparse.
    """
    fallback = _noise_scale(positions, times)
    dt = np.diff(times)
    sigma = fallback.copy()
    for axis in range(3):
        v = velocities[:, axis]
        both = np.isfinite(v[:-1]) & np.isfinite(v[1:])
        if both.sum() < 4:
            continue
        r = (np.diff(positions[:, axis]) - 0.5 * (v[:-1] + v[1:]) * dt)[both]
        spread = (np.median(np.abs(r - np.median(r))) / 0.67448975) ** 2
        floor = np.median(dt[both] ** 2) * np.median(velocity_sigma[:, axis] ** 2) / 2
        sigma[axis] = np.sqrt(max(spread - floor, 0.0) / 2)
    return np.clip(sigma, 0.5, 100.0)


def _fused_axis(t, y, v, sigma, velocity_sigma, jerk):
    """Constant-acceleration Kalman filter and RTS smoother for one axis.

    Measures position where ``y`` is finite (NaN marks an output-only time) and
    velocity where ``v`` is finite; ``velocity_sigma`` is per point.  Returns
    smoothed position, velocity and the innovation log-likelihood of the
    measurements under ``jerk``.
    """
    n = len(t)
    a, b = np.flatnonzero(np.isfinite(y))[:2]
    first = v[0] if np.isfinite(v[0]) else (y[b] - y[a]) / (t[b] - t[a])
    state = np.array([y[0], first, 0.0])
    covariance = np.diag([sigma * sigma, velocity_sigma[0] ** 2 if np.isfinite(v[0]) else 1e4, 4.0])
    loglik = 0.0
    filtered, filtered_cov = np.empty((n, 3)), np.empty((n, 3, 3))
    predicted, predicted_cov, transitions = np.empty((n, 3)), np.empty((n, 3, 3)), np.empty((n, 3, 3))
    for i in range(n):
        dt = 0.0 if i == 0 else t[i] - t[i - 1]
        f = np.array([[1.0, dt, dt * dt / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        q = jerk * np.array(
            [
                [dt**5 / 20, dt**4 / 8, dt**3 / 6],
                [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                [dt**3 / 6, dt**2 / 2, dt],
            ]
        )
        state = f @ state
        covariance = f @ covariance @ f.T + q
        predicted[i], predicted_cov[i], transitions[i] = state, covariance, f
        if not np.isfinite(y[i]):
            filtered[i], filtered_cov[i] = state, covariance
            continue
        if np.isfinite(v[i]):
            h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            z, r = np.array([y[i], v[i]]), np.diag([sigma * sigma, velocity_sigma[i] ** 2])
        else:
            h, z, r = np.array([[1.0, 0.0, 0.0]]), np.array([y[i]]), np.array([[sigma * sigma]])
        innovation_cov = h @ covariance @ h.T + r
        innovation = z - h @ state
        if i > 0:
            loglik -= 0.5 * (
                np.linalg.slogdet(innovation_cov)[1]
                + innovation @ np.linalg.solve(innovation_cov, innovation)
                + len(z) * np.log(2 * np.pi)
            )
        gain = np.linalg.solve(innovation_cov, h @ covariance).T
        state = state + gain @ innovation
        covariance = covariance - gain @ h @ covariance
        covariance = (covariance + covariance.T) / 2
        filtered[i], filtered_cov[i] = state, covariance
    smooth = filtered.copy()
    for i in range(n - 2, -1, -1):
        gain = np.linalg.solve(predicted_cov[i + 1], transitions[i + 1] @ filtered_cov[i]).T
        smooth[i] = filtered[i] + gain @ (smooth[i + 1] - predicted[i + 1])
    return smooth[:, 0], smooth[:, 1], loglik


def _vertical_rate_sigma(track, geometric_sigma, baro_sigma):
    """Per-point vertical-rate sigma from the run-length source list, if consistent."""
    runs = track.metadata.get("vertical_rate_source_runs")
    sigma = np.full(len(track.times), geometric_sigma)
    if runs and sum(count for _, count in runs) == len(track.times):
        labels = np.concatenate([[label] * count for label, count in runs])
        sigma[labels == "baro"] = baro_sigma
    return sigma


def _densify(t, step):
    """Insert equally spaced output times so that no interval exceeds ``step``."""
    pieces = [t[:1]]
    for start, end in zip(t[:-1], t[1:], strict=True):
        count = max(int(np.ceil((end - start) / step)), 1)
        pieces.append(start + (end - start) * np.arange(1, count + 1) / count)
    dense = np.concatenate(pieces)
    dense[np.searchsorted(dense, t[1:] - 1e-9)] = t[1:]  # keep report times exact
    return dense


def fuse_velocity(
    track,
    jerk=None,
    horizontal_sigma=HORIZONTAL_VELOCITY_SIGMA,
    geometric_sigma=GEOMETRIC_RATE_SIGMA,
    baro_sigma=BARO_RATE_SIGMA,
    output_step=2.0,
):
    """Jointly smooth reported positions and velocities per axis and segment.

    Handles irregular report spacing and missing velocity components; without
    any velocities it reduces to a position-only constant-acceleration smoother.
    The returned trajectory uses the smoothed velocities as Hermite derivatives.
    ``jerk`` fixes the white-jerk density (m^2/s^5); None selects it per segment
    by maximum likelihood over ``JERK_GRID``.  Smoothed states are also output
    every ``output_step`` seconds between reports (None: report times only), so
    that Hermite interpolation between sparse reports follows the smoother's
    constant-acceleration dynamics instead of a cubic through two end states.
    """
    grid = JERK_GRID if jerk is None else (jerk,)
    if (
        not all(np.isfinite(q) and q > 0 for q in grid)
        or min(horizontal_sigma, geometric_sigma, baro_sigma) <= 0
    ):
        raise ValueError("jerk density and velocity noise must be positive")
    n = len(track.times)
    reported = track.velocities if track.velocities is not None else np.full((n, 3), np.nan)
    velocity_sigma = np.column_stack(
        (
            np.full(n, horizontal_sigma),
            np.full(n, horizontal_sigma),
            _vertical_rate_sigma(track, geometric_sigma, baro_sigma),
        )
    )
    if output_step is not None and not (np.isfinite(output_step) and output_step > 0):
        raise ValueError("output_step must be positive or None")
    times, positions, velocities = [], [], []
    estimates, selected = [], []
    groups = np.split(np.arange(n), np.flatnonzero(np.diff(track.times) > track.max_gap) + 1)
    for idx in groups:
        if len(idx) < 3:
            times.append(track.times[idx])
            positions.append(track.positions[idx])
            velocities.append(reported[idx])
            continue
        t = track.times[idx] - track.times[idx[0]]
        sigma = _velocity_noise_scale(track.positions[idx], reported[idx], t, velocity_sigma[idx])
        estimates.append(sigma.tolist())
        dense = t if output_step is None else _densify(t, output_step)
        at = np.searchsorted(dense, t)
        y, v, vs = (np.full((len(dense), 3), np.nan) for _ in range(3))
        y[at], v[at], vs[at] = track.positions[idx], reported[idx], velocity_sigma[idx]
        p_out, v_out, choice = np.empty_like(y), np.empty_like(y), []
        for axes in ((0, 1), (2,)):
            fits = [[_fused_axis(dense, y[:, a], v[:, a], sigma[a], vs[:, a], q) for a in axes] for q in grid]
            best = int(np.argmax([sum(f[2] for f in fit) for fit in fits]))
            choice.append(grid[best])
            for a, (pa, va, _) in zip(axes, fits[best], strict=True):
                p_out[:, a], v_out[:, a] = pa, va
        selected.append(choice)
        times.append(dense + track.times[idx[0]])
        positions.append(p_out)
        velocities.append(v_out)
    return Trajectory(
        track.id,
        np.concatenate(times),
        np.concatenate(positions),
        track.max_gap,
        {
            **track.metadata,
            "velocity_fusion": "kalman",
            "estimated_position_noise_m": estimates,
            "selected_jerk_m2_s5": selected,
            "fusion_parameters": {
                "jerk_m2_s5": "maximum likelihood over JERK_GRID" if jerk is None else jerk,
                "horizontal_sigma_m_s": horizontal_sigma,
                "geometric_rate_sigma_m_s": geometric_sigma,
                "baro_rate_sigma_m_s": baro_sigma,
            },
        },
        velocities=np.concatenate(velocities),
        use_velocities=True,
    )


def prepare_track(track, smoothing="none", velocity="differenced"):
    """Apply a fit's position smoothing and velocity model to one raw track."""
    if velocity == "fused":
        return fuse_velocity(track)
    if smoothing != "none":
        track = smooth_trajectory(track, smoothing)
    if velocity == "reported":
        track = Trajectory(
            track.id, track.times, track.positions, track.max_gap, track.metadata, track.velocities, True
        )
    elif velocity != "differenced":
        raise ValueError("velocity must be differenced, reported or fused")
    return track


def project_source(factor, relative_emission, observed, weights, degree, anchor, robust=True, prior=True):
    """Eliminate source coefficients in Hz, Hz/60s, Hz/(60s)^2; no truth input."""
    factor, relative_emission, observed, weights = map(
        np.asarray, (factor, relative_emission, observed, weights)
    )
    if degree not in (0, 1, 2) or any(
        a.shape != observed.shape for a in (factor, relative_emission, weights)
    ):
        raise ValueError("invalid source regression shape or degree")
    if not np.isfinite(np.r_[factor, relative_emission, observed, weights]).all() or np.any(weights < 0):
        raise ValueError("source regression inputs must be finite and weights nonnegative")
    design = factor[:, None] * np.polynomial.polynomial.polyvander(relative_emission / 60, degree)
    prior_sigma = np.array([5.0, 1.2, 0.8])[: degree + 1]
    target = np.r_[anchor, np.zeros(degree)]
    penalty = np.diag(1 / prior_sigma) if prior else np.zeros((degree + 1, degree + 1))
    irls = np.ones(len(observed))
    coefficients = target.copy()
    for _ in range(5 if robust else 1):
        scale = np.sqrt(weights * irls) / 0.3
        a = np.vstack((design * scale[:, None], penalty))
        b = np.r_[observed * scale, penalty @ target]
        coefficients = np.linalg.lstsq(a, b, rcond=None)[0]
        limits = np.array([6.0, 3.0, 6.0])[: degree + 1]
        if prior and np.any(np.abs(coefficients - target) > limits):
            coefficients = lsq_linear(a, b, bounds=(target - limits, target + limits), method="bvls").x
        residual = design @ coefficients - observed
        if robust:
            irls = 1 / np.sqrt(1 + (residual / 0.3) ** 2)
    return coefficients, design @ coefficients - observed


def _peak_ridge(spec, event, times, model, width=7.0):
    offsets = np.arange(-width, width + 0.0625, 0.125)
    intensity = np.zeros((len(times), len(offsets)))
    for h in range(1, event.harmonics + 1):
        hz = (model[:, None] + offsets) * h
        center = spec.sample(times[:, None], hz)
        background = 0.5 * (spec.sample(times[:, None], hz - 3) + spec.sample(times[:, None], hz + 3))
        intensity += np.maximum(center - background, 0) / np.sqrt(h)
    indices = np.argmax(intensity, axis=1)
    rows = np.arange(len(times))
    peak = intensity[rows, indices]
    background = np.median(intensity, axis=1)
    dispersion = 1.4826 * np.median(np.abs(intensity - background[:, None]), axis=1)
    valid = (
        (indices > 0)
        & (indices < len(offsets) - 1)
        & np.isfinite(model)
        & (peak - background > np.maximum(0.06, 2.5 * dispersion))
    )
    ridge = model + offsets[indices]
    good = np.flatnonzero(valid)
    left, center, right = (intensity[good, indices[good] + delta] for delta in (-1, 0, 1))
    curvature = left - 2 * center + right
    subbin = np.divide(0.5 * (left - right), curvature, out=np.zeros_like(center), where=curvature < -1e-9)
    ridge[good] += np.clip(subbin, -0.5, 0.5) * 0.125
    return ridge, valid, np.clip((peak - background) / np.maximum(0.2, 3 * dispersion), 0.2, 1.0)


def _observations(spec, events, tracks, candidate, options, atmosphere):
    observations = []
    for i, (event, track) in enumerate(zip(events, tracks, strict=True)):
        times = event.sample_times(spec, maximum=140)
        predicted, _ = predict(track, np.array(candidate["position"]), times, candidate["f0"][i], atmosphere)
        if options.ridge == "peak":
            ridge, valid, weights = _peak_ridge(spec, event, times, predicted)
        else:
            ridge, valid, quality = extract_ridge(spec, event, predicted, times, width=7.0)
            weights = np.clip(quality, 0.2, 1.0)
        if valid.sum() < 12:
            raise ValueError(f"insufficient ridge samples for {track.id}")
        observations.append((times, ridge, valid, weights))
    return observations


def predict_result(track, event, times, result, index):
    """One interpretation of nuisance parameters for validation and exported results."""
    smoothing = result.get("options", {}).get("smoothing", "none")
    velocity = result.get("options", {}).get("velocity", "differenced")
    if velocity == "fused":
        if track.metadata.get("velocity_fusion") != "kalman":
            track = fuse_velocity(track)
    else:
        if smoothing != "none" and track.metadata.get("smoothing") != smoothing:
            track = smooth_trajectory(track, smoothing)
        track = prepare_track(track, "none", velocity)
    reference = result.get("source_reference_utc", [result["clock"] + event.cpa] * (index + 1))[index]
    origin = result["clock"]
    local = track.relative_to(origin)
    receive = np.asarray(times) * (1 + result.get("clock_rate_ppm", 0.0) * 1e-6)
    observer = np.asarray(result["position"]) - np.array(
        [0.0, 0.0, result.get("aircraft_altitude_offset_m", 0.0)]
    )
    atmosphere = Atmosphere(
        result.get("sound_speed_m_s", 343.0), tuple(result.get("wind_enu_m_s", (0.0, 0.0, 0.0)))
    )
    offsets = result.get("track_time_offsets_s")
    if offsets:
        factor, emit = predict(local, observer, receive + offsets[index], atmosphere=atmosphere)
        emit = emit - offsets[index]
    else:
        factor, emit = predict(local, observer, receive, atmosphere=atmosphere)
    elapsed = emit - (reference - origin)
    drift = result.get("frequency_drift_hz_s", [0.0] * (index + 1))[index]
    curvature = result.get("frequency_curvature_hz_s2", [0.0] * (index + 1))[index]
    return factor * (result["f0"][index] + drift * elapsed + curvature * elapsed**2)


def fit_projected(
    spec,
    events,
    tracks,
    candidate,
    options=ModelOptions(),
    atmosphere=Atmosphere(),
    validation_fold=None,
    exclude_event=None,
    frame=None,
    elevation=None,
):
    if len(events) != len(tracks) or len(events) != len(candidate["f0"]) or not events:
        raise ValueError("events, flights and source frequencies must match")
    if validation_fold not in (None, 0, 1, 2):
        raise ValueError("validation_fold must be None or 0, 1, 2")
    working = (
        [prepare_track(t, options.smoothing, options.velocity) for t in tracks]
        if options.smoothing != "none" or options.velocity != "differenced"
        else tracks
    )
    clock = float(candidate["clock"])
    local = [t.relative_to(clock) for t in working]
    observations = _observations(spec, events, local, candidate, options, atmosphere)
    train = []
    for i, (times, _, valid, _) in enumerate(observations):
        mask = valid.copy()
        if validation_fold is not None:
            mask &= np.floor((times - events[i].window[0]) / 15).astype(int) % 3 != validation_fold
        if i == exclude_event:
            mask[:] = False
        elif mask.sum() < 10:
            raise ValueError("too few training samples after blocking")
        train.append(mask)
    names = ["east", "north", "clock"]
    x0 = [0.0, 0.0, 0.0]
    lower = [-options.radius, -options.radius, -8.0]
    upper = [options.radius, options.radius, 8.0]
    scale = [200.0, 200.0, 1.0]
    for enabled, name, value, lo, hi, unit in [
        (options.fit_sound_speed, "sound_speed", atmosphere.c, atmosphere.c - 18, atmosphere.c + 18, 4.0),
        (options.fit_wind, "wind_e", atmosphere.wind[0], -30.0, 30.0, 8.0),
        (options.fit_wind, "wind_n", atmosphere.wind[1], -30.0, 30.0, 8.0),
        (options.fit_altitude, "altitude", 0.0, -350.0, 350.0, 80.0),
        (options.fit_clock_rate, "clock_rate", 0.0, -2500.0, 2500.0, 500.0),
    ]:
        if enabled:
            names.append(name)
            x0.append(value)
            lower.append(lo)
            upper.append(hi)
            scale.append(unit)
    if options.fit_track_offsets:
        for i in range(1, len(events)):
            names.append(f"track_offset_{i}")
            x0.append(0.0)
            lower.append(-1.5)
            upper.append(1.5)
            scale.append(0.5)
    base = np.asarray(candidate["position"], dtype=float)
    cache = {}

    def state(p):
        values = dict(zip(names, p, strict=True))
        observer = base + np.r_[p[:2], 0.0]
        if frame is not None and elevation is not None:
            lat, lon, _ = frame.to_geodetic(observer)
            observer = frame.to_local([lat, lon, float(elevation(lat, lon))])
        atmo = Atmosphere(
            values.get("sound_speed", atmosphere.c),
            (
                values.get("wind_e", atmosphere.wind[0]),
                values.get("wind_n", atmosphere.wind[1]),
                atmosphere.wind[2],
            ),
        )
        return values, observer, atmo

    def residual(p):
        values, observer, atmo = state(p)
        r = []
        coefficients = []
        predictions = []
        effective_observer = observer - np.array([0.0, 0.0, values.get("altitude", 0.0)])
        for i, (event, track, (times, ridge, valid, weights), mask) in enumerate(
            zip(events, local, observations, train, strict=True)
        ):
            receive = times * (1 + values.get("clock_rate", 0.0) * 1e-6) + p[2]
            if options.fit_track_offsets:
                # Track labels late by d: true position at t is track(t + d).
                offset = values.get(f"track_offset_{i}", 0.0)
                factor, emit = predict(track, effective_observer, receive + offset, atmosphere=atmo)
                emit = emit - offset
            else:
                factor, emit = predict(track, effective_observer, receive, atmosphere=atmo)
            elapsed = emit - event.cpa
            finite = np.isfinite(factor) & np.isfinite(elapsed)
            use = mask & finite
            if use.sum() >= options.degree + 2:
                coefs, _ = project_source(
                    factor[use],
                    elapsed[use],
                    ridge[use],
                    weights[use],
                    options.degree,
                    candidate["f0"][i],
                    options.robust_source,
                )
            else:
                coefs = np.r_[candidate["f0"][i], np.zeros(options.degree)]
            prediction = factor * np.polynomial.polynomial.polyval(elapsed / 60, coefs)
            coefficients.append(coefs)
            predictions.append(prediction)
            r.extend(np.nan_to_num((prediction[mask] - ridge[mask]) * np.sqrt(weights[mask]) / 0.3, nan=50.0))
            if i != exclude_event:
                r.extend(
                    (coefs - np.r_[candidate["f0"][i], np.zeros(options.degree)])
                    / np.array([5.0, 1.2, 0.8])[: options.degree + 1]
                )
        if options.fit_sound_speed:
            r.append((values["sound_speed"] - atmosphere.c) / 6.0)
        if options.fit_wind:
            r.extend(
                [
                    (values["wind_e"] - atmosphere.wind[0]) / 12.0,
                    (values["wind_n"] - atmosphere.wind[1]) / 12.0,
                ]
            )
        if options.fit_altitude:
            r.append(values["altitude"] / 150.0)
        if options.fit_clock_rate:
            r.append(values["clock_rate"] / 1500.0)
        if options.fit_track_offsets:
            r.extend(values[f"track_offset_{i}"] / 0.5 for i in range(1, len(events)))
        cache["coefficients"], cache["predictions"] = coefficients, predictions
        return np.array(r)

    fit = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        x_scale=scale,
        loss=options.loss,
        f_scale=1.0,
        max_nfev=options.max_nfev,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
    )
    residual(fit.x)
    coefficients = cache["coefficients"]
    predictions = cache["predictions"]
    values, observer, atmo = state(fit.x)
    losses, coverage, scores, validation = [], [], [], []
    for event, (times, ridge, valid, _), mask, prediction in zip(
        events, observations, train, predictions, strict=True
    ):
        error = np.nan_to_num(np.abs(prediction - ridge), nan=15.0)
        losses.append(float(np.median(error[valid])))
        held = valid & ~mask
        if held.any():
            # bounded influence for individual corrupted frames; retain entire held-out block
            validation.extend(np.minimum(error[held] ** 2, 9.0))
        coverage.append(float(np.mean(np.isfinite(prediction))))
        scores.append(score_curve(spec, times, prediction, "contrast", event.harmonics))
    return {
        "position": observer.tolist(),
        "clock": float(clock + fit.x[2]),
        "flights": [t.id for t in tracks],
        "f0": [float(c[0]) for c in coefficients],
        "frequency_drift_hz_s": [float(c[1] / 60) if len(c) > 1 else 0.0 for c in coefficients],
        "frequency_curvature_hz_s2": [float(c[2] / 3600) if len(c) > 2 else 0.0 for c in coefficients],
        "source_reference_utc": [clock + e.cpa for e in events],
        "sound_speed_m_s": atmo.c,
        "wind_enu_m_s": list(atmo.wind),
        "aircraft_altitude_offset_m": float(values.get("altitude", 0.0)),
        "clock_rate_ppm": float(values.get("clock_rate", 0.0)),
        **(
            {
                "track_time_offsets_s": [
                    float(values.get(f"track_offset_{i}", 0.0)) for i in range(len(events))
                ]
            }
            if options.fit_track_offsets
            else {}
        ),
        "method": "projected",
        "options": asdict(options),
        "score": float(np.mean(scores)),
        "event_scores": scores,
        "event_residual_hz": losses,
        "coverage": coverage,
        "validation_loss": float(np.mean(validation)) if validation else None,
        "validation_samples": len(validation),
        "excluded_event": exclude_event,
        "success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "nfev": fit.nfev,
        "at_bounds": bool(
            np.any(np.isclose(fit.x, lower, atol=0.01)) or np.any(np.isclose(fit.x, upper, atol=0.01))
        ),
        "uncertainty_note": "Conditional experimental fit; nuisance parameters can imitate position error.",
    }


def select_model(
    spec,
    events,
    tracks,
    candidate,
    options=None,
    atmosphere=Atmosphere(),
    folds=(0, 1, 2),
    frame=None,
    elevation=None,
):
    """Blocked validation on the same recording; never uses truth or scenario labels.

    Ridge peaks are extracted independently per frame from the common coarse
    candidate. No Viterbi smoothing is allowed across training/validation blocks.
    """
    if options is None:
        options = [ModelOptions(degree=d) for d in (0, 1, 2)]
    trials = []
    for choice in options:
        choice = replace(choice, ridge="peak")
        losses = []
        for fold in folds:
            result = fit_projected(
                spec,
                events,
                tracks,
                candidate,
                choice,
                atmosphere,
                validation_fold=fold,
                frame=frame,
                elevation=elevation,
            )
            losses.append(result["validation_loss"] if result["validation_loss"] is not None else 9.0)
        trials.append(
            {
                "options": asdict(choice),
                "mean": float(np.mean(losses)),
                "se": float(np.std(losses) / np.sqrt(len(losses))),
                "fold_losses": losses,
            }
        )
    best = min(trials, key=lambda row: row["mean"])
    threshold = best["mean"] + best["se"]

    # Select by parameter count, independently of the caller's option order.
    def complexity(row):
        opts = row["options"]
        count = (
            opts["degree"] * len(events)
            + sum(opts[name] for name in ("fit_sound_speed", "fit_altitude", "fit_clock_rate"))
            + 2 * opts["fit_wind"]
        )
        return count, opts["degree"], repr(sorted(opts.items()))

    selected = min((row for row in trials if row["mean"] <= threshold), key=complexity)
    result = fit_projected(
        spec,
        events,
        tracks,
        candidate,
        ModelOptions(**selected["options"]),
        atmosphere,
        frame=frame,
        elevation=elevation,
    )
    result["selection"] = "blocked validation, one-standard-error rule"
    result["selection_trials"] = trials
    return result


def fit_consensus(
    spec,
    events,
    tracks,
    candidate,
    options=ModelOptions(degree=1),
    atmosphere=Atmosphere(),
    frame=None,
    elevation=None,
):
    """Compare all-events and leave-one-event-out fits; retain a majority consensus."""
    if len(events) < 3:
        raise ValueError("event consensus requires at least three events")
    attempts = []
    for excluded in [None, *range(len(events))]:
        result = fit_projected(
            spec,
            events,
            tracks,
            candidate,
            options,
            atmosphere,
            exclude_event=excluded,
            frame=frame,
            elevation=elevation,
        )
        residuals = np.sort(result["event_residual_hz"])
        # Same fixed number of events for every hypothesis, including all-events fit.
        result["consensus_loss"] = float(np.mean(residuals[: len(events) - 1]))
        attempts.append(result)
    best = min(attempts, key=lambda row: row["consensus_loss"])
    best["consensus_trials"] = [
        {"excluded_event": r["excluded_event"], "loss": r["consensus_loss"]} for r in attempts
    ]
    return best
