"""Experimental separable inversion and nuisance-model selection.

For fixed geometry, source frequency and its polynomial drift enter linearly.
Solve those small regressions inside the geometry optimizer (variable projection)
instead of making every nuisance coefficient another nonlinear search dimension.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares, lsq_linear

from .atmosphere import atmosphere_from_result
from .demod import demodulate, stationary_lines
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
    # Measurement options.  The defaults reproduce one extraction around the start.
    # max_frames: frames per event (None keeps every spectrogram frame).
    # reextract: re-measure around the fitted model and refit up to this many times.
    # noise: "fixed" 0.3 Hz; "robust" per-event scale estimated from the residuals;
    #   "snr" also derives per-frame uncertainty from measurement signal-to-noise.
    # harmonic_mode: "sum" merges harmonic intensity; "ratio" measures each harmonic
    #   and averages them as Doppler ratios with inverse-variance weights.
    # max_harmonic: highest harmonic searched (0 uses each event's harmonic count).
    # ridge="demod" measures from raw audio (fit_projected(..., audio=...)).
    max_frames: int | None = 140
    reextract: int = 0
    noise: str = "fixed"
    harmonic_mode: str = "sum"
    max_harmonic: int = 0
    demod_window: float = 2.0

    def __post_init__(self):
        if self.degree not in (0, 1, 2) or self.ridge not in ("peak", "viterbi", "demod"):
            raise ValueError("degree must be 0, 1 or 2 and ridge must be peak, viterbi or demod")
        if (
            (self.max_frames is not None and self.max_frames < 5)
            or not 0 <= self.reextract <= 10
            or self.noise not in ("fixed", "robust", "snr")
            or self.harmonic_mode not in ("sum", "ratio")
            or self.max_harmonic not in (0, 1, 2, 3, 4)
            or not 0.5 <= self.demod_window <= 8.0
        ):
            raise ValueError("unsupported measurement options")
        if self.ridge == "viterbi" and (
            self.harmonic_mode != "sum" or self.noise == "snr" or self.max_harmonic
        ):
            raise ValueError("viterbi ridges support only summed harmonics without SNR weights")
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


def _peak_contrast(intensity, indices):
    """Peak height above the frame median, in whitened (log-power over robust spread) units."""
    return intensity[np.arange(len(indices)), indices] - np.median(intensity, axis=1)


def _harmonic_peak(spec, times, model, harmonic, width=7.0):
    """``_peak_ridge`` for one harmonic on a 0.125 Hz grid in that harmonic's Hz.

    The search covers +/-``width`` fundamental-equivalent Hz.  Returns the
    fundamental-equivalent ridge, validity, the legacy weight and peak contrast.
    """
    step = 0.125
    offsets = np.arange(-width * harmonic, width * harmonic + step / 2, step)
    hz = model[:, None] * harmonic + offsets
    center = spec.sample(times[:, None], hz)
    background = 0.5 * (spec.sample(times[:, None], hz - 3) + spec.sample(times[:, None], hz + 3))
    intensity = np.maximum(center - background, 0)
    indices = np.argmax(intensity, axis=1)
    rows = np.arange(len(times))
    peak = intensity[rows, indices]
    level = np.median(intensity, axis=1)
    dispersion = 1.4826 * np.median(np.abs(intensity - level[:, None]), axis=1)
    valid = (
        (indices > 0)
        & (indices < len(offsets) - 1)
        & np.isfinite(model)
        & (peak - level > np.maximum(0.06, 2.5 * dispersion))
    )
    shift = offsets[indices]
    good = np.flatnonzero(valid)
    left, middle, right = (intensity[good, indices[good] + delta] for delta in (-1, 0, 1))
    curvature = left - 2 * middle + right
    subbin = np.divide(0.5 * (left - right), curvature, out=np.zeros_like(middle), where=curvature < -1e-9)
    shift[good] += np.clip(subbin, -0.5, 0.5) * step
    weight = np.clip((peak - level) / np.maximum(0.2, 3 * dispersion), 0.2, 1.0)
    return model + shift / harmonic, valid, weight, _peak_contrast(intensity, indices)


def _peak_sigma(contrast, window_seconds, sweep):
    """STFT peak-frequency uncertainty (Hz at the measured harmonic).

    Noise: the Cramér-Rao bound for a Hann-windowed tone, sqrt(8 / rho) / (2 pi T),
    with power SNR rho ~ exp(contrast) because whitened power is roughly log power
    in units of its robust spread.  Smear: a tone sweeping ``sweep`` Hz within one
    window.  On synthetic waveform dev seeds 0-2 (4 s window) the peak error was
    0.055 Hz below 1 Hz of sweep, grew about 0.06 x sweep, and broke down beyond
    ~8 Hz, where the smeared ridge falls under the +/-3 Hz sideband subtraction.
    """
    rho = np.exp(np.clip(contrast, 0.0, 30.0))
    noise = np.sqrt(8.0 / rho) / (2 * np.pi * window_seconds)
    smear = np.hypot(0.22 / window_seconds, 0.06 * sweep) * (1 + (sweep / 9.5) ** 4)
    return np.hypot(noise, smear)


def _sweep(times, model, harmonic, window_seconds):
    if len(times) < 2:
        return np.zeros(len(times))
    return harmonic * np.abs(np.nan_to_num(np.gradient(model, times))) * window_seconds


def _combine_harmonics(ridges, valids, sigmas):
    """Inverse-variance mean of fundamental-equivalent harmonic estimates.

    Harmonic h measures the same Doppler ratio with h times finer resolution in
    fundamental Hz.  Harmonics that disagree with the most precise valid one
    (a neighbouring line, not the same tone) are dropped for that frame.
    """
    ridges, valids, sigmas = map(np.asarray, (ridges, valids, sigmas))
    valids = valids & np.isfinite(ridges) & np.isfinite(sigmas)
    ridges, sigmas = np.where(valids, ridges, 0.0), np.where(valids, sigmas, np.inf)
    precision = np.where(valids, 1 / np.maximum(sigmas, 1e-6) ** 2, 0.0)
    rows = np.arange(ridges.shape[1])
    best = np.argmax(precision, axis=0)
    reference, reference_sigma = ridges[best, rows], sigmas[best, rows]
    tolerance = 3 * np.hypot(sigmas, reference_sigma) + 0.2
    precision = np.where(np.abs(ridges - reference) <= tolerance, precision, 0.0)
    total = precision.sum(axis=0)
    valid = total > 0
    safe = np.where(valid, total, 1.0)
    ridge = np.where(valid, (precision * ridges).sum(axis=0) / safe, reference)
    return ridge, valid, np.where(valid, 1 / np.sqrt(safe), np.inf)


def _scatter(ridge, valid):
    """A harmonic's realised precision: robust scatter of second differences of its valid frames.

    Independent errors of size sigma give r[k] - (r[k-1] + r[k+1]) / 2 a spread of
    sqrt(1.5) sigma; the smooth Doppler curve contributes little at 0.5-1 s spacing.
    """
    values = ridge[valid]
    if len(values) < 12:
        return np.inf
    second = values[1:-1] - 0.5 * (values[:-2] + values[2:])
    return float(max(1.4826 * np.median(np.abs(second)) / np.sqrt(1.5), 1e-3))


def _agreement(ridge, valid, sigma, reference, reference_valid, reference_sigma):
    # The tolerance depends only on the reference: a buried harmonic's own large
    # sigma must not make it "agree" with everything.
    both = valid & reference_valid
    if both.sum() < 12:
        return 0.0
    tolerance = 3 * reference_sigma[both] + 0.2
    return float(np.mean(np.abs(ridge[both] - reference[both]) <= tolerance))


def _measure(spec, event, times, guide, options, audio, lines, half_band):
    """Per-frame fundamental-equivalent ridge, validity and 1-sigma uncertainty in Hz."""
    predicted = guide(times)
    stft_seconds = 1.0 / (spec.frequencies[1] - spec.frequencies[0])
    measured = {}
    for h in range(1, (options.max_harmonic or event.harmonics) + 1):
        if options.ridge == "demod":
            ridge, valid, sigma, _ = demodulate(
                audio, times, guide, h, half_band * h, options.demod_window, lines
            )
            if options.noise != "snr":
                sigma = np.full(len(times), 0.3 / h)
        else:
            ridge, valid, weight, contrast = _harmonic_peak(spec, times, predicted, h, half_band)
            if options.noise == "snr":
                sigma = _peak_sigma(contrast, stft_seconds, _sweep(times, predicted, h, stft_seconds)) / h
            else:
                sigma = 0.3 / (h * np.sqrt(weight))
            # A spectrogram peak's precision is not known a priori (smearing, a
            # buried harmonic): rescale it to that harmonic's realised scatter.
            if valid.any():
                sigma = sigma * _scatter(ridge, valid) / np.median(sigma[valid])
        measured[h] = (ridge, valid, sigma)
    declared = [measured[h] for h in measured if h <= event.harmonics]
    reference = _combine_harmonics(*zip(*declared))
    # A harmonic beyond those declared counts only if it mostly agrees with them.
    extra = [m for h, m in measured.items() if h > event.harmonics and _agreement(*m, *reference) >= 0.5]
    if not extra:
        return reference
    return _combine_harmonics(*zip(*declared, *extra))


def _observations(
    spec, events, tracks, candidate, options, atmosphere, guides=None, audio=None, lines=None, half_bands=None
):
    """Measure each event around ``guides`` (callables of time), by default the start candidate.

    Weights encode the per-frame uncertainty sigma as (0.3 / sigma)**2, so the
    residual (predicted - ridge) * sqrt(w) / 0.3 is in units of sigma.
    """
    observations = []
    for i, (event, track) in enumerate(zip(events, tracks, strict=True)):
        times = event.sample_times(spec, maximum=options.max_frames or len(spec.times))
        if guides is None:
            predicted, _ = predict(
                track, np.array(candidate["position"]), times, candidate["f0"][i], atmosphere
            )
        else:
            predicted = guides[i](times)
        if options.ridge == "demod" or options.harmonic_mode == "ratio":
            guide = guides[i] if guides is not None else _start_guide(track, candidate, i, atmosphere)
            half_band = 7.0 if half_bands is None else half_bands[i]
            line_list = lines[i] if lines is not None else ()
            ridge, valid, sigma = _measure(spec, event, times, guide, options, audio, line_list, half_band)
            center = np.median(sigma[valid]) if valid.any() else 0.3
            if options.noise == "snr":
                # Bounded influence: no frame counts for more than 9x the event median.
                sigma = np.maximum(sigma, center / 3)
            elif options.noise == "fixed":
                # The fixed convention: the event's median frame uncertainty is 0.3 Hz.
                sigma = sigma * 0.3 / center
            weights = np.where(valid, (0.3 / np.where(valid, sigma, 1.0)) ** 2, 0.0)
        elif options.ridge == "peak":
            if options.max_harmonic:
                event = replace(event, harmonics=options.max_harmonic)
            ridge, valid, weights = _peak_ridge(spec, event, times, predicted)
            if options.noise == "snr":
                intensity_sigma = _summed_peak_sigma(spec, event, times, predicted)
                center = np.median(intensity_sigma[valid]) if valid.any() else 0.3
                weights = (0.3 / np.maximum(intensity_sigma, center / 3)) ** 2
        else:
            ridge, valid, quality = extract_ridge(spec, event, predicted, times, width=7.0)
            weights = np.clip(quality, 0.2, 1.0)
        if valid.sum() < 12:
            raise ValueError(f"insufficient ridge samples for {track.id}")
        observations.append((times, ridge, valid, weights))
    return observations


def _start_guide(track, candidate, index, atmosphere):
    position = np.array(candidate["position"])
    return lambda t: predict(track, position, t, candidate["f0"][index], atmosphere)[0]


def _summed_peak_sigma(spec, event, times, model, width=7.0):
    """Contrast-derived uncertainty for the summed-harmonic ridge of ``_peak_ridge``."""
    offsets = np.arange(-width, width + 0.0625, 0.125)
    intensity = np.zeros((len(times), len(offsets)))
    for h in range(1, event.harmonics + 1):
        hz = (model[:, None] + offsets) * h
        center = spec.sample(times[:, None], hz)
        background = 0.5 * (spec.sample(times[:, None], hz - 3) + spec.sample(times[:, None], hz + 3))
        intensity += np.maximum(center - background, 0) / np.sqrt(h)
    contrast = _peak_contrast(intensity, np.argmax(intensity, axis=1))
    window = 1.0 / (spec.frequencies[1] - spec.frequencies[0])
    return _peak_sigma(contrast, window, _sweep(times, model, 1, window))


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
    atmosphere = atmosphere_from_result(result)
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
    audio=None,
    keep_observations=False,
    uncertainty=False,
):
    """Projected least squares; ``audio`` (``demod.Audio``) is required for ``ridge="demod"``.

    ``keep_observations`` adds the final measured ridges to the result for diagnostics.
    ``uncertainty`` (True or ``uncertainty.InputPriors``) adds a linearised error
    region under ``result["uncertainty"]``; the fit itself is unchanged.
    """
    if len(events) != len(tracks) or len(events) != len(candidate["f0"]) or not events:
        raise ValueError("events, flights and source frequencies must match")
    if validation_fold not in (None, 0, 1, 2):
        raise ValueError("validation_fold must be None or 0, 1, 2")
    if options.ridge == "demod" and audio is None:
        raise ValueError("demodulation ridges need the raw audio")
    working = (
        [prepare_track(t, options.smoothing, options.velocity) for t in tracks]
        if options.smoothing != "none" or options.velocity != "differenced"
        else tracks
    )
    clock = float(candidate["clock"])
    local = [t.relative_to(clock) for t in working]
    lines = None
    if options.ridge == "demod":
        # Hum is persistent; +/-300 s is long enough that an aircraft tone is not.
        lines = [stationary_lines(audio, (e.window[0] - 300, e.window[1] + 300)) for e in events]
    measured = _observations(spec, events, local, candidate, options, atmosphere, audio=audio, lines=lines)

    def blocked(observations):
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
        return train

    observations = measured
    train = blocked(observations)
    names = ["east", "north", "clock"]
    x0 = [0.0, 0.0, 0.0]
    lower = [-options.radius, -options.radius, -8.0]
    upper = [options.radius, options.radius, 8.0]
    scale = [200.0, 200.0, 1.0]
    # A layered profile keeps its shape: sound speed and wind become constant offsets
    # (same bounds and prior widths, centred on zero).
    layered = not isinstance(atmosphere, Atmosphere)
    c_ref, we_ref, wn_ref = (0.0, 0.0, 0.0) if layered else (atmosphere.c, *atmosphere.wind[:2])
    for enabled, name, value, lo, hi, unit in [
        (options.fit_sound_speed, "sound_speed", c_ref, c_ref - 18, c_ref + 18, 4.0),
        (options.fit_wind, "wind_e", we_ref, -30.0, 30.0, 8.0),
        (options.fit_wind, "wind_n", wn_ref, -30.0, 30.0, 8.0),
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
        if layered:
            atmo = atmosphere.offset(
                values.get("sound_speed", 0.0), (values.get("wind_e", 0.0), values.get("wind_n", 0.0))
            )
        else:
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
            r.append((values["sound_speed"] - c_ref) / 6.0)
        if options.fit_wind:
            r.extend(
                [
                    (values["wind_e"] - we_ref) / 12.0,
                    (values["wind_n"] - wn_ref) / 12.0,
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

    def solve(start):
        return least_squares(
            residual,
            start,
            bounds=(lower, upper),
            x_scale=scale,
            loss=options.loss,
            f_scale=1.0,
            max_nfev=options.max_nfev,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )

    def fitted_guide(p, coefficients, i):
        values, observer, atmo = state(p)
        effective_observer = observer - np.array([0.0, 0.0, values.get("altitude", 0.0)])
        rate = 1 + values.get("clock_rate", 0.0) * 1e-6

        offset = values.get(f"track_offset_{i}", 0.0)

        def guide(t):
            receive = np.asarray(t) * rate + p[2]
            if options.fit_track_offsets:
                factor, emit = predict(local[i], effective_observer, receive + offset, atmosphere=atmo)
                emit = emit - offset
            else:
                factor, emit = predict(local[i], effective_observer, receive, atmosphere=atmo)
            return factor * np.polynomial.polynomial.polyval((emit - events[i].cpa) / 60, coefficients[i])

        return guide

    inflation = _correlation_inflation(spec, measured, options)
    noise_scales = None if options.noise == "fixed" else _start_scales(measured)
    history, start, fit = [], x0, None
    for iteration in range(options.reextract + 1):
        if iteration:
            guides = [fitted_guide(fit.x, cache["coefficients"], i) for i in range(len(events))]
            spread = [_robust_scale(o[1], o[2], p) for o, p in zip(observations, cache["predictions"])]
            # Later passes start near the data, so the search narrows to cover the residual misfit.
            half_bands = [float(np.clip(1.0 + 6.0 * s, 2.0, 7.0)) for s in spread]
            measured = _observations(
                spec, events, local, candidate, options, atmosphere, guides, audio, lines, half_bands
            )
        for _ in range(1 if noise_scales is None else 2):
            if noise_scales is not None:
                # sigma = inflation * sqrt((scale * frame sigma)^2 + floor^2), as a weight (0.3 / sigma)^2.
                observations = [
                    (t, r, v, 0.09 / (c * c * (s * s * 0.09 / np.maximum(w, 1e-12) + f * f)))
                    for (t, r, v, w), (s, f), c in zip(measured, noise_scales, inflation, strict=True)
                ]
            else:
                observations = measured
            train = blocked(observations)
            fit = solve(start)
            residual(fit.x)
            start = fit.x
            if noise_scales is not None:
                noise_scales = _residual_scales(
                    measured, train, cache["predictions"], noise_scales, additive=options.noise == "snr"
                )
        history.append(state(fit.x)[1])
        if iteration and np.linalg.norm(history[-1][:2] - history[-2][:2]) < 1.0:
            break
    coefficients = cache["coefficients"]
    predictions = cache["predictions"]
    values, observer, atmo = state(fit.x)
    losses, coverage, scores, validation, rms = [], [], [], [], []
    for event, (times, ridge, valid, _), mask, prediction in zip(
        events, observations, train, predictions, strict=True
    ):
        error = np.nan_to_num(np.abs(prediction - ridge), nan=15.0)
        losses.append(float(np.median(error[valid])))
        rms.append(float(np.sqrt(np.mean(error[valid] ** 2))))
        held = valid & ~mask
        if held.any():
            # bounded influence for individual corrupted frames; retain entire held-out block
            validation.extend(np.minimum(error[held] ** 2, 9.0))
        coverage.append(float(np.mean(np.isfinite(prediction))))
        scores.append(score_curve(spec, times, prediction, "contrast", event.harmonics))
    result = {
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
        "event_residual_rms_hz": rms,
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
    if layered:
        # sound_speed_m_s / wind_enu_m_s above are the profile's values at 0 m.
        result["atmosphere"] = atmo.to_record()
    if len(history) > 1 or noise_scales is not None or options.ridge == "demod":
        result["measurement"] = {
            "passes": len(history),
            "position_history": [p.tolist() for p in history],
            "noise_scale_and_floor_hz": None if noise_scales is None else [list(s) for s in noise_scales],
            "correlation_inflation": [float(c) for c in inflation],
            "valid_frames": [int(o[2].sum()) for o in observations],
            "frames": [len(o[0]) for o in observations],
            "median_sigma_hz": [
                float(np.median(0.3 / np.sqrt(o[3][o[2]]))) if o[2].any() else None for o in observations
            ],
            "stationary_lines_hz": None if lines is None else [[float(f) for f in x] for x in lines],
        }
    if keep_observations:
        result["observations"] = [
            {
                "times": times.tolist(),
                "ridge_hz": np.nan_to_num(ridge, nan=0.0).tolist(),
                "valid": valid.tolist(),
                "sigma_hz": np.where(valid, 0.3 / np.sqrt(np.maximum(weights, 1e-12)), 0.0).tolist(),
                "model_hz": np.nan_to_num(prediction, nan=0.0).tolist(),
            }
            for (times, ridge, valid, weights), prediction in zip(observations, predictions, strict=True)
        ]
    if uncertainty:
        from .uncertainty import InputPriors, position_uncertainty

        try:
            result["uncertainty"] = position_uncertainty(
                result,
                events,
                tracks,
                frame,
                elevation,
                uncertainty if isinstance(uncertainty, InputPriors) else None,
                [(t, r, m, w) for (t, r, _, w), m in zip(observations, train, strict=True)],
            )
        except (ValueError, np.linalg.LinAlgError) as exc:  # keep the fit; report why there is no region
            result["uncertainty"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def _robust_scale(ridge, valid, prediction):
    error = (prediction - ridge)[valid & np.isfinite(prediction)]
    if len(error) < 5:
        return 1.0
    return float(1.4826 * np.median(np.abs(error - np.median(error))))


def _correlation_inflation(spec, observations, options):
    """sqrt(frames per independent measurement) for overlapping analysis windows.

    A Hann window of length T averages noise over 2T/3 s, so frames closer
    than that share noise; without this, keeping every frame would overstate
    the information relative to the priors.
    """
    window = options.demod_window if options.ridge == "demod" else 1 / np.diff(spec.frequencies[:2])[0]
    spacing = [float(np.median(np.diff(o[0]))) for o in observations]
    return [float(np.sqrt(max(1.0, 2 * window / 3 / s))) for s in spacing]


def _start_scales(observations):
    # Start from the fixed 0.3 Hz convention: scale so the median frame sigma is 0.3 Hz, no floor.
    return [(float(0.3 / np.median(0.3 / np.sqrt(w[v]))), 0.0) for _, _, v, w in observations]


def _residual_scales(observations, train, predictions, previous, additive=False):
    """Per-event (scale, floor in Hz) making residuals unit-variance in a robust sense.

    ``robust``: sigma = scale * frame sigma.  ``snr`` (``additive``): the SNR-derived
    sigma is a lower bound, so excess residual is an additive floor
    sigma^2 = frame sigma^2 + floor^2 (model misfit, real tone wander), not a
    multiple of the frame's measurement noise.
    """
    scales = []
    for (_, ridge, _, weights), mask, prediction, old in zip(observations, train, predictions, previous):
        use = mask & np.isfinite(prediction)
        if use.sum() < 12:
            scales.append(old)
            continue
        error = np.abs(prediction - ridge)[use]
        sigma = 0.3 / np.sqrt(np.maximum(weights[use], 1e-12))
        ratio = 1.4826 * np.median(error / sigma)
        if not additive or ratio <= 1:
            scales.append((float(max(ratio, 0.05)), 0.0))
            continue
        low, high = 0.0, float(np.max(error)) + 1.0
        for _ in range(50):
            floor = 0.5 * (low + high)
            if 1.4826 * np.median(error / np.hypot(sigma, floor)) > 1:
                low = floor
            else:
                high = floor
        scales.append((1.0, float(0.5 * (low + high))))
    return scales


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
