"""Experimental separable inversion and nuisance-model selection.

For fixed geometry, source frequency and its polynomial drift enter linearly.
Solve those small regressions inside the geometry optimizer (variable projection)
instead of making every nuisance coefficient another nonlinear search dimension.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares, lsq_linear

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
    )


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
    precision = np.where(valids, 1 / np.maximum(np.nan_to_num(sigmas, nan=np.inf), 1e-6) ** 2, 0.0)
    rows = np.arange(ridges.shape[1])
    best = np.argmax(precision, axis=0)
    reference, reference_sigma = ridges[best, rows], sigmas[best, rows]
    tolerance = 3 * np.sqrt(np.nan_to_num(sigmas, nan=np.inf) ** 2 + reference_sigma**2) + 0.2
    precision = np.where(np.abs(np.nan_to_num(ridges - reference, nan=np.inf)) <= tolerance, precision, 0.0)
    total = precision.sum(axis=0)
    valid = total > 0
    ridge = np.where(
        valid, (precision * np.nan_to_num(ridges)).sum(axis=0) / np.maximum(total, 1e-300), reference
    )
    return ridge, valid, np.where(valid, 1 / np.sqrt(np.maximum(total, 1e-300)), np.inf)


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
    both = valid & reference_valid
    if both.sum() < 12:
        return 0.0
    tolerance = 3 * np.hypot(sigma[both], reference_sigma[both]) + 0.2
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
    if smoothing != "none" and track.metadata.get("smoothing") != smoothing:
        track = smooth_trajectory(track, smoothing)
    reference = result.get("source_reference_utc", [result["clock"] + event.cpa] * (index + 1))[index]
    origin = result["clock"]
    local = Trajectory(track.id, track.times - origin, track.positions, track.max_gap, track.metadata)
    receive = np.asarray(times) * (1 + result.get("clock_rate_ppm", 0.0) * 1e-6)
    observer = np.asarray(result["position"]) - np.array(
        [0.0, 0.0, result.get("aircraft_altitude_offset_m", 0.0)]
    )
    atmosphere = Atmosphere(
        result.get("sound_speed_m_s", 343.0), tuple(result.get("wind_enu_m_s", (0.0, 0.0, 0.0)))
    )
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
):
    """Projected least squares; ``audio`` (``demod.Audio``) is required for ``ridge="demod"``.

    ``keep_observations`` adds the final measured ridges to the result for diagnostics.
    """
    if len(events) != len(tracks) or len(events) != len(candidate["f0"]) or not events:
        raise ValueError("events, flights and source frequencies must match")
    if validation_fold not in (None, 0, 1, 2):
        raise ValueError("validation_fold must be None or 0, 1, 2")
    if options.ridge == "demod" and audio is None:
        raise ValueError("demodulation ridges need the raw audio")
    working = (
        [smooth_trajectory(t, options.smoothing) for t in tracks] if options.smoothing != "none" else tracks
    )
    clock = float(candidate["clock"])
    local = [Trajectory(t.id, t.times - clock, t.positions, t.max_gap, t.metadata) for t in working]
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

        def guide(t):
            factor, emit = predict(local[i], effective_observer, np.asarray(t) * rate + p[2], atmosphere=atmo)
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
