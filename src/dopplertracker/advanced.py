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
):
    if len(events) != len(tracks) or len(events) != len(candidate["f0"]) or not events:
        raise ValueError("events, flights and source frequencies must match")
    if validation_fold not in (None, 0, 1, 2):
        raise ValueError("validation_fold must be None or 0, 1, 2")
    working = (
        [smooth_trajectory(t, options.smoothing) for t in tracks] if options.smoothing != "none" else tracks
    )
    clock = float(candidate["clock"])
    local = [Trajectory(t.id, t.times - clock, t.positions, t.max_gap, t.metadata) for t in working]
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
