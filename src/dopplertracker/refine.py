"""Continuous robust multi-event fit to fixed, locally extracted spectral ridges."""

import numpy as np
from scipy.optimize import least_squares, minimize

from .physics import Atmosphere, Trajectory, predict
from .scoring import score_curve


def refine_spectral(
    spec,
    events,
    tracks,
    candidate,
    method="brightness",
    atmosphere=Atmosphere(),
    radius=1000.0,
    frame=None,
    elevation=None,
):
    """Continuous version of the original spectral scorer, for a fair ablation.

    Same shared clock and constant per-event source frequencies as the baseline;
    no ridge extraction or source-frequency drift model.
    """
    base = np.asarray(candidate["position"], dtype=float)
    clock = candidate["clock"]
    local_tracks = [Trajectory(t.id, t.times - clock, t.positions, t.max_gap, t.metadata) for t in tracks]
    times_list = [event.sample_times(spec) for event in events]
    f0 = np.asarray(candidate["f0"])
    n = len(events)

    def observer(p):
        position = base + np.r_[p[:2] * 200, 0.0]
        if frame is not None and elevation is not None:
            lat, lon, _ = frame.to_geodetic(position)
            position = frame.to_local([lat, lon, float(elevation(lat, lon))])
        return position

    def objective(p):
        position = observer(p)
        scores = []
        for i, (event, track, times) in enumerate(zip(events, local_tracks, times_list, strict=True)):
            f, _ = predict(track, position, times + p[2], f0[i] + p[3 + i], atmosphere)
            scores.append(score_curve(spec, times, f, method, event.harmonics))
        return -float(np.mean(scores))

    initial = np.zeros(3 + n)
    simplex = np.vstack([initial, initial + np.diag([0.5, 0.5, 0.3, *([0.3] * n)])])
    fit = minimize(
        objective,
        initial,
        method="Nelder-Mead",
        bounds=[(-radius / 200, radius / 200)] * 2 + [(-10, 10)] + [(-5, 5)] * n,
        options={"initial_simplex": simplex, "maxiter": 1400, "xatol": 1e-4, "fatol": 1e-5},
    )
    return {
        "position": observer(fit.x).tolist(),
        "clock": float(clock + fit.x[2]),
        "f0": (f0 + fit.x[3:]).tolist(),
        "flights": [t.id for t in tracks],
        "score": float(-fit.fun),
        "success": bool(fit.success),
        "method": method + "-continuous",
        "sound_speed_m_s": atmosphere.c,
        "wind_enu_m_s": list(atmosphere.wind),
    }


def extract_ridge(spec, event, predicted, times, width=5.0, step=0.25, smoothness=2.0):
    """Viterbi offsets around a candidate curve; freezes observations BEFORE fitting.

    This is candidate-guided extraction, not independent confirmation of a flight.
    """
    offsets = np.arange(-width, width + step * 0.5, step)
    evidence = np.zeros((len(times), len(offsets)))
    for h in range(1, event.harmonics + 1):
        hz = (predicted[:, None] + offsets) * h
        center = spec.sample(times[:, None], hz)
        bg = 0.5 * (spec.sample(times[:, None], hz - 3) + spec.sample(times[:, None], hz + 3))
        evidence += np.maximum(center - bg, 0) / np.sqrt(h)
    # Bound single-frame influence; a loud transient must not drag the entire ridge.
    evidence = np.minimum(evidence, np.maximum(np.quantile(evidence, 0.95), 1.0))
    penalty = smoothness * (offsets[:, None] - offsets[None, :]) ** 2
    values = evidence[0].copy()
    parents = np.empty(evidence.shape, dtype=np.int32)
    for i in range(1, len(times)):
        choices = values[:, None] - penalty
        parents[i] = np.argmax(choices, axis=0)
        values = evidence[i] + np.max(choices, axis=0)
    path = np.empty(len(times), dtype=int)
    path[-1] = np.argmax(values)
    for i in range(len(times) - 1, 0, -1):
        path[i - 1] = parents[i, path[i]]
    ridge = predicted + offsets[path]
    quality = evidence[np.arange(len(times)), path]
    # Refine the quantized peak by a three-bin parabola.
    inside = (path > 0) & (path < len(offsets) - 1) & np.isfinite(predicted)
    rows = np.flatnonzero(inside)
    left, center, right = (
        evidence[rows, path[rows] - 1],
        evidence[rows, path[rows]],
        evidence[rows, path[rows] + 1],
    )
    curvature = left - 2 * center + right
    displacement = np.divide(
        0.5 * (left - right), curvature, out=np.zeros_like(center), where=curvature < -1e-8
    )
    ridge[rows] += np.clip(displacement, -0.5, 0.5) * step
    valid = inside & (quality > max(0.05, float(np.max(quality)) * 0.08))
    return ridge, valid, quality


def refine_candidate(
    spec,
    events,
    tracks,
    candidate,
    atmosphere=Atmosphere(),
    radius=1000.0,
    fit_drift=False,
    fit_sound_speed=False,
    frame=None,
    elevation=None,
    ridge_width=5.0,
    loss="soft_l1",
):
    if len(events) != len(tracks) or len(events) != len(candidate["f0"]):
        raise ValueError("events, assigned flights and source frequencies must have equal length")
    if radius <= 0:
        raise ValueError("refinement radius must be positive")
    base = np.array(candidate["position"], dtype=float)
    clock = candidate["clock"]
    # Optimize in a recording-relative clock: finite differences at Unix epochs
    # otherwise round away small clock perturbations and destroy that Jacobian column.
    tracks = [Trajectory(t.id, t.times - clock, t.positions, t.max_gap, t.metadata) for t in tracks]
    observations, masks, times_list, references = [], [], [], []
    for e, track, f0 in zip(events, tracks, candidate["f0"], strict=True):
        times = e.sample_times(spec)
        model, _ = predict(track, base, times, f0, atmosphere)
        ridge, valid, _ = extract_ridge(spec, e, model, times, width=ridge_width)
        if valid.sum() < 12:
            raise ValueError(f"insufficient ridge evidence for flight {track.id}")
        observations.append(ridge)
        masks.append(valid)
        times_list.append(times)
        references.append(e.cpa)
    n = len(events)
    initial = [0.0, 0.0, 0.0, *candidate["f0"]]
    low = [-radius, -radius, -10.0, *[f - 5 for f in candidate["f0"]]]
    high = [radius, radius, 10.0, *[f + 5 for f in candidate["f0"]]]
    scales = [200.0, 200.0, 1.0, *([1.0] * n)]
    if fit_drift:
        initial += [0.0] * n
        low += [-0.04] * n
        high += [0.04] * n
        scales += [0.01] * n
    if fit_sound_speed:
        initial += [atmosphere.c]
        low += [max(np.linalg.norm(atmosphere.wind) + 1, atmosphere.c - 15)]
        high += [atmosphere.c + 15]
        scales += [3.0]

    def observer(p):
        position = base + np.array([p[0], p[1], 0.0])
        if frame is not None and elevation is not None:
            lat, lon, _ = frame.to_geodetic(position)
            position = frame.to_local([lat, lon, float(elevation(lat, lon))])
        return position

    def residual(p, priors=True):
        atmo = Atmosphere(p[-1], atmosphere.wind) if fit_sound_speed else atmosphere
        result = []
        for i, (track, times, observed, mask) in enumerate(
            zip(tracks, times_list, observations, masks, strict=True)
        ):
            drift = p[3 + n + i] if fit_drift else 0.0
            predicted, _ = predict(track, observer(p), times + p[2], p[3 + i], atmo, drift, references[i])
            # A lost sample is a large error, not a dropped residual.
            result.extend(np.nan_to_num((predicted[mask] - observed[mask]) / 0.3, nan=50.0))
        if priors:
            if fit_drift:
                result.extend(p[3 + n : 3 + 2 * n] / 0.02)
            if fit_sound_speed:
                result.append((p[-1] - atmosphere.c) / 5.0)
        return np.array(result)

    fit = least_squares(
        residual,
        initial,
        bounds=(low, high),
        x_scale=scales,
        loss=loss,
        f_scale=1.0,
        max_nfev=300,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    position = observer(fit.x)
    atmo = Atmosphere(fit.x[-1], atmosphere.wind) if fit_sound_speed else atmosphere
    event_residuals, spectral_scores, coverage = [], [], []
    for i, (event, track, times, observed, mask) in enumerate(
        zip(events, tracks, times_list, observations, masks, strict=True)
    ):
        drift = fit.x[3 + n + i] if fit_drift else 0.0
        frequency, _ = predict(track, position, times + fit.x[2], fit.x[3 + i], atmo, drift, references[i])
        error = np.abs(frequency[mask] - observed[mask])
        event_residuals.append(float(np.median(np.nan_to_num(error, nan=15.0))))
        spectral_scores.append(score_curve(spec, times, frequency, "contrast", event.harmonics))
        coverage.append(float(np.mean(np.isfinite(frequency))))
    # Marginalize clock/frequency/drift/sound speed when describing position geometry.
    # Use data rows only: priors must not pretend to create geometric information.
    data_rows = sum(mask.sum() for mask in masks)
    j = fit.jac[:data_rows]
    nuisance = j[:, 2:]
    projected = j[:, :2] - nuisance @ np.linalg.lstsq(nuisance, j[:, :2], rcond=None)[0]
    eigenvalues = np.linalg.eigvalsh(projected.T @ projected)
    condition = None if eigenvalues[0] <= 1e-12 else float(eigenvalues[-1] / eigenvalues[0])
    at_bounds = bool(
        np.any(np.isclose(fit.x, low, atol=0.001)) or np.any(np.isclose(fit.x, high, atol=0.001))
    )
    return {
        "position": position.tolist(),
        "clock": float(clock + fit.x[2]),
        "flights": [t.id for t in tracks],
        "f0": fit.x[3 : 3 + n].tolist(),
        "frequency_drift_hz_s": fit.x[3 + n : 3 + 2 * n].tolist() if fit_drift else [0.0] * n,
        "source_reference_utc": [clock + event.cpa for event in events],
        "sound_speed_m_s": atmo.c,
        "wind_enu_m_s": list(atmo.wind),
        "score": float(np.mean(spectral_scores)),
        "event_scores": spectral_scores,
        "event_residual_hz": event_residuals,
        "coverage": coverage,
        "ridge_fraction": [float(m.mean()) for m in masks],
        "geometry_condition": condition,
        "geometry_eigenvalues": eigenvalues.tolist(),
        "at_bounds": at_bounds,
        "success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "method": "robust-ridge",
        "uncertainty_note": "Local conditioning only; not a calibrated confidence interval.",
    }
