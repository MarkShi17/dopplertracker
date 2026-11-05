"""Independent, deliberately conservative checks for a proposed acoustic match.

This module does not search or refit a candidate.  It asks whether the frozen
candidate's predicted curves contain a narrow-band ridge more consistently than
nearby, frequency-shifted versions of those same curves.
"""

import numpy as np

from .physics import Atmosphere, predict


def _per_event(value, index, n, default=0.0):
    if value is None:
        return default
    if np.isscalar(value):
        return float(value)
    if len(value) != n:
        raise ValueError("candidate per-event fields must match events")
    return float(value[index])


def _atmosphere(candidate, default):
    c = float(candidate.get("sound_speed_m_s", default.c))
    wind = tuple(candidate.get("wind_enu_m_s", candidate.get("wind", default.wind)))
    return Atmosphere(c, wind)


def _contrast(spec, times, frequency, harmonics, available=None):
    """Local narrow-band energy, with NaNs retained for invalid predictions."""
    output = np.zeros(len(times), dtype=float)
    weight = np.zeros(len(times), dtype=float)
    for harmonic in range(1, harmonics + 1):
        hz = frequency * harmonic
        valid = np.isfinite(hz) & (hz - 3 >= spec.frequencies[0]) & (hz + 3 <= spec.frequencies[-1])
        if available is not None:
            valid &= available[harmonic - 1]
        value = spec.sample(times, hz) - 0.5 * (spec.sample(times, hz - 3) + spec.sample(times, hz + 3))
        output += (1 / np.sqrt(harmonic)) * np.where(valid, value, 0.0)
        weight += valid / np.sqrt(harmonic)
    return np.divide(output, weight, out=np.zeros_like(output), where=weight > 0)


def assess_candidate(spec, events, tracks, candidate, atmosphere=Atmosphere()):
    """Assess frozen candidate curves using nearby frequency-shifted null curves.

    Parameters are the spectrogram, matched ``events`` and ``tracks``, and a
    candidate mapping containing ``position``, ``clock``, and per-event ``f0``.
    Optional candidate fields are ``frequency_drift_hz_s``,
    ``frequency_curvature_hz_s2`` (the coefficient of ``dt**2``),
    ``source_reference_utc``, ``clock_rate_ppm``, ``aircraft_altitude_offset_m``,
    ``sound_speed_m_s``, and ``wind_enu_m_s``.  Event times are
    recording-relative; the shared candidate clock is applied to reception
    times and, absent ``source_reference_utc``, to drift references.

    The returned mapping is JSON-safe.  ``accepted`` is a conservative heuristic
    evidence decision, *not* a calibrated probability or a multiple-search
    p-value: a candidate produced by a search/refinement has already selected
    favourable noise.
    """
    if len(events) != len(tracks) or not events:
        raise ValueError("events and tracks must be non-empty and have equal length")
    if not all(key in candidate for key in ("position", "clock", "f0")):
        raise ValueError("candidate needs position, clock, and f0")
    n = len(events)
    if len(candidate["f0"]) != n:
        raise ValueError("candidate f0 must match events")
    position = np.asarray(candidate["position"], dtype=float)
    clock = float(candidate["clock"])
    if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(clock):
        raise ValueError("candidate position and clock must be finite")
    atmo = _atmosphere(candidate, atmosphere)
    options = candidate.get("options", {})
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("candidate options must be a mapping")
    smoothing = options.get("smoothing", "none")
    if smoothing not in ("none", "spline", "kalman"):
        raise ValueError("candidate options smoothing must be none, spline, or kalman")
    # Advanced fitting smooths raw ADS-B positions before prediction.  Repeat it
    # here so validation evaluates the fitted model, not a subtly different one.
    velocity = options.get("velocity", "differenced")
    if smoothing != "none" or velocity != "differenced":
        from .advanced import prepare_track  # lazy: advanced imports other fitting modules

        tracks = [prepare_track(track, smoothing, velocity) for track in tracks]
    clock_rate = float(candidate.get("clock_rate_ppm", 0.0)) * 1e-6
    altitude_offset = float(candidate.get("aircraft_altitude_offset_m", 0.0))
    if not np.isfinite(clock_rate) or not np.isfinite(altitude_offset):
        raise ValueError("candidate clock rate and altitude offset must be finite")
    observer = position - np.array([0.0, 0.0, altitude_offset])
    # Excluding close offsets avoids measuring a true ridge's finite linewidth.
    shifts = np.r_[-np.arange(6.0, 17.0), np.arange(6.0, 17.0)]
    diagnostics = []
    for i, (event, track) in enumerate(zip(events, tracks, strict=True)):
        times = event.sample_times(spec)
        f0 = _per_event(candidate["f0"], i, n)
        drift = _per_event(candidate.get("frequency_drift_hz_s"), i, n)
        curvature = _per_event(candidate.get("frequency_curvature_hz_s2"), i, n)
        receive = clock + times * (1.0 + clock_rate)
        factor, emission = predict(track, observer, receive, 1.0, atmo)
        reference = _per_event(candidate.get("source_reference_utc"), i, n, clock + event.cpa)
        dt = emission - reference
        source = f0 + drift * dt + curvature * dt * dt
        frequency = factor * source
        # Compare the same available harmonics for the center and every null.
        # An absent second harmonic must not invalidate an observed fundamental,
        # nor may out-of-band nulls artificially improve the center's score.
        available = []
        for harmonic in range(1, event.harmonics + 1):
            lower = (frequency - np.abs(factor) * np.max(np.abs(shifts))) * harmonic - 3
            upper = (frequency + np.abs(factor) * np.max(np.abs(shifts))) * harmonic + 3
            available.append(
                np.isfinite(frequency)
                & (source > 0)
                & (lower >= spec.frequencies[0])
                & (upper <= spec.frequencies[-1])
            )
        available = np.asarray(available)
        valid = np.any(available, axis=0)
        duration = event.window[1] - event.window[0]
        recorded_fraction = min(1.0, (times[-1] - times[0] + np.median(np.diff(times))) / duration)
        # Use original frames: sample_times intentionally thins long windows.
        # Count represented hops rather than treating an internal gap as audio.
        recorded = spec.times[(spec.times >= event.window[0]) & (spec.times <= event.window[1])]
        hop = float(np.median(np.diff(spec.times)))
        represented = float(np.minimum(np.diff(recorded), hop).sum() + hop)
        recorded_fraction *= min(1.0, represented / (recorded[-1] - recorded[0] + hop))
        coverage = float(np.mean(valid) * recorded_fraction)
        center = _contrast(spec, times, frequency, event.harmonics, available)
        nulls = np.stack(
            [
                _contrast(spec, times, frequency + factor * offset, event.harmonics, available)
                for offset in shifts
            ]
        )
        # Framewise robust scale means broad loud periods cannot create support.
        scale = np.median(np.abs(nulls), axis=0)
        floor = max(float(np.median(scale[valid])) * 0.15 if valid.any() else 0.0, 1e-6)
        z = center / np.maximum(scale, floor)
        frame_support = valid & (z > 3.0)
        support = float(np.mean(frame_support[valid])) if valid.any() else 0.0
        center_score = float(np.median(z[valid])) if valid.any() else 0.0
        null_scores = np.array(
            [np.median(row[valid] / np.maximum(scale[valid], floor)) if valid.any() else 0 for row in nulls]
        )
        null_median = float(np.median(null_scores))
        null_max = float(np.max(null_scores))
        # Coverage and per-event gates prevent one excellent event masking a lost one.
        passed = bool(coverage >= 0.75 and support >= 0.35 and center_score > null_max + 1.0)
        diagnostics.append(
            {
                "flight": str(track.id),
                "frames": int(len(times)),
                "valid_frames": int(valid.sum()),
                "coverage": coverage,
                "frame_support": support,
                "center_score": center_score,
                "null_median_score": null_median,
                "null_max_score": null_max,
                "support_over_null": float(center_score - null_median),
                "accepted": passed,
            }
        )
    coverages = [item["coverage"] for item in diagnostics]
    supports = [item["frame_support"] for item in diagnostics]
    return {
        "accepted": bool(all(item["accepted"] for item in diagnostics)),
        "coverage": float(min(coverages)),
        "mean_coverage": float(np.mean(coverages)),
        "mean_frame_support": float(np.mean(supports)),
        "per_event": diagnostics,
        "null_offsets_hz": shifts.tolist(),
        "method": "frozen-curve local-contrast versus frequency-shifted empirical nulls",
        "calibration_note": "Heuristic empirical comparison only; not a calibrated probability or multiple-search p-value. Candidate search/refinement invalidates naive p-values.",
    }
