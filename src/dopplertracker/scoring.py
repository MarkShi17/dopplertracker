"""Full-window spectral evidence; no reward for missing track coverage."""

import numpy as np

from .physics import Atmosphere, predict


def score_curve(spec, times, frequency, method="brightness", harmonics=1):
    if method not in ("brightness", "contrast") or harmonics < 1:
        raise ValueError("method must be brightness or contrast; harmonics must be positive")
    frequency = np.asarray(frequency)
    values = np.zeros_like(frequency, dtype=float)
    weight_sum = 0.0
    for h in range(1, harmonics + 1):
        weight = 1 / np.sqrt(h)
        center = spec.sample(times, frequency * h)
        if method == "contrast":
            background = 0.5 * (spec.sample(times, frequency * h - 3) + spec.sample(times, frequency * h + 3))
            # Positive local evidence retains zero as the missing-data baseline.
            center = np.maximum(center - background, 0)
        values += weight * center
        weight_sum += weight
    return float(np.mean(values) / weight_sum)


def _uniform(axis):
    steps = np.diff(axis)
    return bool(np.allclose(steps, steps[0], rtol=1e-6, atol=0))


def sample_bilinear(spec, times, frequencies):
    """``spec.sample`` (zero outside the axes), with a fast path for uniform axes.

    Curve scoring evaluates thousands of curves; on uniform STFT grids direct
    index arithmetic avoids the general interpolation overhead.
    """
    cached = getattr(spec, "_uniform_axes", None)
    if cached is None:
        cached = _uniform(spec.times) and _uniform(spec.frequencies)
        spec._uniform_axes = cached
    if not cached:
        return spec.sample(times, frequencies)
    times, frequencies = np.broadcast_arrays(np.asarray(times, float), np.asarray(frequencies, float))
    ti = (times - spec.times[0]) / (spec.times[1] - spec.times[0])
    fi = (frequencies - spec.frequencies[0]) / (spec.frequencies[1] - spec.frequencies[0])
    nf, nt = spec.power.shape
    with np.errstate(invalid="ignore"):
        valid = (ti >= 0) & (ti <= nt - 1) & (fi >= 0) & (fi <= nf - 1)
    ti = np.where(valid, ti, 0.0)
    fi = np.where(valid, fi, 0.0)
    t0 = np.minimum(ti.astype(int), nt - 2)
    f0 = np.minimum(fi.astype(int), nf - 2)
    a, b = ti - t0, fi - f0
    p = spec.power
    value = (1 - b) * ((1 - a) * p[f0, t0] + a * p[f0, t0 + 1]) + b * (
        (1 - a) * p[f0 + 1, t0] + a * p[f0 + 1, t0 + 1]
    )
    return np.where(valid, value, 0.0)


def ridge_contrast(spec, times, frequency, harmonics=1, sidebands=(0.03, 0.06), clip=(-1.0, 4.0)):
    """Per-sample narrow-band contrast against sidebands at ``frequency * (1 +/- s)``.

    Relative sidebands scale with the ridge, so broadband energy cancels at every
    harmonic.  Clipping bounds both a crossing tone's reward and a sideband tone's
    penalty.  Harmonics are weighted 1/sqrt(h) and renormalised over those whose
    ridge and sidebands lie inside the spectrogram; samples with none score zero.
    """
    times = np.asarray(times, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    sidebands = np.asarray(sidebands, dtype=float)
    if harmonics < 1 or not len(sidebands) or np.any(sidebands <= 0) or not clip[0] < clip[1]:
        raise ValueError("ridge contrast needs positive harmonics/sidebands and an increasing clip")
    # Axes: harmonic, (ridge, lower sidebands, upper sidebands), sample.
    harmonic = np.arange(1, harmonics + 1, dtype=float)[:, None, None]
    scale = np.r_[1.0, 1.0 - sidebands, 1.0 + sidebands][None, :, None]
    hz = harmonic * scale * frequency[None, None, :]
    with np.errstate(invalid="ignore"):
        valid = (
            np.isfinite(hz).all(axis=1)
            & (hz.min(axis=1) >= spec.frequencies[0])
            & (hz.max(axis=1) <= spec.frequencies[-1])
        )
    values = sample_bilinear(spec, np.broadcast_to(times, hz.shape), hz)
    contrast = np.clip(values[:, 0] - values[:, 1:].mean(axis=1), *clip)
    weights = 1 / np.sqrt(harmonic[:, :, 0])
    total = (np.where(valid, contrast, 0.0) * weights).sum(axis=0)
    weight = (valid * weights).sum(axis=0)
    return np.divide(total, weight, out=np.zeros_like(total), where=weight > 0)


def slope_weights(times, frequency, f0, slope_scale=0.002):
    """Weight in [0, 1]: full once a curve sweeps ``slope_scale * f0`` Hz per second.

    Flat portions of a Doppler curve are indistinguishable from stationary hum, so
    they carry little evidence; undefined samples carry none.
    """
    frequency = np.asarray(frequency, dtype=float)
    if len(frequency) < 2 or f0 <= 0 or slope_scale <= 0:
        raise ValueError("slope weights need two samples and positive f0 and slope scale")
    times = np.asarray(times, dtype=float)
    slope = np.empty(len(frequency))
    slope[1:-1] = (frequency[2:] - frequency[:-2]) / (times[2:] - times[:-2])
    slope[0] = (frequency[1] - frequency[0]) / (times[1] - times[0])
    slope[-1] = (frequency[-1] - frequency[-2]) / (times[-1] - times[-2])
    weight = np.minimum(1.0, np.abs(slope) / (slope_scale * f0))
    return np.where(np.isfinite(weight), weight, 0.0)


def joint_score(
    spec,
    events,
    tracks,
    observer,
    clock,
    method="brightness",
    atmosphere=Atmosphere(),
    shift_radius=3.0,
    shift_steps=7,
    frequency_radius=2.0,
    frequency_steps=9,
):
    """clock is UTC recording start. ONE shift is shared by every event."""
    best = (-np.inf, clock, [], [])
    for shift in np.linspace(-shift_radius, shift_radius, shift_steps):
        scores, fitted = [], []
        for event, track in zip(events, tracks, strict=True):
            times = event.sample_times(spec)
            factor, _ = predict(track, observer, times + clock + shift, atmosphere=atmosphere)
            options = np.linspace(event.f0 - frequency_radius, event.f0 + frequency_radius, frequency_steps)
            values = [score_curve(spec, times, factor * f0, method, event.harmonics) for f0 in options]
            idx = int(np.argmax(values))
            scores.append(values[idx])
            fitted.append(float(options[idx]))
        total = float(np.mean(scores))
        if total > best[0]:
            best = total, float(clock + shift), fitted, scores
    return best
