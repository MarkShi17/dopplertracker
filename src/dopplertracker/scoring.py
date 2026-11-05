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
