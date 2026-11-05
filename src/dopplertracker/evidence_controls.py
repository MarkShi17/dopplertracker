"""Frozen held-out controls for the heuristic evidence gate.

These are synthetic controls, not an estimate of performance on real recordings.
They intentionally do not alter evidence thresholds.
"""

import numpy as np

from .detection import Event
from .evidence import assess_candidate
from .physics import Trajectory, closest_approaches, predict
from .signal import Spectrogram


def independent_tone_scene(seed=0):
    """Separate known-model fixture for evidence checks, not independent physics.

    Uses the production predictor intentionally. Independent forward generation
    is exercised by stress_scenes and audio_benchmark instead.
    """
    rng = np.random.default_rng(seed)
    times, frequencies = np.arange(0.0, 260.0), np.arange(45.0, 330.0, 0.25)
    truth = np.array([90.0, -55.0, 0.0])
    power = rng.exponential(0.08, (len(frequencies), len(times)))
    tracks, events = [], []
    for i, (center, velocity, origin, f0) in enumerate(
        [
            (80.0, [85.0, 0, 0], [0, 1500, 1000], 105.0),
            (180.0, [0, 90.0, 0], [-1600, 0, 1100], 125.0),
        ]
    ):
        track_times = np.arange(center - 130, center + 131, 2.0)
        track = Trajectory(str(i), track_times, np.array(origin) + (track_times - center)[:, None] * velocity)
        heard, distance, speed = closest_approaches(track, truth)[0]
        event = Event((heard - 45, heard + 45), heard, f0, speed, distance, 2)
        curve, _ = predict(track, truth, times, f0)
        for harmonic in (1, 2):
            power += np.nan_to_num(
                6 / harmonic * np.exp(-0.5 * ((frequencies[:, None] - harmonic * curve) / 0.38) ** 2)
            )
        tracks.append(track)
        events.append(event)
    return {
        "spectrogram": Spectrogram(times, frequencies, power.astype(np.float32)),
        "tracks": tracks,
        "events": events,
        "truth": truth,
    }


def run_evidence_controls(seeds=range(100, 120)):
    """Return JSON-safe held-out acceptance counts without retuning thresholds."""
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one control seed is required")
    counts = {name: 0 for name in ("true", "noise", "broadband", "wrong_position", "wrong_clock")}
    for seed in seeds:
        scene = independent_tone_scene(seed)
        spec, events, tracks, truth = scene["spectrogram"], scene["events"], scene["tracks"], scene["truth"]
        candidate = {"position": truth.tolist(), "clock": 0.0, "f0": [105.0, 125.0]}
        rng = np.random.default_rng(seed + 5000)
        noise = Spectrogram(
            spec.times, spec.frequencies, rng.exponential(0.08, spec.power.shape).astype(np.float32)
        )
        broadband = Spectrogram(
            noise.times,
            noise.frequencies,
            (noise.power + rng.uniform(0, 20, len(noise.times))[None, :]).astype(np.float32),
        )
        variants = {
            "true": (spec, candidate),
            "noise": (noise, candidate),
            "broadband": (broadband, candidate),
            "wrong_position": (spec, {**candidate, "position": (truth + [1200, -900, 0]).tolist()}),
            "wrong_clock": (spec, {**candidate, "clock": 15.0}),
        }
        for name, (control_spec, control_candidate) in variants.items():
            counts[name] += int(assess_candidate(control_spec, events, tracks, control_candidate)["accepted"])
    trials = len(seeds)
    return {
        "trials": trials,
        "seeds": list(seeds),
        "acceptance": {
            name: {"accepted": count, "rejected": trials - count, "rate": count / trials}
            for name, count in counts.items()
        },
        "threshold_policy": "Frozen evidence thresholds; controls do not tune them.",
        "limitation": "Synthetic direct-tone controls; not a calibrated real-world false-match rate.",
    }
