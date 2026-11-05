"""Waveform-to-STFT benchmark for Doppler localization.

Unlike stress spectra, this exercises phase integration, finite STFT windows,
frequency-bin quantization, whitening, and ridge extraction.  It is synthetic
waveform validation, not a claim about real-world recordings.
"""

import json
import time
from pathlib import Path

import numpy as np

from .advanced import ModelOptions, fit_projected
from .benchmark import analytic_pass
from .detection import Event
from .physics import Trajectory
from .signal import from_samples

SCENARIOS = ("clean_audio", "noisy_audio")
SAMPLE_RATE = 2048
DURATION_SECONDS = 600.0


def audio_scene(seed=0, scenario="clean_audio", sample_rate=SAMPLE_RATE, duration=DURATION_SECONDS):
    """Generate a physical constant-source chirp waveform using analytic_pass.

    Clean/noisy Gaussian sample-noise amplitudes are 0.025/0.12.  Both include
    a 60-Hz stationary hum (0.025/0.08), which whitening should suppress.
    """
    if scenario not in SCENARIOS or sample_rate <= 0 or duration < 30:
        raise ValueError("invalid audio scenario, sample rate, or duration")
    rng = np.random.default_rng(seed)
    sample_count = int(round(sample_rate * duration))
    receive = np.arange(sample_count, dtype=float) / sample_rate
    truth = np.array([135.0, -85.0, 0.0]) + np.r_[rng.uniform(-35, 35, 2), 0.0]
    waveform = np.zeros(sample_count, dtype=np.float64)
    tracks, events, observed = [], [], []
    geometries = [
        ([82.0, 0.0, 0.0], [0.0, 1700.0, 1000.0], 105.0, 111.0),
        ([0.0, 94.0, 0.0], [-1900.0, 0.0, 1300.0], 300.0, 127.0),
        ([67.0, -55.0, 0.0], [1100.0, 1100.0, 850.0], 495.0, 103.0),
    ]
    for i, (velocity, origin, center, f0) in enumerate(geometries):
        frequency, emission, heard_cpa, distance = analytic_pass(
            receive, velocity, origin, center, truth, f0, 343.0
        )
        track_times = np.arange(center - 145, center + 146, 2.0)
        track = Trajectory(
            f"audio-flight-{i}", track_times, np.asarray(origin) + (track_times - center)[:, None] * velocity
        )
        visible = (emission >= track_times[0]) & (emission <= track_times[-1])
        # Integrating instantaneous received frequency is a waveform chirp;
        # a smooth fade avoids artificial broadband edges at visibility bounds.
        phase = 2 * np.pi * np.cumsum(np.where(visible, frequency, 0.0)) / sample_rate
        fade = visible.astype(float) * np.exp(-0.5 * ((receive - heard_cpa) / 85.0) ** 2)
        waveform += 0.26 * fade * np.sin(phase + rng.uniform(0, 2 * np.pi))
        waveform += 0.10 * fade * np.sin(2 * phase + rng.uniform(0, 2 * np.pi))
        tracks.append(track)
        events.append(
            Event(
                (heard_cpa - 55, heard_cpa + 55),
                heard_cpa,
                f0 + rng.normal(0, 0.12),
                float(np.linalg.norm(velocity)),
                distance,
                2,
            )
        )
        observed.append(frequency)
    noise, hum = (0.025, 0.025) if scenario == "clean_audio" else (0.12, 0.08)
    waveform += hum * np.sin(2 * np.pi * 60 * receive + 0.4)
    waveform += 0.45 * hum * np.sin(2 * np.pi * 180 * receive + 0.9)
    waveform += rng.normal(0, noise, sample_count)
    return {
        "waveform": waveform.astype(np.float32),
        "sample_rate": sample_rate,
        "clock": 0.0,
        "truth": truth,
        "tracks": tracks,
        "events": events,
        "observed_frequency_hz": observed,
        "scenario": scenario,
        "seed": seed,
        "noise_amplitude": noise,
        "hum_amplitude": hum,
    }


def run_audio_benchmark(seeds=(100, 101), windows=(1.0, 2.0, 4.0, 8.0), output_dir="outputs/audio-benchmark"):
    """Run waveform controls and write ``results.json``; failures remain trials."""
    seeds, windows = tuple(int(s) for s in seeds), tuple(float(w) for w in windows)
    if not seeds or not windows or any(not np.isfinite(w) or w <= 0 for w in windows):
        raise ValueError("seeds and positive STFT windows are required")
    rows = []
    for scenario in SCENARIOS:
        for seed in seeds:
            scene = audio_scene(seed, scenario)
            candidate = {
                "position": [0.0, 0.0, 0.0],
                "clock": 0.35,
                "f0": [event.f0 for event in scene["events"]],
                "flights": [track.id for track in scene["tracks"]],
            }
            for window in windows:
                spec = from_samples(
                    scene["waveform"], scene["sample_rate"], window, hop_seconds=0.5, band=(40, 350)
                ).whiten()
                for degree in (0, 1):
                    started = time.perf_counter()
                    row = {
                        "scenario": scenario,
                        "seed": seed,
                        "window_seconds": window,
                        "frequency_bin_hz": 1.0 / window,
                        "degree": degree,
                        "spectrogram_shape": list(spec.power.shape),
                        "spectrogram_bytes": int(spec.power.nbytes),
                        "success": False,
                        "error": None,
                        "location_error_m": None,
                        "clock_error_s": None,
                        "ridge_residual_hz": None,
                    }
                    try:
                        fit = fit_projected(
                            spec,
                            scene["events"],
                            scene["tracks"],
                            candidate,
                            ModelOptions(degree=degree, max_nfev=55, radius=900.0),
                        )
                        row.update(
                            success=bool(fit["success"]),
                            location_error_m=float(
                                np.linalg.norm(np.asarray(fit["position"])[:2] - scene["truth"][:2])
                            ),
                            clock_error_s=float(fit["clock"]),
                            ridge_residual_hz=float(np.mean(fit["event_residual_hz"])),
                            optimizer_message=fit["optimizer_message"],
                            nfev=int(fit["nfev"]),
                        )
                    except Exception as exc:  # failures are outcomes, not silently dropped
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    row["seconds"] = time.perf_counter() - started
                    rows.append(row)
    report = {
        "validation": "synthetic waveform; not real recording validation",
        "generator": "independent analytic_pass Doppler followed by phase-integrated time-domain chirps",
        "sample_rate_hz": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "noise_documentation": {
            "clean_audio": {"gaussian_amplitude": 0.025, "hum_amplitude": 0.025},
            "noisy_audio": {"gaussian_amplitude": 0.12, "hum_amplitude": 0.08},
        },
        "seeds": list(seeds),
        "windows": list(windows),
        "trials": rows,
        "attempted_trials": len(rows),
        "failed_trials": sum(not row["success"] for row in rows),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
