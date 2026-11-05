import shutil

import numpy as np
import pytest
from scipy.io import wavfile

from dopplertracker.advanced import ModelOptions, _combine_harmonics, fit_projected
from dopplertracker.audio_benchmark import (
    audio_scene,
    measurement_scene,
    run_measurement_benchmark,
    track_errors,
)
from dopplertracker.demod import Audio, demodulate, stationary_lines
from dopplertracker.signal import decode_samples, from_samples
from dopplertracker.stress_scenes import stress_scene

FS = 2048


def _sweep(duration=120.0, hum=0.0, noise=0.3, seed=0):
    t = np.arange(int(duration * FS)) / FS
    frequency = 125 - 15 * np.tanh((t - duration / 2) / 5)
    phase = 2 * np.pi * np.cumsum(frequency) / FS
    rng = np.random.default_rng(seed)
    x = 0.3 * np.sin(phase) + 0.1 * np.sin(2 * phase) + hum * np.sin(2 * np.pi * 120 * t)
    x += rng.normal(0, noise, len(t))
    return Audio(x.astype(np.float32), FS), lambda s: 125 - 15 * np.tanh((s - duration / 2) / 5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_frames": 2},
        {"reextract": -1},
        {"noise": "unknown"},
        {"harmonic_mode": "unknown"},
        {"max_harmonic": 5},
        {"demod_window": 0.1},
        {"ridge": "viterbi", "harmonic_mode": "ratio"},
        {"ridge": "viterbi", "noise": "snr"},
    ],
)
def test_measurement_options_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ModelOptions(**kwargs)


def test_measurement_defaults_preserve_original_options():
    options = ModelOptions()
    assert (options.max_frames, options.reextract, options.noise, options.harmonic_mode) == (
        140,
        0,
        "fixed",
        "sum",
    )
    assert options.max_harmonic == 0


def test_demodulation_needs_audio():
    scene = stress_scene("clean", 0)
    with pytest.raises(ValueError, match="raw audio"):
        fit_projected(
            scene["spectrogram"],
            scene["events"],
            scene["tracks"],
            scene["candidate"],
            ModelOptions(ridge="demod"),
        )


@pytest.mark.parametrize("offset", [0.0, 3.0])
def test_demodulation_tracks_fast_sweep_and_calibrates_uncertainty(offset):
    audio, truth = _sweep()
    times = np.arange(10.0, 110.0, 0.5)
    for harmonic in (1, 2):
        frequency, valid, sigma, snr = demodulate(
            audio, times, lambda s: truth(s + 0.4) + offset, harmonic, 8.0 * harmonic, 2.0
        )
        error = (frequency - truth(times))[valid]
        assert valid.mean() > 0.9
        assert np.sqrt(np.mean(error**2)) < 0.03
        # The Cramér-Rao uncertainty is the right size for white noise.
        assert 0.5 < np.sqrt(np.mean(error**2)) / np.median(sigma[valid]) < 2.0
        assert np.all(snr[valid] >= 8.0)


def test_stationary_lines_find_hum_but_not_a_passing_tone():
    t = np.arange(600 * FS) / FS
    tone = 0.3 * np.sin(2 * np.pi * np.cumsum(150 - 0.05 * t) / FS)  # 150 -> 120 Hz
    hum = 0.1 * np.sin(2 * np.pi * 100 * t)
    noise = np.random.default_rng(0).normal(0, 0.3, len(t))
    lines = stationary_lines(Audio((tone + hum + noise).astype(np.float32), FS), (0.0, 600.0))
    assert np.any(np.abs(lines - 100.0) < 0.2)
    assert not np.any((lines > 118) & (lines < 152))


def test_demodulation_does_not_lock_onto_a_crossing_hum_line():
    # The tone sweeps 140 -> 110 Hz through a louder 120 Hz hum line.
    audio, truth = _sweep(hum=0.6, noise=0.1)
    times = np.arange(10.0, 110.0, 0.5)
    frequency, valid, _, _ = demodulate(audio, times, lambda s: truth(s) + 2.0, 1, 8.0, 2.0, [120.0])
    error = np.abs(frequency - truth(times))[valid]
    assert valid.mean() > 0.7
    assert error.max() < 0.1
    near = np.abs(truth(times) - 120.0) < 0.5
    assert not valid[near].any()


def test_harmonic_combination_weights_precision_and_drops_disagreement():
    ridges = np.array([[100.0, 100.0, 100.0], [100.2, 100.2, 104.0]])
    valids = np.ones((2, 3), dtype=bool)
    sigmas = np.array([[0.2, 0.2, 0.2], [0.1, 0.1, 0.1]])
    ridge, valid, sigma = _combine_harmonics(ridges, valids, sigmas)
    np.testing.assert_allclose(ridge[:2], 100.16)
    np.testing.assert_allclose(sigma[:2], 1 / np.sqrt(25 + 100))
    # The more precise harmonic is the reference; the disagreeing fundamental is dropped.
    assert ridge[2] == pytest.approx(104.0) and sigma[2] == pytest.approx(0.1)
    assert valid.all()


def test_all_frames_and_observations_are_reported_without_changing_the_fit():
    scene = stress_scene("clean", 0)
    args = (scene["spectrogram"], scene["events"], scene["tracks"], scene["candidate"])
    plain = fit_projected(*args, ModelOptions(degree=1))
    kept = fit_projected(*args, ModelOptions(degree=1), keep_observations=True)
    assert plain["position"] == kept["position"] and "measurement" not in plain
    every = fit_projected(*args, ModelOptions(degree=1, max_frames=None), keep_observations=True)
    times = scene["spectrogram"].times
    window = scene["events"][0].window
    assert len(every["observations"][0]["times"]) == np.sum((times >= window[0]) & (times <= window[1]))
    assert len(kept["observations"][0]["times"]) <= 140


@pytest.mark.parametrize("noise", ["robust", "snr"])
def test_noise_models_estimate_event_scales_and_still_locate(noise):
    scene = stress_scene("clean", 0)
    result = fit_projected(
        scene["spectrogram"],
        scene["events"],
        scene["tracks"],
        scene["candidate"],
        ModelOptions(degree=1, noise=noise, harmonic_mode="ratio"),
    )
    scales = result["measurement"]["noise_scale_and_floor_hz"]
    assert len(scales) == len(scene["events"]) and all(s > 0 and f >= 0 for s, f in scales)
    assert np.linalg.norm(np.array(result["position"])[:2] - scene["truth"][:2]) < 20


def test_reextraction_records_passes_and_converges():
    scene = stress_scene("clean", 0)
    result = fit_projected(
        scene["spectrogram"],
        scene["events"],
        scene["tracks"],
        scene["candidate"],
        ModelOptions(degree=1, reextract=4),
    )
    history = np.array(result["measurement"]["position_history"])
    assert 2 <= result["measurement"]["passes"] == len(history) <= 5
    if len(history) < 5:
        assert np.linalg.norm(history[-1, :2] - history[-2, :2]) < 1.0
    assert np.linalg.norm(history[-1, :2] - scene["truth"][:2]) < 20


@pytest.fixture(scope="module")
def chirp_scene():
    scene = measurement_scene(0, "fast_chirp")
    spec = from_samples(scene["waveform"], FS, 4.0, 0.5, (40, 450)).whiten()
    candidate = {"position": [0.0, 0.0, 0.0], "clock": 0.35, "f0": [e.f0 for e in scene["events"]]}
    return scene, spec, candidate


def test_demodulation_beats_stft_ridge_on_a_fast_close_pass(chirp_scene):
    scene, spec, candidate = chirp_scene
    audio = Audio(scene["waveform"], FS)
    rows = {}
    for name, options in [
        ("stft", ModelOptions(degree=1, radius=900.0)),
        ("demod", ModelOptions(degree=1, radius=900.0, ridge="demod", reextract=2)),
    ]:
        fit = fit_projected(
            spec, scene["events"], scene["tracks"], candidate, options, audio=audio, keep_observations=True
        )
        error = np.linalg.norm(np.asarray(fit["position"])[:2] - scene["truth"][:2])
        rows[name] = (error, track_errors(scene, fit))
    assert rows["demod"][1]["track_cpa_rms_hz"] < 0.1 * rows["stft"][1]["track_cpa_rms_hz"]
    assert rows["demod"][0] < 1.0 < rows["stft"][0]


def test_absent_extra_harmonic_is_ignored():
    # audio_scene has harmonics 1-2 only; asking for 3 must not change the measurement.
    scene = audio_scene(0, "clean_audio", duration=300.0)
    spec = from_samples(scene["waveform"], FS, 4.0, 0.5, (40, 450)).whiten()
    candidate = {"position": [0.0, 0.0, 0.0], "clock": 0.35, "f0": [e.f0 for e in scene["events"][:2]]}
    args = (spec, scene["events"][:2], scene["tracks"][:2], candidate)
    two = fit_projected(*args, ModelOptions(degree=1, harmonic_mode="ratio"), keep_observations=True)
    three = fit_projected(
        *args, ModelOptions(degree=1, harmonic_mode="ratio", max_harmonic=3), keep_observations=True
    )
    assert two["observations"][0]["ridge_hz"] == three["observations"][0]["ridge_hz"]


def test_measurement_scenes_use_their_own_random_stream():
    reference = audio_scene(1, "clean_audio", duration=60.0)
    assert "hum_hz" not in reference
    scene = measurement_scene(1, "low_snr_audio", duration=60.0)
    assert scene["noise_amplitude"] == 3.0
    np.testing.assert_array_equal(
        audio_scene(1, "clean_audio", duration=60.0)["waveform"], reference["waveform"]
    )


def test_measurement_benchmark_smoke(tmp_path):
    report = run_measurement_benchmark(
        seeds=(0,), scenarios=("fast_chirp",), methods=("stft-peak",), output_dir=tmp_path, progress=None
    )
    assert len(report["trials"]) == 1 and report["trials"][0]["error"] is None
    assert (tmp_path / "results.json").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_decode_samples_returns_aligned_mono_samples(tmp_path):
    t = np.arange(4 * FS) / FS
    tone = (0.5 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    wavfile.write(tmp_path / "tone.wav", FS, tone)
    samples = decode_samples(tmp_path / "tone.wav", FS)
    assert samples.dtype == np.float32 and len(samples) == len(tone)
    np.testing.assert_allclose(samples, tone, atol=1e-4)
