import numpy as np
import pytest

from dopplertracker.advanced import ModelOptions, fit_projected
from dopplertracker.audio_benchmark import HUM_METHODS, measurement_scene, track_errors
from dopplertracker.demod import Audio, demodulate, remove_lines, stationary_lines
from dopplertracker.signal import from_samples

FS = 2048


def _crossing(duration=200.0, hum=0.3, noise=0.3, seed=0):
    """A tone sweeping 140 -> 110 Hz through a slowly modulated, slightly drifting 120 Hz line."""
    t = np.arange(int(duration * FS)) / FS

    def truth(s):
        return 125 - 15 * np.tanh((np.asarray(s) - duration / 2) / 20)

    phase = 2 * np.pi * np.cumsum(truth(t)) / FS
    line = hum * (1 + 0.2 * np.sin(2 * np.pi * t / 150)) * np.sin(2 * np.pi * (120.03 * t + 5e-5 * t**2))
    rng = np.random.default_rng(seed)
    x = 0.3 * np.sin(phase) + 0.1 * np.sin(2 * phase) + line + rng.normal(0, noise, len(t))
    return Audio(x.astype(np.float32), FS), truth, line


def test_hum_option_defaults_to_masking_and_is_validated():
    assert ModelOptions().hum == "mask"
    assert ModelOptions(ridge="demod", hum="subtract").hum == "subtract"
    with pytest.raises(ValueError):
        ModelOptions(ridge="demod", hum="unknown")
    with pytest.raises(ValueError, match="demodulation"):
        ModelOptions(hum="subtract")


def test_subtraction_removes_a_line_but_keeps_the_crossing_tone():
    audio, truth, line = _crossing()
    lines = stationary_lines(audio, (0.0, 200.0))
    assert np.any(np.abs(lines - 120.0) < 0.2)
    cleaned = remove_lines(audio, lines, (20.0, 180.0))
    assert cleaned.start == pytest.approx(0.0) and len(cleaned.samples) == len(audio.samples)
    # What is removed is the line, not the tone that crosses it at closest approach.
    removed = np.asarray(audio.samples, dtype=float) - cleaned.samples
    inner = slice(30 * FS, 170 * FS)
    assert np.sqrt(np.mean((removed[inner] - line[inner]) ** 2)) < 0.15 * np.sqrt(np.mean(line[inner] ** 2))


def test_removal_keeps_crossing_frames_usable_where_masking_drops_them():
    audio, truth, _ = _crossing()
    lines = stationary_lines(audio, (0.0, 200.0))
    times = np.arange(20.0, 180.0, 0.5)
    near = np.abs(truth(times) - 120.0) < 1.0

    def measure(source, masked):
        return demodulate(source, times, lambda s: truth(s + 0.3) + 0.5, 1, 6.0, 2.0, masked)

    frequency, valid, _, _ = measure(audio, lines)
    assert valid[near].mean() < 0.5
    for method in ("subtract", "notch"):
        frequency, valid, _, _ = measure(remove_lines(audio, lines, (20.0, 180.0), method), ())
        error = np.abs(frequency - truth(times))
        assert valid[near].all()
        assert np.sqrt(np.mean(error[valid] ** 2)) < 0.05


def test_removal_without_lines_returns_the_same_samples_on_the_audio_clock():
    audio, _, _ = _crossing(duration=60.0)
    copy = remove_lines(Audio(audio.samples, FS, 100.0), [], (110.0, 140.0))
    assert copy.start == pytest.approx(100.0)
    np.testing.assert_array_equal(copy.samples, np.asarray(audio.samples, dtype=float))
    with pytest.raises(ValueError):
        remove_lines(audio, [120.0], (10.0, 20.0), "unknown")


def test_hum_crossing_scene_puts_lines_on_the_tone_near_closest_approach():
    scene = measurement_scene(0, "hum_crossing")
    assert scene["hum_hz"][0] == 60.0 and len(scene["hum_hz"]) == 1 + 4 * len(scene["events"])
    for event, frequency, lines in zip(
        scene["events"], scene["observed_frequency_hz"], np.reshape(scene["hum_hz"][1:], (-1, 4))
    ):
        assert lines[1] == pytest.approx(frequency[int(round(event.cpa * FS))])
        assert lines[3] == pytest.approx(2 * frequency[int(round((event.cpa + 2) * FS))])
    # The other development scenes are unchanged by the new scene.
    assert measurement_scene(0, "fast_chirp")["hum_hz"] == [60.0, 120.0, 180.0]


def test_subtraction_recovers_closest_approach_frames_in_a_fit():
    scene = measurement_scene(0, "hum_crossing")
    spec = from_samples(scene["waveform"], FS, 4.0, 0.5, (40, 450)).whiten()
    candidate = {"position": [0.0, 0.0, 0.0], "clock": 0.35, "f0": [e.f0 for e in scene["events"]]}
    audio = Audio(scene["waveform"], FS)
    rows = {}
    for name in ("demod-snr-mask", "demod-snr-subtract"):
        fit = fit_projected(
            spec,
            scene["events"],
            scene["tracks"],
            candidate,
            HUM_METHODS[name],
            audio=audio,
            keep_observations=True,
        )
        rows[name] = (fit, track_errors(scene, fit))
    masked, removed = rows["demod-snr-mask"][1], rows["demod-snr-subtract"][1]
    assert sum(removed["cpa_valid_frames"]) == sum(removed["cpa_frames"]) > sum(masked["cpa_valid_frames"])
    assert removed["track_rms_hz"] < 0.01
    fit = rows["demod-snr-subtract"][0]
    assert fit["options"]["hum"] == "subtract"
    assert len(fit["measurement"]["stationary_lines_hz"]) == len(scene["events"])
    assert np.linalg.norm(np.asarray(fit["position"])[:2] - scene["truth"][:2]) < 1.0
