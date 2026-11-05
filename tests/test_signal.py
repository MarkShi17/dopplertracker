import numpy as np
import pytest

from dopplertracker.detection import detect, fit_event
from dopplertracker.physics import ideal_curve
from dopplertracker.scoring import score_curve
from dopplertracker.signal import Spectrogram, from_samples


def test_spectrogram_tracks_a_known_audio_tone_and_roundtrips(tmp_path):
    fs = 2048
    t = np.arange(fs * 12) / fs
    spec = from_samples(np.sin(2 * np.pi * 123.5 * t), fs, window_seconds=4, hop_seconds=0.5)
    peak = spec.frequencies[np.argmax(spec.power.mean(axis=1))]
    assert peak == pytest.approx(123.5, abs=0.3)
    path = tmp_path / "spec.npz"
    spec.save(path)
    np.testing.assert_array_equal(Spectrogram.load(path).power, spec.power)


def test_missing_curve_coverage_is_not_rewarded():
    t, f = np.arange(10.0), np.arange(90.0, 111.0)
    spec = Spectrogram(t, f, np.ones((len(f), len(t))))
    full = score_curve(spec, t, np.full(10, 100.0), method="brightness")
    partial = score_curve(spec, t, np.r_[np.full(5, 100.0), np.full(5, np.nan)], method="brightness")
    assert partial == pytest.approx(full / 2)


def test_contrast_rejects_broadband_noise():
    t, f = np.arange(20.0), np.arange(80.0, 160.0, 0.25)
    power = np.full((len(f), len(t)), 4.0)
    power += 3 * np.exp(-0.5 * ((f[:, None] - 120) / 0.3) ** 2)
    spec = Spectrogram(t, f, power)
    assert score_curve(spec, t, np.full(20, 120.0), method="contrast") > 2
    assert abs(score_curve(spec, t, np.full(20, 100.0), method="contrast")) < 0.01


def test_ideal_curve_fit_recovers_event_timing():
    t, f = np.arange(0.0, 120.0, 1.0), np.arange(60.0, 180.0, 0.25)
    ridge = ideal_curve(t, 110.0, 85.0, 1800.0, 60.0)
    spec = Spectrogram(t, f, 8 * np.exp(-0.5 * ((f[:, None] - ridge) / 0.6) ** 2))
    event = fit_event(spec, (0, 119), (95, 125), seed=12, iterations=45)
    assert event.cpa == pytest.approx(60, abs=2)
    assert event.f0 == pytest.approx(110, abs=2)


def test_spectrogram_rejects_bad_axes():
    with pytest.raises(ValueError):
        Spectrogram(np.array([1.0, 0.0]), np.arange(10.0), np.zeros((10, 2)))


def test_silent_spectrogram_has_no_detected_flyovers():
    spec = Spectrogram(np.arange(130.0), np.arange(50.0, 200.0), np.zeros((150, 130)))
    assert detect(spec, top=2) == []


@pytest.mark.parametrize("gain_squared", [1e-12, 1e12])
def test_whitening_preserves_evidence_when_only_recording_gain_changes(gain_squared):
    rng = np.random.default_rng(31)
    power = rng.lognormal(-2, 0.8, (10, 30))
    power[4, 10:20] *= 20
    normal = Spectrogram(np.arange(30.0), np.arange(10.0), power).whiten()
    rescaled = Spectrogram(np.arange(30.0), np.arange(10.0), power * gain_squared).whiten()
    np.testing.assert_allclose(normal.power, rescaled.power, atol=1e-5)
