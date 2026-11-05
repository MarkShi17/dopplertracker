import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from dopplertracker.detection import (
    Event,
    FlyoverOptions,
    detect_flyovers,
    fit_flyover,
    load_events,
    save_events,
    suppress_tones,
)
from dopplertracker.physics import ideal_curve
from dopplertracker.scoring import ridge_contrast, slope_weights
from dopplertracker.signal import Spectrogram

TIMES = np.arange(0.0, 320.0, 0.5)
FREQUENCIES = np.arange(50.0, 320.0, 0.25)
PASS = dict(f0=110.0, speed=95.0, distance=1600.0, cpa=170.0)


def scene(seed=0, flyover=True, tone=True, amplitude=3.0):
    """Exponential noise, optionally one two-harmonic pass and an 80 s steady tone."""
    rng = np.random.default_rng(seed)
    power = rng.exponential(0.2, (len(FREQUENCIES), len(TIMES)))
    if flyover:
        ridge = ideal_curve(TIMES, PASS["f0"], PASS["speed"], PASS["distance"], PASS["cpa"])
        for harmonic in (1, 2):
            power += (amplitude / harmonic) * np.exp(
                -0.5 * ((FREQUENCIES[:, None] - harmonic * ridge[None, :]) / 0.45) ** 2
            )
    if tone:
        # Present for a quarter of the recording, so whitening alone keeps it.
        on = (TIMES > 120) & (TIMES < 200)
        power += 6.0 * np.exp(-0.5 * ((FREQUENCIES[:, None] - 150.0) / 0.4) ** 2) * on[None, :]
    return Spectrogram(TIMES, FREQUENCIES, power).whiten()


@pytest.fixture(scope="module")
def detected():
    report = {}
    events = detect_flyovers(scene(), report=report)
    return events, report


def test_ridge_contrast_cancels_broadband_energy_and_rewards_a_ridge():
    t = np.arange(20.0)
    f = np.arange(60.0, 200.0, 0.25)
    power = np.full((len(f), len(t)), 4.0) + 3 * np.exp(-0.5 * ((f[:, None] - 120) / 0.3) ** 2)
    spec = Spectrogram(t, f, power)
    on_ridge = ridge_contrast(spec, t, np.full(20, 120.0), clip=(-10, 10))
    off_ridge = ridge_contrast(spec, t, np.full(20, 90.0), clip=(-10, 10))
    np.testing.assert_allclose(on_ridge, 3.0, atol=1e-6)
    np.testing.assert_allclose(off_ridge, 0.0, atol=1e-6)
    # Outside the spectrogram there is no evidence, not a penalty.
    assert np.all(ridge_contrast(spec, t, np.full(20, 400.0)) == 0)


def test_flat_curves_carry_no_slope_weight():
    t = np.arange(0.0, 60.0)
    assert np.all(slope_weights(t, np.full(60, 100.0), 100.0) == 0)
    steep = slope_weights(t, 100.0 - t, 100.0)
    assert np.all(steep == 1.0)


def test_tone_suppression_removes_long_tones_but_keeps_a_sweep():
    spec = scene()
    cleaned = suppress_tones(spec, 60.0)
    tone_row = np.argmin(np.abs(FREQUENCIES - 150.0))
    during = (TIMES > 135) & (TIMES < 185)
    assert np.median(spec.power[tone_row, during]) > 1
    assert np.median(cleaned.power[tone_row, during]) < 0.3
    ridge = ideal_curve(TIMES, PASS["f0"], PASS["speed"], PASS["distance"], PASS["cpa"])
    steep = np.abs(TIMES - PASS["cpa"]) < 15
    assert np.median(cleaned.sample(TIMES[steep], ridge[steep])) > 1


def test_detects_the_pass_not_the_tone_or_its_harmonic(detected):
    events, report = detected
    assert len(events) == 1
    event = events[0]
    assert event.cpa == pytest.approx(PASS["cpa"], abs=1.5)
    assert event.f0 == pytest.approx(PASS["f0"], rel=0.01)
    assert event.z >= FlyoverOptions().min_z
    assert event.speed_range[0] <= PASS["speed"] <= event.speed_range[1]
    assert event.distance_range[0] <= PASS["distance"] <= event.distance_range[1]
    assert event.cpa_range[0] <= event.cpa <= event.cpa_range[1]
    assert event.window[0] <= event.cpa <= event.window[1]
    assert all(np.isfinite(c["z"]) for c in report["candidates"])
    assert sum(c["accepted"] for c in report["candidates"]) == 1


def test_detection_is_deterministic(detected):
    again = detect_flyovers(scene())
    assert [e.cpa for e in again] == [e.cpa for e in detected[0]]
    assert [e.z for e in again] == [e.z for e in detected[0]]


@pytest.mark.parametrize("tone", [False, True])
def test_detector_abstains_without_a_flyover(tone):
    report = {}
    assert detect_flyovers(scene(seed=3, flyover=False, tone=tone), report=report) == []
    assert all(not c["accepted"] for c in report["candidates"])


def test_silent_spectrogram_abstains():
    spec = Spectrogram(np.arange(200.0), np.arange(50.0, 250.0), np.zeros((200, 200)))
    assert detect_flyovers(spec) == []


def test_time_reversed_pass_is_a_null_not_a_detection():
    spec = scene(tone=False)
    flipped = Spectrogram(spec.times, spec.frequencies, spec.power[:, ::-1])
    assert detect_flyovers(flipped) == []


def test_fit_flyover_never_abstains_and_respects_window():
    spec = scene(seed=5, flyover=False, tone=False)
    event = fit_flyover(spec, (100.0, 200.0), ranges=False)
    assert 100.0 <= event.cpa <= 200.0
    assert np.isfinite(event.z)
    fitted = fit_flyover(scene(), (150.0, 190.0), ranges=False)
    assert fitted.cpa == pytest.approx(PASS["cpa"], abs=1.5)


def test_options_are_validated():
    with pytest.raises(ValueError):
        FlyoverOptions(band=(300.0, 60.0))
    with pytest.raises(ValueError):
        replace(FlyoverOptions(), min_sweep=1.5)
    with pytest.raises(ValueError):
        detect_flyovers(scene(), top=0)


def test_event_json_roundtrip_keeps_calibration_and_reads_old_files(tmp_path, detected):
    path = tmp_path / "events.json"
    save_events(detected[0], path)
    loaded = load_events([str(path)])
    assert loaded[0].z == pytest.approx(detected[0][0].z)
    assert tuple(loaded[0].cpa_range) == pytest.approx(detected[0][0].cpa_range)
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"window": [0, 10], "cpa": 5, "f0": 100, "speed": 80, "distance": 900}))
    assert load_events([str(old)])[0].z is None
    with pytest.raises(ValueError):
        Event((0, 10), 5, 100, 80, 900, cpa_range=(6, 7))


def test_cli_calibrated_detect_writes_z_scored_events(tmp_path):
    scene().save(tmp_path / "spec.npz")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dopplertracker",
            "detect",
            str(tmp_path / "spec.npz"),
            "--raw-score",
            "--detector",
            "calibrated",
            "--out",
            str(tmp_path / "events.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    events = json.loads((tmp_path / "events.json").read_text())
    assert len(events) == 1 and events[0]["z"] > 0
    assert "z=" in result.stdout
