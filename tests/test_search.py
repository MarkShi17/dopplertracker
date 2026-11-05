import numpy as np
import pytest

from dopplertracker.detection import Event
from dopplertracker.physics import Trajectory, closest_approaches, predict
from dopplertracker.refine import refine_candidate
from dopplertracker.search import search
from dopplertracker.signal import Spectrogram


def scene():
    truth = np.array([135.0, -85.0, 0.0])
    tracks, events = [], []
    times, frequencies = np.arange(0.0, 400.0, 1.0), np.arange(60.0, 260.0, 0.25)
    power = np.full((len(frequencies), len(times)), 0.1)
    for i, (center, velocity, position, f0) in enumerate(
        [
            (100.0, [80.0, 0, 0], [0.0, 1500.0, 1000.0], 110.0),
            (300.0, [0.0, 95.0, 0], [-1800.0, 0.0, 1400.0], 130.0),
        ]
    ):
        t = np.arange(center - 110, center + 111, 2.0)
        p = np.array(position) + (t - center)[:, None] * velocity
        track = Trajectory(str(i), t, p)
        cpa, distance, speed = closest_approaches(track, truth)[0]
        event = Event((cpa - 55, cpa + 55), cpa, f0, speed, distance, 1)
        f, _ = predict(track, truth, times, f0)
        delta = frequencies[:, None] - f
        power += np.nan_to_num(8 * np.exp(-0.5 * (delta / 0.35) ** 2))
        tracks.append(track)
        events.append(event)
    return truth, tracks, events, Spectrogram(times, frequencies, power)


def test_search_unknown_clock_ranks_correct_location():
    truth, tracks, events, spec = scene()
    points = np.array([truth, truth + [900, 800, 0], truth + [-700, -800, 0]])
    results = search(spec, events, tracks, points, shift_radius=1, shift_steps=3)
    assert results
    np.testing.assert_allclose(results[0]["position"], truth)
    assert results[0]["clock"] == pytest.approx(0, abs=0.01)


def test_single_event_requires_recording_time():
    truth, tracks, events, spec = scene()
    with pytest.raises(ValueError, match="clock"):
        search(spec, events[:1], tracks, np.array([truth]))


def test_robust_continuous_refinement_beats_grid_rounding():
    truth, tracks, events, spec = scene()
    initial = {"position": [200.0, -200.0, 0.0], "clock": 0.0, "f0": [110.0, 130.0], "flights": ["0", "1"]}
    result = refine_candidate(spec, events, tracks, initial, radius=500)
    assert result["success"]
    assert np.linalg.norm(np.array(result["position"])[:2] - truth[:2]) < 25
    assert len(result["event_residual_hz"]) == 2


def test_incompatible_second_event_time_cannot_slide_independently():
    truth, tracks, events, spec = scene()
    events[1].cpa += 45
    assert search(spec, events, tracks, np.array([truth]), offset_tolerance=10) == []


def test_refinement_clock_remains_identifiable_at_unix_epoch():
    truth, tracks, events, spec = scene()
    epoch = 1_700_000_000.0
    shifted = [Trajectory(t.id, t.times + epoch, t.positions) for t in tracks]
    initial = {
        "position": [200.0, -200.0, 0.0],
        "clock": epoch + 0.8,
        "f0": [110.0, 130.0],
        "flights": ["0", "1"],
    }
    result = refine_candidate(spec, events, shifted, initial, radius=500)
    assert abs(result["clock"] - epoch) < 0.1
    assert np.linalg.norm(np.array(result["position"])[:2] - truth[:2]) < 25


def test_search_accepts_detector_score_order():
    truth, tracks, events, spec = scene()
    results = search(spec, list(reversed(events)), tracks, np.array([truth]), shift_radius=0, shift_steps=1)
    assert results[0]["flights"] == ["0", "1"]


def test_silent_recording_cannot_produce_a_location():
    truth, tracks, events, spec = scene()
    spec.power[:] = 0
    assert search(spec, events, tracks, np.array([truth])) == []
