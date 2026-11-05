import inspect

import numpy as np
import pytest

from dopplertracker.physics import predict
from dopplertracker.stress_scenes import STRESS_SCENARIOS, stress_scene


def test_stress_scene_reproducible_and_compact():
    a, b = stress_scene("clean", 9), stress_scene("clean", 9)
    assert len(STRESS_SCENARIOS) >= 16
    assert np.array_equal(a["truth"], b["truth"])
    assert np.array_equal(a["spectrogram"].power, b["spectrogram"].power)
    assert a["spectrogram"].power.shape == (1301, 1300)
    assert [e.cpa for e in a["events"]] == sorted(e.cpa for e in a["events"])


def test_candidate_is_not_truth_derived():
    a, b = stress_scene("clean", 1), stress_scene("clean", 2)
    assert a["candidate"]["position"] == b["candidate"]["position"] == [0.0, 0.0, 0.0]
    assert np.linalg.norm(a["candidate"]["position"][:2] - a["truth"][:2]) > 20
    assert a["clock"] == 12345.6 and abs(a["candidate"]["clock"] - a["clock"]) < 4


def test_nuisances_are_real_observable_changes():
    clean, noisy = stress_scene("clean", 4), stress_scene("adsb_noise", 4)
    assert not np.allclose(clean["tracks"][0].positions, noisy["tracks"][0].positions)
    gaps = stress_scene("gaps", 4)["tracks"][0]
    assert np.max(np.diff(gaps.times)) > gaps.max_gap
    wind = stress_scene("wind", 4)
    assert wind["atmosphere_truth"]["wind"] != (0.0, 0.0, 0.0)
    climb = clean["tracks"][0].positions[:, 2]
    assert np.ptp(climb) > 100


@pytest.mark.parametrize("scenario", STRESS_SCENARIOS)
def test_audio_time_events_have_track_prediction_coverage(scenario):
    scene = stress_scene(scenario, 5)
    spec = scene["spectrogram"]
    assert spec.times[0] == 0.0
    for event, track in zip(scene["events"], scene["tracks"]):
        audio_times = event.sample_times(spec)
        _, emission = predict(track, scene["truth"], audio_times + scene["clock"], event.f0)
        coverage = np.isfinite(emission).mean()
        # Gaps are deliberately allowed to remove a material section; all
        # other stressors retain enough trajectory support for scoring.
        assert coverage > (0.35 if scenario in {"gaps", "combined_errors"} else 0.9)


def test_generator_does_not_use_inference_predictors():
    source = inspect.getsource(stress_scene)
    assert "predict(" not in source and "closest_approaches" not in source and "ideal_curve" not in source
