import numpy as np
import pytest

from dopplertracker.benchmark import synthetic_scene
from dopplertracker.evidence import assess_candidate
from dopplertracker.evidence_controls import independent_tone_scene, run_evidence_controls
from dopplertracker.physics import Trajectory
from dopplertracker.signal import Spectrogram


def _candidate(scene, **changes):
    result = {
        "position": scene["truth"].tolist(),
        "clock": 0.0,
        "f0": [event.f0 for event in scene["events"]],
    }
    result.update(changes)
    return result


@pytest.mark.parametrize("seed", range(10))  # development seeds, fixed before held-out check
def test_true_explicit_tones_are_accepted(seed):
    scene = independent_tone_scene(seed)
    spec, events, tracks, truth = scene["spectrogram"], scene["events"], scene["tracks"], scene["truth"]
    result = assess_candidate(spec, events, tracks, {"position": truth, "clock": 0.0, "f0": [105, 125]})
    assert result["accepted"]
    assert result["coverage"] >= 0.9
    assert all(item["support_over_null"] > 1 for item in result["per_event"])


@pytest.mark.parametrize("seed", range(10))
def test_development_nulls_wrong_flights_and_clock_are_rejected(seed):
    scene = synthetic_scene("clean", seed)
    rng = np.random.default_rng(seed + 1000)
    noise = Spectrogram(
        scene["spectrogram"].times,
        scene["spectrogram"].frequencies,
        rng.exponential(0.15, scene["spectrogram"].power.shape).astype(np.float32),
    )
    broad = Spectrogram(
        noise.times,
        noise.frequencies,
        (noise.power + rng.uniform(0, 20, len(noise.times))[None, :]).astype(np.float32),
    )
    wrong_position = _candidate(scene, position=(scene["truth"] + [1200, -900, 0]).tolist())
    wrong_clock = _candidate(scene, clock=15.0)
    assert not assess_candidate(noise, scene["events"], scene["tracks"], _candidate(scene))["accepted"]
    assert not assess_candidate(broad, scene["events"], scene["tracks"], _candidate(scene))["accepted"]
    assert not assess_candidate(scene["spectrogram"], scene["events"], scene["tracks"], wrong_position)[
        "accepted"
    ]
    assert not assess_candidate(scene["spectrogram"], scene["events"], scene["tracks"], wrong_clock)[
        "accepted"
    ]


@pytest.mark.parametrize("seed", range(100, 106))  # held out: thresholds above were not tuned here
def test_held_out_true_and_null_rates(seed):
    scene = synthetic_scene("clean", seed)
    assert assess_candidate(scene["spectrogram"], scene["events"], scene["tracks"], _candidate(scene))[
        "accepted"
    ]
    rng = np.random.default_rng(seed)
    silence = Spectrogram(
        scene["spectrogram"].times,
        scene["spectrogram"].frequencies,
        rng.exponential(0.15, scene["spectrogram"].power.shape).astype(np.float32),
    )
    assert not assess_candidate(silence, scene["events"], scene["tracks"], _candidate(scene))["accepted"]


def test_partial_track_and_harmonic_out_of_band_cannot_hide_under_other_events():
    scene = synthetic_scene("clean", 2)
    partial = list(scene["tracks"])
    track = partial[1]
    partial[1] = Trajectory(track.id, track.times[:25], track.positions[:25], track.max_gap, track.metadata)
    result = assess_candidate(scene["spectrogram"], scene["events"], partial, _candidate(scene))
    assert not result["accepted"]
    assert result["per_event"][1]["coverage"] < 0.75
    out_of_band = _candidate(scene, f0=[event.f0 for event in scene["events"]])
    out_of_band["f0"][0] = 600.0  # all claimed harmonics exceed the recording band
    result = assess_candidate(scene["spectrogram"], scene["events"], scene["tracks"], out_of_band)
    assert not result["accepted"]
    assert result["per_event"][0]["coverage"] < 0.75


def test_curvature_and_json_safe_result():
    scene = synthetic_scene("clean", 1)
    candidate = _candidate(scene, frequency_drift_hz_s=[0, 0, 0], frequency_curvature_hz_s2=[0, 0, 0])
    result = assess_candidate(scene["spectrogram"], scene["events"], scene["tracks"], candidate)
    import json

    json.dumps(result, allow_nan=False)
    assert "not a calibrated probability" in result["calibration_note"]


def test_no_signal_is_rejected():
    scene = synthetic_scene("clean", 0)
    silent = Spectrogram(
        scene["spectrogram"].times,
        scene["spectrogram"].frequencies,
        np.zeros_like(scene["spectrogram"].power),
    )
    assert not assess_candidate(silent, scene["events"], scene["tracks"], _candidate(scene))["accepted"]


def test_controls_report_records_frozen_held_out_rates():
    report = run_evidence_controls(range(100, 103))
    assert report["trials"] == 3
    assert report["acceptance"]["true"]["accepted"] == 3
    for name in ("noise", "broadband", "wrong_position", "wrong_clock"):
        assert report["acceptance"][name]["accepted"] == 0


def test_candidate_smoothing_reapplies_the_fitted_trajectory_model():
    from dopplertracker.advanced import smooth_trajectory

    scene = synthetic_scene("clean", 3)
    rng = np.random.default_rng(88)
    noisy = [
        Trajectory(
            track.id,
            track.times,
            track.positions + rng.normal(0, 12, track.positions.shape),
            track.max_gap,
            track.metadata,
        )
        for track in scene["tracks"]
    ]
    result = _candidate(scene, options={"smoothing": "spline"})
    assessed = assess_candidate(scene["spectrogram"], scene["events"], noisy, result)
    manually_smoothed = [smooth_trajectory(track, "spline") for track in noisy]
    expected = assess_candidate(scene["spectrogram"], scene["events"], manually_smoothed, _candidate(scene))
    assert assessed["accepted"] == expected["accepted"]
    np.testing.assert_allclose(
        [x["center_score"] for x in assessed["per_event"]], [x["center_score"] for x in expected["per_event"]]
    )


def test_available_fundamental_is_not_rejected_when_second_harmonic_leaves_band():
    from dopplertracker.stress_scenes import stress_scene

    scene = stress_scene("clean", 0)
    candidate = {
        "position": scene["truth"].tolist(),
        "clock": scene["clock"],
        "f0": [f["source_f0_hz"] for f in scene["metadata"]["flights"]],
    }
    result = assess_candidate(scene["spectrogram"], scene["events"], scene["tracks"], candidate)
    assert result["coverage"] > 0.95
    assert result["accepted"]


def test_event_window_mostly_outside_recording_counts_as_missing_coverage():
    from dataclasses import replace

    scene = independent_tone_scene(0)
    event = replace(scene["events"][0], window=(-1000.0, 1000.0))
    result = assess_candidate(
        scene["spectrogram"],
        [event],
        scene["tracks"][:1],
        {"position": scene["truth"], "clock": 0.0, "f0": [105.0]},
    )
    assert not result["accepted"]
    assert result["coverage"] < 0.3


def test_internal_audio_gap_cannot_count_as_recorded_support():
    from dataclasses import replace

    scene = independent_tone_scene(0)
    spec = scene["spectrogram"]
    keep = np.r_[np.arange(35, 40), np.arange(120, 125)]
    sparse = Spectrogram(spec.times[keep], spec.frequencies, spec.power[:, keep])
    event = replace(scene["events"][0], window=(35.0, 125.0))
    result = assess_candidate(
        sparse, [event], scene["tracks"][:1], {"position": scene["truth"], "clock": 0.0, "f0": [105.0]}
    )
    assert result["coverage"] < 0.2
    assert not result["accepted"]


def test_spectral_refinement_exports_atmosphere_for_frozen_validation():
    from dopplertracker.physics import Atmosphere
    from dopplertracker.refine import refine_spectral

    scene = synthetic_scene("clean", 0)
    atmosphere = Atmosphere(337.0, (4.0, -2.0, 0.0))
    result = refine_spectral(
        scene["spectrogram"], scene["events"], scene["tracks"], _candidate(scene), atmosphere=atmosphere
    )
    assert result["sound_speed_m_s"] == atmosphere.c
    assert result["wind_enu_m_s"] == list(atmosphere.wind)
    explicit = assess_candidate(scene["spectrogram"], scene["events"], scene["tracks"], result, atmosphere)
    restored = assess_candidate(scene["spectrogram"], scene["events"], scene["tracks"], result)
    assert explicit == restored
