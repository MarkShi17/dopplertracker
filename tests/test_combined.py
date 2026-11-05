"""The opt-in atmosphere, velocity and measurement models compose in one fit and are replayed."""

import numpy as np
import pytest

from dopplertracker import evidence
from dopplertracker.advanced import ModelOptions, fit_projected, predict_result
from dopplertracker.atmosphere import LayeredAtmosphere
from dopplertracker.audio_benchmark import measurement_scene
from dopplertracker.demod import Audio
from dopplertracker.physics import Trajectory
from dopplertracker.signal import from_samples
from dopplertracker.stress_benchmark import ATMOSPHERE_METHODS, LAYERED_METHODS
from dopplertracker.stress_scenes import (
    EXTENDED_SCENARIOS,
    STRESS_SCENARIOS,
    layered_profiles,
    quantize_velocity,
    stress_scene,
)

FS = 2048


@pytest.fixture(scope="module")
def combined_fit():
    scene = measurement_scene(0, "fast_chirp")
    spec = from_samples(scene["waveform"], FS, 4.0, 0.5, (40, 450)).whiten()
    # Straight constant-velocity passes: 5 m position noise, exact quantised velocity,
    # so the fused and differenced trajectories differ measurably.
    rng = np.random.default_rng(7)
    tracks = [
        Trajectory(
            t.id,
            t.times,
            t.positions + rng.normal(0, 5.0, t.positions.shape),
            t.max_gap,
            {"synthetic": True},
            velocities=quantize_velocity(np.gradient(t.positions, t.times, axis=0)),
        )
        for t in scene["tracks"]
    ]
    candidate = {"position": [0.0, 0.0, 0.0], "clock": 0.35, "f0": [e.f0 for e in scene["events"]]}
    profile = LayeredAtmosphere.from_lapse_rate(
        20.0, wind_surface=(2.0, 0.0), wind_top=(8.0, 3.0), wind_top_height_m=3000.0
    )
    options = ModelOptions(
        degree=1,
        radius=900.0,
        ridge="demod",
        reextract=1,
        noise="snr",
        velocity="fused",
        fit_sound_speed=True,
        fit_wind=True,
        fit_track_offsets=True,
    )
    result = fit_projected(
        spec,
        scene["events"],
        tracks,
        candidate,
        options,
        profile,
        audio=Audio(scene["waveform"], FS),
        keep_observations=True,
    )
    return scene, spec, tracks, result


def test_layered_fused_demodulated_fit_records_every_model(combined_fit):
    scene, _, _, result = combined_fit
    assert result["atmosphere"]["type"] == "layered"
    assert result["options"]["velocity"] == "fused" and result["options"]["ridge"] == "demod"
    assert len(result["track_time_offsets_s"]) == 3 and result["track_time_offsets_s"][0] == 0.0
    assert result["measurement"]["passes"] >= 1
    assert np.linalg.norm(np.asarray(result["position"])[:2] - scene["truth"][:2]) < 50.0


def test_predict_result_replays_the_combined_fit(combined_fit):
    scene, _, tracks, result = combined_fit
    for i, (event, track, observed) in enumerate(
        zip(scene["events"], tracks, result["observations"], strict=True)
    ):
        valid = np.asarray(observed["valid"])
        curve = predict_result(track, event, np.asarray(observed["times"]), result, i)
        np.testing.assert_allclose(curve[valid], np.asarray(observed["model_hz"])[valid], atol=1e-6)


def test_evidence_replays_the_combined_fit(combined_fit, monkeypatch):
    scene, spec, tracks, result = combined_fit
    calls = []

    def recording(track, observer, receive, f0, atmosphere):
        factor, emission = original(track, observer, receive, f0, atmosphere)
        calls.append((track, atmosphere, factor, emission))
        return factor, emission

    original = evidence.predict
    monkeypatch.setattr(evidence, "predict", recording)
    evidence.assess_candidate(spec, scene["events"], tracks, result)
    assert len(calls) == 3
    for i, (event, (track, medium, factor, emission)) in enumerate(zip(scene["events"], calls, strict=True)):
        assert track.metadata.get("velocity_fusion") == "kalman"
        assert isinstance(medium, LayeredAtmosphere)
        assert medium.sound_speed_offset == result["atmosphere"]["sound_speed_offset_m_s"]
        dt = emission - result["track_time_offsets_s"][i] - result["source_reference_utc"][i]
        frequency = factor * (result["f0"][i] + result["frequency_drift_hz_s"][i] * dt)
        expected = predict_result(tracks[i], event, event.sample_times(spec), result, i)
        finite = np.isfinite(expected)
        assert finite.mean() > 0.5
        np.testing.assert_allclose(frequency[finite], expected[finite], atol=1e-6)


def test_realistic_combined_scene_is_opt_in_and_combines_every_nuisance():
    assert "realistic_combined" in EXTENDED_SCENARIOS and "realistic_combined" not in STRESS_SCENARIOS
    assert set(LAYERED_METHODS) < set(ATMOSPHERE_METHODS)
    scene = stress_scene("realistic_combined", 1)
    truth, forecast = layered_profiles(1)
    assert scene["atmosphere_profile"] == truth and scene["atmosphere_forecast"] == forecast
    p = scene["metadata"]["parameters"]
    assert p["layered"] and p["rpm_drift"] > 0
    heights = [np.median(t.positions[:, 2]) for t in scene["tracks"]]
    assert min(heights) < 1100 and max(heights) > 5300
    gaps = np.concatenate([np.diff(t.times) for t in scene["tracks"]])
    assert gaps.min() >= 1.0 and gaps.max() <= 20.0
    for track in scene["tracks"]:
        steps = track.velocities[:, :2] / (1852 / 3600)
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)
    # Acoustics share the layered scene's main stream; only the tracks differ in kind.
    base = stress_scene("layered_atmosphere", 1)
    np.testing.assert_array_equal(scene["truth"], base["truth"])
    assert len(scene["tracks"][0].times) != len(base["tracks"][0].times)
