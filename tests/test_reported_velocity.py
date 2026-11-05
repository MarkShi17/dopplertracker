import numpy as np
import pytest

from dopplertracker.advanced import (
    ModelOptions,
    fuse_velocity,
    predict_result,
    prepare_track,
    smooth_trajectory,
)
from dopplertracker.data import KNOT, load_tracks, local_to_frame_velocity, parse_trace, save_tracks
from dopplertracker.geo import LocalFrame
from dopplertracker.physics import Trajectory
from dopplertracker.stress_scenes import (
    EXTENDED_SCENARIOS,
    STRESS_SCENARIOS,
    quantize_velocity,
    stress_scene,
)


def trace(rows):
    return {"icao": "abc123", "t": "PC12", "timestamp": 1000.0, "trace": rows}


def row(t, lon, speed=100, track=90.0, flags=0, rate=0, source="adsb_icao", geometric_rate=None):
    return [t, 40.0, lon, 3000, speed, track, flags, rate, None, source, 3200, geometric_rate, None, None]


def test_reported_velocity_parsed_in_frame_axes_with_source_flags():
    rows = [
        row(0, -73.0, geometric_rate=640),
        row(10, -72.99, rate=-128, flags=4),
        row(20, -72.98, rate=320),
        row(30, -72.97, source="mlat"),
    ]
    (track,) = parse_trace(trace(rows), LocalFrame(40, -73))
    v = track.velocities
    np.testing.assert_allclose(v[0], [100 * KNOT, 0, 640 * 0.3048 / 60], atol=0.02)
    # Frame up differs from local up by the 850 m baseline's curvature tilt.
    np.testing.assert_allclose(v[1, 2], -128 * 0.3048 / 60, atol=0.02)
    assert np.isnan(v[3]).all()  # MLAT velocity is not the aircraft's own report
    assert track.metadata["vertical_rate_source_runs"] == [["geometric", 2], ["baro", 1], ["none", 1]]
    assert track.metadata["position_source_runs"] == [["adsb_icao", 3], ["mlat", 1]]
    # Positions and segmentation do not depend on velocity fields.
    for r in rows:
        r[4] = r[5] = r[11] = None
    (plain,) = parse_trace(trace(rows), LocalFrame(40, -73))
    np.testing.assert_array_equal(plain.positions, track.positions)
    assert np.isnan(plain.velocities[:, :2]).all()


def test_velocity_rotation_matches_differenced_geodetic_motion():
    """Meridian convergence: local north 1.5 deg east of the frame origin is not frame north."""
    frame = LocalFrame(40.85, -73.5)
    lat, lon, h = 40.9, -72.0, 1500.0
    v_local = np.array([[30.0, 120.0, 2.0]])
    rotated = local_to_frame_velocity(frame, [[lat, lon]], v_local)[0]
    # Independent check: move the geodetic point by v dt along the local axes.
    dt = 1e-3
    a, e2, s = 6378137.0, 6.6943799901413165e-3, np.sin(np.deg2rad(lat))
    m_lat = np.deg2rad(a * (1 - e2) / (1 - e2 * s * s) ** 1.5 + h)
    m_lon = np.deg2rad((a / np.sqrt(1 - e2 * s * s) + h) * np.cos(np.deg2rad(lat)))
    ahead = [lat + 120.0 * dt / m_lat, lon + 30.0 * dt / m_lon, h + 2.0 * dt]
    numeric = (frame.to_local(ahead) - frame.to_local([lat, lon, h])) / dt
    np.testing.assert_allclose(rotated, numeric, atol=0.02)
    convergence = np.degrees(np.arctan2(rotated[0], rotated[1]) - np.arctan2(30.0, 120.0))
    assert 0.5 < abs(convergence) < 1.5
    partial = local_to_frame_velocity(frame, [[lat, lon]], [[30.0, 120.0, np.nan]])[0]
    np.testing.assert_allclose(partial[:2], rotated[:2], atol=0.05)
    assert np.isnan(partial[2])


def test_velocity_npz_round_trip_and_old_files(tmp_path):
    frame = LocalFrame(40, -73)
    tracks = parse_trace(trace([row(0, -73.0), row(10, -72.99), row(20, -72.98)]), frame)
    save_tracks(tmp_path / "v.npz", tracks, frame)
    (loaded,), _ = load_tracks(tmp_path / "v.npz")
    np.testing.assert_array_equal(loaded.velocities, tracks[0].velocities)
    old = Trajectory(tracks[0].id, tracks[0].times, tracks[0].positions, 30.0, tracks[0].metadata)
    save_tracks(tmp_path / "old.npz", [old], frame)
    (legacy,), _ = load_tracks(tmp_path / "old.npz")
    assert legacy.velocities is None


def test_hermite_uses_reported_velocity_only_when_requested():
    t = np.array([0.0, 10.0, 20.0, 30.0])
    p = np.column_stack((100 * t, 0 * t, 0 * t + 1000))
    reported = np.tile([90.0, 5.0, np.nan], (4, 1))
    plain = Trajectory("a", t, p)
    stored = Trajectory("a", t, p, velocities=reported)
    used = Trajectory("a", t, p, velocities=reported, use_velocities=True)
    query = np.linspace(0, 30, 31)
    np.testing.assert_array_equal(stored.at(query)[1], plain.at(query)[1])
    _, v = used.at(t)
    np.testing.assert_allclose(v[:, :2], reported[:, :2])
    np.testing.assert_allclose(v[:, 2], 0.0)  # NaN components fall back to differencing
    shifted = used.relative_to(5.0)
    np.testing.assert_allclose(shifted.at(t - 5.0)[1], v)
    with pytest.raises(ValueError):
        Trajectory("a", t, p, velocities=np.ones((3, 3)))


def curved(t):
    """Smooth turn at 110 m/s: position and exact velocity."""
    w = 0.02
    pos = np.column_stack((5500 * np.sin(w * t), 5500 * (1 - np.cos(w * t)), 1000 + 1.5 * t))
    vel = np.column_stack((110 * np.cos(w * t), 110 * np.sin(w * t), 1.5 + 0 * t))
    return pos, vel


def test_fused_smoother_beats_differencing_on_irregular_noisy_turn():
    rng = np.random.default_rng(3)
    t = np.cumsum(rng.uniform(1, 20, 25))
    pos, vel = curved(t)
    noisy = pos + rng.normal(0, 8, pos.shape)
    reported = quantize_velocity(vel)
    reported[::5, 2] = np.nan  # missing vertical rates
    track = Trajectory("turn", t, noisy, max_gap=30, velocities=reported)
    query = np.linspace(t[0], t[-1], 400)
    _, truth_v = curved(query)
    errors = {}
    for name, candidate in {
        "differenced": track,
        "spline": smooth_trajectory(track, "spline"),
        "reported": prepare_track(track, "none", "reported"),
        "fused": fuse_velocity(track),
    }.items():
        errors[name] = np.sqrt(np.mean((candidate.at(query)[1] - truth_v) ** 2))
    assert errors["fused"] < 0.5 * errors["differenced"]
    assert errors["fused"] < 0.5 * errors["spline"]
    assert errors["reported"] < errors["differenced"]
    fused = fuse_velocity(track)
    assert np.isfinite(fused.velocities).all() and fused.use_velocities
    assert fused.metadata["velocity_fusion"] == "kalman"
    position_only = fuse_velocity(Trajectory("turn", t, noisy, max_gap=30))
    assert np.isfinite(position_only.positions).all()


def test_velocity_options_validate_and_default_is_unchanged():
    assert ModelOptions().velocity == "differenced"
    with pytest.raises(ValueError):
        ModelOptions(velocity="gps")
    with pytest.raises(ValueError):
        ModelOptions(velocity="fused", smoothing="spline")
    track = stress_scene("clean", 2)["tracks"][0]
    reported = prepare_track(track, "spline", "reported")  # no velocities: exact fallback
    expected = smooth_trajectory(track, "spline")
    query = np.linspace(track.times[0], track.times[-1], 50)
    np.testing.assert_array_equal(reported.at(query)[1], expected.at(query)[1])


def test_extended_scenes_are_opt_in_and_quantised():
    assert not set(EXTENDED_SCENARIOS) & set(STRESS_SCENARIOS)
    paired, base = stress_scene("adsb_noise_velocity", 1), stress_scene("adsb_noise", 1)
    np.testing.assert_array_equal(paired["spectrogram"].power, base["spectrogram"].power)
    for a, b in zip(paired["tracks"], base["tracks"], strict=True):
        np.testing.assert_array_equal(a.positions, b.positions)
        assert b.velocities is None
        steps = a.velocities[:, :2] / (1852 / 3600)
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)
    irregular = stress_scene("irregular_cadence", 1)
    gaps = np.concatenate([np.diff(t.times) for t in irregular["tracks"]])
    assert gaps.min() >= 1.0 and gaps.max() <= 20.0 and np.std(gaps) > 3
    turning = stress_scene("turning_sparse", 1)
    for track in turning["tracks"]:
        heading = np.unwrap(np.arctan2(track.velocities[:, 1], track.velocities[:, 0]))
        assert np.ptp(heading) > 0.4
        assert np.diff(track.times).min() >= 8.0


def test_track_offset_equals_relabelled_track_and_latency_scene():
    scene, clean = stress_scene("adsb_latency", 0), stress_scene("clean", 0)
    late, base = scene["tracks"][2], clean["tracks"][2]
    np.testing.assert_allclose(late.times - base.times, 0.6)
    event = scene["events"][2]
    times = event.sample_times(scene["spectrogram"])
    result = {
        "position": scene["truth"].tolist(),
        "clock": scene["clock"],
        "f0": [100.0] * 4,
        "track_time_offsets_s": [0.0, 0.0, 0.6, 0.0],
    }
    corrected = predict_result(late, event, times, result, 2)
    expected = predict_result(base, event, times, {**result, "track_time_offsets_s": None}, 2)
    finite = np.isfinite(expected) & np.isfinite(corrected)
    assert finite.mean() > 0.9
    np.testing.assert_allclose(corrected[finite], expected[finite], rtol=1e-9)
    assert not ModelOptions().fit_track_offsets
