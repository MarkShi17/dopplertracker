import json

import numpy as np
import pytest
from test_search import scene

from dopplertracker.cli import main
from dopplertracker.data import load_tracks, save_tracks
from dopplertracker.detection import save_events
from dopplertracker.geo import LocalFrame
from dopplertracker.identification import _distractors
from dopplertracker.physics import Atmosphere, Trajectory, closest_approaches
from dopplertracker.scoring import joint_score, subsonic
from dopplertracker.search import (
    _deduplicate,
    _polish,
    _refine_minima,
    _Samples,
    _Scorer,
    _window_minima,
    fast_search,
    search,
)
from dopplertracker.stress_scenes import stress_scene


def grid(half=1500.0, count=13, z=0.0):
    axis = np.linspace(-half, half, count)
    xx, yy = np.meshgrid(axis, axis)
    return np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))


def keyed(results):
    return {(tuple(np.round(r["position"], 6)), tuple(r["flights"])): r for r in results}


def passes(track, observer, atmosphere, step=1.0):
    """All fast-path closest approaches of one track at one observer."""
    samples = _Samples([track], (-1e9, 1e9, -1e9, 1e9), step)
    wind = np.asarray(atmosphere.wind, float)
    index, endpoint = [], []
    for window in samples.windows(8):
        _, i, e = _window_minima(samples, window, observer[None], atmosphere.c, wind, 1e9)
        index.append(i)
        endpoint.append(e)
    index, endpoint = np.concatenate(index), np.concatenate(endpoint)
    observers = np.repeat(observer[None], len(index), axis=0)
    medium = (atmosphere.c, wind)
    heard, distance, speed, delay, raw_id, valid = _refine_minima(
        samples, observers, index, endpoint, *medium
    )
    keep = valid & _deduplicate(np.zeros(len(index), int), raw_id, delay)
    run = samples.sample_run[index[keep]]
    heard, distance, speed, _ = _polish(
        samples, [track], run, (heard - delay)[keep], raw_id[keep], endpoint[keep], observers[keep], *medium
    )
    return sorted(zip(heard, distance, speed))


@pytest.mark.parametrize("wind", [(0.0, 0.0, 0.0), (14.0, -9.0, 0.0)])
def test_straight_pass_closest_approach_matches_scalar_solver(wind):
    rng = np.random.default_rng(0)
    atmosphere = Atmosphere(343.0, wind)
    times = np.sort(rng.uniform(0, 300, 40))
    track = Trajectory("line", times, np.column_stack((-12000 + 80 * times, 600 + 0 * times, 1500 + times)))
    for observer in rng.uniform([-8000, -5000, 0], [8000, 5000, 50], (20, 3)):
        expected = closest_approaches(track, observer, atmosphere)
        found = passes(track, observer, atmosphere)
        assert len(found) == len(expected)
        tolerance = 1e-5 if not any(wind) else 0.01
        np.testing.assert_allclose(found, expected, atol=tolerance, rtol=0)


@pytest.mark.parametrize("wind", [(0.0, 0.0, 0.0), (14.0, -9.0, 0.0)])
def test_turning_track_closest_approaches_match_scalar_solver(wind):
    # A climbing, nearly full circle (radius 3 km, 1.8 deg/s) sampled every 4 s, with
    # observers inside, outside and near its open end, where the interpolant wiggles
    # between raw samples (no approach is reported there).
    times = np.arange(0.0, 200.0, 4.0)
    angle = times / 200.0 * 2 * np.pi
    positions = np.column_stack((3000 * np.cos(angle), 3000 * np.sin(angle), 900 + times))
    track = Trajectory("arc", times, positions)
    atmosphere = Atmosphere(343.0, wind)
    observers = ([500.0, 200.0, 0.0], [-4500.0, 800.0, 10.0], [2500.0, -2500.0, 0.0], [3500.0, -300.0, 0.0])
    for observer in map(np.array, observers):
        expected = closest_approaches(track, observer, atmosphere)
        found = passes(track, observer, atmosphere)
        assert len(found) == len(expected)
        np.testing.assert_allclose(found, expected, atol=2e-3, rtol=0)


def test_noisy_densely_sampled_track_matches_scalar_solver():
    # Sub-second reports with 15 m jitter make the interpolant's speed swing by tens
    # of m/s within a second; approaches must still follow the interpolant itself.
    rng = np.random.default_rng(4)
    times = np.cumsum(rng.choice([0.25, 0.5, 4.0, 12.0], 150))
    positions = np.column_stack((-9000 + 120 * times, 2000 + 0 * times, 1200 + 0 * times))
    track = Trajectory("noisy", times, positions + rng.normal(0, 15, positions.shape))
    same = total = 0
    for observer in rng.uniform([-6000, -6000, 0], [6000, 6000, 0], (25, 3)):
        expected = np.array(closest_approaches(track, observer)).reshape(-1, 3)
        found = np.array(passes(track, observer, Atmosphere())).reshape(-1, 3)
        assert len(found) == len(expected)
        # Speed is steep here: a 0.3 ms emission difference can move it by 0.2 m/s.
        close = (np.abs(found[:, :2] - expected[:, :2]).max(axis=1) < 2e-3) & np.isclose(
            found[:, 2], expected[:, 2], rtol=5e-3
        )
        # A jittered interpolant can hold two minima in one raw bracket; the bounded
        # scalar solver may stop in the shallower one; ours is never more than 1 cm shallower.
        assert np.all(close | (found[:, 1] < expected[:, 1] + 0.01))
        same, total = same + close.sum(), total + len(close)
    assert same >= 0.9 * total


def test_batched_scorer_matches_joint_score():
    truth, tracks, events, spec = scene()
    for method in ("brightness", "contrast"):
        scorer = _Scorer(spec, events, method, Atmosphere(), 3.0, 7)
        for observer in grid(800.0, 4):
            expected = joint_score(spec, events, tracks, observer, 0.4, method)
            score, clock, f0, scores = scorer(tracks, observer, 0.4)
            assert score == pytest.approx(expected[0], abs=1e-9)
            assert clock == expected[1] and f0 == expected[2]
            np.testing.assert_allclose(scores, expected[3], atol=1e-9)


def test_fast_search_scoring_everything_reproduces_search():
    truth, tracks, events, spec = scene()
    points = grid()
    slow = search(spec, events, tracks, points, shift_radius=1, shift_steps=3)
    fast, summary = fast_search(
        spec, events, tracks, points, shift_radius=1, shift_steps=3, per_combination=None
    )
    assert slow and set(keyed(slow)) == set(keyed(fast))
    for key, result in keyed(slow).items():
        other = keyed(fast)[key]
        assert other["score"] == pytest.approx(result["score"], abs=1e-4)
        assert other["clock"] == pytest.approx(result["clock"], abs=0.02)
    assert summary["scored"] == summary["candidates"] == len(slow)
    np.testing.assert_allclose(fast[0]["position"], slow[0]["position"])


def test_known_clock_and_silence_filters_match_search():
    truth, tracks, events, spec = scene()
    # A third aircraft passes the west of the grid between the two events.
    t = np.arange(100.0, 320.0, 2.0)
    intruder = Trajectory("2", t, np.column_stack((-1200 + 0 * t, -9000 + (t - 100) * 85, 800 + 0 * t)))
    tracks = [*tracks, intruder]
    points = grid()
    for options in (dict(clock=0.0, clock_tolerance=2.0), dict(silence_radius=2500.0)):
        slow = search(spec, events, tracks, points, shift_radius=1, shift_steps=3, **options)
        fast, _ = fast_search(
            spec, events, tracks, points, shift_radius=1, shift_steps=3, per_combination=None, **options
        )
        assert set(keyed(slow)) == set(keyed(fast))
    assert len(slow) < len(search(spec, events, tracks, points, shift_radius=1, shift_steps=3))


def test_thinned_scoring_climbs_to_the_exhaustive_optimum():
    truth, tracks, events, spec = scene()
    points = grid(1500.0, 31)
    everything, _ = fast_search(
        spec, events, tracks, points, shift_radius=1, shift_steps=3, per_combination=None
    )
    thinned, summary = fast_search(
        spec, events, tracks, points, shift_radius=1, shift_steps=3, per_combination=12
    )
    assert summary["scored"] < summary["candidates"] / 4
    np.testing.assert_allclose(thinned[0]["position"], everything[0]["position"])
    assert thinned[0]["score"] == pytest.approx(everything[0]["score"])


def test_decoy_scene_ranks_the_same_flights_as_search():
    scene_ = stress_scene("clean", 0)
    rng = np.random.default_rng(3)
    tracks = [*scene_["tracks"][:2], *_distractors(scene_["tracks"], 6, rng)]
    points = grid(400.0, 5)
    arguments = dict(offset_tolerance=14.0, shift_radius=3.0, shift_steps=3)
    slow = search(scene_["spectrogram"], scene_["events"][:2], tracks, points, **arguments)
    fast, _ = fast_search(scene_["spectrogram"], scene_["events"][:2], tracks, points, **arguments)
    assert slow[0]["flights"] == fast[0]["flights"] == [t.id for t in scene_["tracks"][:2]]
    np.testing.assert_allclose(fast[0]["position"], slow[0]["position"])


def test_supersonic_track_is_skipped_and_counted():
    truth, tracks, events, spec = scene()
    glitched = tracks[1].positions.copy()
    glitched[len(glitched) // 2, 0] += 1500.0
    bad = Trajectory("1", tracks[1].times, glitched)
    assert not subsonic(bad) and subsonic(tracks[1])
    score, _, f0, _ = joint_score(spec, events, [tracks[0], bad], truth, 0.0)
    assert score == -np.inf and f0 == []
    points = grid(1500.0, 7)
    assert search(spec, events, [tracks[0], bad], points) == []
    results, summary = fast_search(spec, events, [tracks[0], bad], points)
    assert results == [] and summary["skipped_supersonic"] > 0 and summary["scored"] == 0


def test_type_exclusion_drops_jets(tmp_path):
    truth, tracks, events, spec = scene()
    typed = [
        Trajectory(t.id, t.times, t.positions, metadata={"type": kind})
        for t, kind in zip(tracks, ["C172", "B738"])
    ]
    results, summary = fast_search(spec, events, typed, grid(), exclude_types=["B738"])
    assert results == [] and summary["excluded_tracks"] == 1
    frame = LocalFrame(40.85, -73.5)
    save_tracks(tmp_path / "traces.npz", typed, frame)
    loaded, _ = load_tracks(tmp_path / "traces.npz", exclude_types={"B738"})
    assert [t.id for t in loaded] == ["0"]


def test_tiles_without_usable_passes_are_harmless():
    truth, tracks, events, spec = scene()
    # Small chunks and tiles: some flushes keep nothing, some tiles see no window.
    far = grid(500.0, 5) + [60_000.0, 0.0, 0.0]
    points = np.vstack((grid(), far))
    results, summary = fast_search(spec, events, tracks, points, tile_size=1000.0, chunk=3)
    expected, _ = fast_search(spec, events, tracks, grid())
    assert [r["position"] for r in results] == [r["position"] for r in expected]


def test_fast_search_validates_inputs():
    truth, tracks, events, spec = scene()
    with pytest.raises(ValueError, match="clock"):
        fast_search(spec, events[:1], tracks, grid())
    with pytest.raises(ValueError, match="points"):
        fast_search(spec, events, tracks, np.zeros((0, 3)))
    with pytest.raises(ValueError, match="homogeneous"):
        fast_search(spec, events, tracks, grid(), atmosphere=object())


def test_cli_fast_search_agrees_with_default(tmp_path):
    truth, tracks, events, spec = scene()
    frame = LocalFrame(40.85, -73.5)
    origin = frame.to_geodetic(np.zeros(3))
    moved = [Trajectory(t.id, t.times, t.positions, metadata={"type": "C172"}) for t in tracks]
    save_tracks(tmp_path / "traces.npz", moved, frame)
    spec.save(tmp_path / "spec.npz")
    save_events(events, tmp_path / "events.json")
    (tmp_path / "jets.json").write_text(json.dumps(["B738"]))
    lat, lon = origin[:2]
    common = [
        "search",
        str(tmp_path / "spec.npz"),
        "--events",
        str(tmp_path / "events.json"),
        "--traces",
        str(tmp_path / "traces.npz"),
        "--elevation",
        "0",
        "--bbox",
        str(lat - 0.01),
        str(lat + 0.01),
        str(lon - 0.012),
        str(lon + 0.012),
        "--spacing",
        "400",
        "--raw-score",
    ]
    assert main([*common, "--out", str(tmp_path / "slow.json")]) == 0
    assert (
        main(
            [
                *common,
                "--fast",
                "--jet-types",
                str(tmp_path / "jets.json"),
                "--out",
                str(tmp_path / "fast.json"),
            ]
        )
        == 0
    )
    slow = json.loads((tmp_path / "slow.json").read_text())["candidates"]
    fast = json.loads((tmp_path / "fast.json").read_text())["candidates"]
    assert fast[0]["flights"] == slow[0]["flights"]
    np.testing.assert_allclose(fast[0]["position"], slow[0]["position"], atol=1e-6)
