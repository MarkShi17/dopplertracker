import json

import numpy as np
import pytest

from dopplertracker.advanced import ModelOptions, fit_projected
from dopplertracker.data import save_tracks
from dopplertracker.detection import Event, load_events
from dopplertracker.discovery import (
    DiscoveryOptions,
    RidgeMap,
    TraceIndex,
    discover,
    discover_and_refit,
    enumerate_passes,
    icao,
    pass_limit,
)
from dopplertracker.discovery_benchmark import run_discovery_benchmark, summarize
from dopplertracker.geo import LocalFrame
from dopplertracker.identification import _distractors
from dopplertracker.physics import Trajectory
from dopplertracker.signal import Spectrogram
from dopplertracker.stress_scenes import stress_scene

FAST = DiscoveryOptions(band=(80.0, 160.0), f0_step=0.2)


@pytest.fixture(scope="module")
def scene():
    scene = stress_scene("clean", 0)
    decoys = _distractors(scene["tracks"], 4, np.random.default_rng(3))
    scene["all_tracks"] = [*scene["tracks"], *decoys]
    scene["ridges"] = RidgeMap(scene["spectrogram"])
    return scene


def truth_candidate(scene, clock_shift=0.0):
    return {"position": scene["truth"].tolist(), "clock": scene["clock"] + clock_shift}


def test_identity_and_altitude_aware_limit():
    assert icao("A1DFEB:1788899524.900") == "a1dfeb"
    assert icao("ace391") == "ace391"
    assert pass_limit(0.0) == 6000.0
    assert pass_limit(-200.0) == 6000.0
    # A PC-12 at 5.5 km altitude stays in range out to the 10 km cap.
    assert pass_limit(5500.0) == 10000.0
    with pytest.raises(ValueError):
        DiscoveryOptions(null_shifts=(0.0, 20.0))
    with pytest.raises(ValueError):
        DiscoveryOptions(drift_steps=3)


def test_enumeration_keeps_one_pass_per_gap_and_recording_bounds():
    times = np.r_[np.arange(-100.0, 0.0, 4.0), np.arange(30.0, 101.0, 4.0)]
    positions = np.column_stack((110.0 * times, np.full_like(times, 1500.0), np.full_like(times, 900.0)))
    track = Trajectory("abc123:1", 1000.0 + times, positions, max_gap=15.0)
    candidate = {"position": [0.0, 0.0, 0.0], "clock": 900.0}
    passes = enumerate_passes([track], candidate, (0.0, 300.0))
    assert len(passes) == 1
    assert passes[0]["icao"] == "abc123" and passes[0]["altitude_m"] > 800
    assert not enumerate_passes([track], {**candidate, "clock": 1500.0}, (0.0, 300.0))


def test_discovery_confirms_unused_flights_and_rejects_decoys(scene):
    found = discover(
        scene["ridges"],
        scene["all_tracks"],
        truth_candidate(scene),
        options=FAST,
        used_flights=["stress-0", "stress-1"],
    )
    summary = found["summary"]
    assert summary["confirmed_flights"] == ["stress-2", "stress-3"]
    assert summary["used_flights_confirmed"] == ["stress-0", "stress-1"]
    assert summary["confirmed_independent"] == 2 and summary["z_sum"] > 2 * FAST.threshold
    assert all(isinstance(e, Event) for e in found["events"])
    assert [t.id for t in found["tracks"]] == ["stress-2", "stress-3"]
    true_f0 = [94.0, 145.0]
    assert np.allclose([e.f0 for e in found["events"]], true_f0, atol=0.5)
    assert all(not p["confirmed"] for p in summary["passes"] if p["flight"].startswith("decoy"))
    json.dumps(summary, allow_nan=False)


def test_wrong_clock_and_noise_give_no_confirmation(scene):
    wrong = discover(scene["ridges"], scene["all_tracks"], truth_candidate(scene, 75.0), options=FAST)
    assert wrong["summary"]["confirmed_independent"] == 0
    noise = stress_scene("noise_only", 0)
    report = discover(noise["spectrogram"], noise["tracks"], truth_candidate(noise), options=FAST)["summary"]
    assert report["accepted"] == 0


def test_minimum_sweep_rejects_flat_curves(scene):
    options = DiscoveryOptions(band=(80.0, 160.0), f0_step=0.2, min_sweep=5.0)
    summary = discover(scene["ridges"], scene["tracks"], truth_candidate(scene), options=options)["summary"]
    assert {p["status"] for p in summary["passes"]} == {"flat_curve"}
    assert summary["confirmed_independent"] == 0


def test_joint_refit_uses_discovered_events(scene):
    spec, events, tracks = scene["spectrogram"], scene["events"], scene["tracks"]
    initial = {**scene["candidate"], "f0": scene["candidate"]["f0"][:2], "flights": ["stress-0", "stress-1"]}
    two = fit_projected(spec, events[:2], tracks[:2], initial, ModelOptions(degree=1))
    pipeline = discover_and_refit(
        spec, events[:2], tracks[:2], scene["all_tracks"], two, options=FAST, ridges=scene["ridges"]
    )
    result = pipeline["result"]
    assert result["flights"] == ["stress-0", "stress-1", "stress-2", "stress-3"]
    assert np.linalg.norm(np.asarray(result["position"])[:2] - scene["truth"][:2]) < 20
    assert pipeline["second"]["used_flights_confirmed"] == ["stress-0", "stress-1", "stress-2", "stress-3"]


def test_cli_discover_writes_events_and_confirmation(scene, tmp_path):
    from dopplertracker.cli import main

    spec = scene["spectrogram"]
    Spectrogram(spec.times, spec.frequencies, spec.power).save(tmp_path / "spec.npz")
    save_tracks(tmp_path / "traces.npz", scene["all_tracks"], LocalFrame(40.0, -73.0))
    candidate = {**truth_candidate(scene), "flights": ["stress-0", "stress-1"]}
    (tmp_path / "fit.json").write_text(json.dumps({"frame": [40.0, -73.0], "candidates": [candidate]}))
    arguments = [
        "discover",
        str(tmp_path / "spec.npz"),
        str(tmp_path / "fit.json"),
        "--traces",
        str(tmp_path / "traces.npz"),
        "--out",
        str(tmp_path / "found.json"),
        "--confirmation",
        str(tmp_path / "confirmation.json"),
        "--raw-score",
        "--band",
        "80",
        "160",
    ]
    assert main(arguments) == 0
    assert len(load_events([str(tmp_path / "found.json")])) == 2
    assert json.loads((tmp_path / "confirmation.json").read_text())["confirmed_flights"] == [
        "stress-2",
        "stress-3",
    ]
    assert main([*arguments, "--refit"]) == 2


def test_benchmark_guards_reserved_seeds_and_summarises():
    with pytest.raises(ValueError):
        run_discovery_benchmark(seeds=[100])
    with pytest.raises(ValueError):
        run_discovery_benchmark(seeds=[0], final_evaluation=True)
    row = {
        "scenario": "clean",
        "seed": 0,
        "signal_present": True,
        "exception": None,
        "unseen_true_flights": ["stress-2", "stress-3"],
        "first": {"confirmed": ["stress-2", "decoy-1"], "true_positives": 1, "false_positives": 1},
        "second": {"confirmed": ["stress-2"], "true_positives": 1, "false_positives": 0},
        "two_events_error_m": 30.0,
        "joint_error_m": 10.0,
        "four_events_error_m": 5.0,
        "truth_confirmations": {
            "confirmed": ["stress-2", "stress-3"],
            "true_positives": 2,
            "false_positives": 0,
        },
        "wrong": [{"kind": "clock", "confirmed": 0}, {"kind": "position", "confirmed": 1}],
    }
    summary = summarize([row])
    assert summary["first_recall"] == 0.5 and summary["first_precision"] == 0.5
    assert summary["second_precision"] == 1.0
    assert summary["wrong_with_any_confirmation"] == 1
    assert summary["truth_confirmed_mean"] == 2.0


def test_trace_index_prefilters_by_time_and_range(scene, tmp_path):
    save_tracks(tmp_path / "traces.npz", scene["all_tracks"], LocalFrame(40.0, -73.0))
    index = TraceIndex(tmp_path / "traces.npz")
    recording = (0.0, float(scene["spectrogram"].times[-1]))
    assert len(index.select(truth_candidate(scene), recording)) == len(scene["all_tracks"])
    assert index.select({"position": [60000.0, 0.0, 0.0], "clock": scene["clock"]}, recording) == []
    assert index.select(truth_candidate(scene, 7200.0), recording) == []
    assert index.track("stress-2").id == "stress-2"
