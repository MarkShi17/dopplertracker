import numpy as np

from dopplertracker.benchmark import run_benchmark, synthetic_scene
from dopplertracker.refine import refine_candidate


def test_synthetic_ridges_match_independent_generator():
    scene = synthetic_scene("clean", seed=3)
    assert len(scene["events"]) == 3
    assert scene["spectrogram"].power.max() > 5
    assert np.isfinite(scene["truth"]).all()


def test_benchmark_reports_all_methods_and_does_not_claim_real_validation(tmp_path):
    report = run_benchmark(tmp_path, seeds=[2], scenarios=["clean"], grid_extent=400)
    assert report["validation"] == "synthetic; known flight assignments"
    methods = {r["method"] for r in report["trials"]}
    assert {"brightness-grid", "contrast-grid", "robust-ridge", "robust-ridge-drift"} <= methods
    assert all(np.isfinite(r["error_m"]) for r in report["trials"])
    assert (tmp_path / "report.html").exists()


def test_fitting_rpm_drift_reduces_location_bias():
    scene = synthetic_scene("rpm_drift", seed=4)
    candidate = {
        "position": [200.0, -200.0, 0.0],
        "clock": 0.0,
        "f0": [event.f0 for event in scene["events"]],
        "flights": [t.id for t in scene["tracks"]],
    }
    fixed = refine_candidate(scene["spectrogram"], scene["events"], scene["tracks"], candidate)
    drift = refine_candidate(
        scene["spectrogram"], scene["events"], scene["tracks"], candidate, fit_drift=True
    )
    fixed_error = np.linalg.norm(np.array(fixed["position"])[:2] - scene["truth"][:2])
    drift_error = np.linalg.norm(np.array(drift["position"])[:2] - scene["truth"][:2])
    assert drift_error < 15
    assert drift_error < fixed_error * 0.25
