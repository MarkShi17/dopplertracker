import json

from dopplertracker.stress_benchmark import METHODS, run_stress_benchmark, summarize_trials


def test_failures_stay_in_denominators():
    rows = [
        {
            "method": "m",
            "scenario": "clean",
            "error_m": 2.0,
            "success": True,
            "accepted": True,
            "seconds": 1.0,
            "exception": None,
        },
        {
            "method": "m",
            "scenario": "clean",
            "error_m": None,
            "success": False,
            "accepted": False,
            "seconds": 2.0,
            "exception": "failed",
        },
        {
            "method": "m",
            "scenario": "noise_only",
            "error_m": None,
            "success": True,
            "accepted": True,
            "seconds": 1.0,
            "exception": None,
        },
    ]
    result = summarize_trials(rows)["m"]
    assert result["signal_trials"] == 2
    assert result["within_100m_fraction"] == 0.5
    assert result["exceptions"] == 1
    assert result["false_accepts"] == 1
    assert result["negative_trials"] == 1


def test_stress_runner_exports_all_rows_and_controls(tmp_path):
    report = run_stress_benchmark(
        tmp_path,
        seeds=[1],
        scenarios=["clean", "noise_only"],
        methods=["projected-constant", "projected-linear"],
        split="development",
    )
    assert len(report["trials"]) == 4
    assert (
        report["validation"]
        == "synthetic conditional localization; known flight assignments and seeded event windows"
    )
    assert report["source_hashes"]
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "trials.csv").exists()
    json.dumps(report, allow_nan=False)


def test_method_registry_includes_independent_families():
    assert len(METHODS) >= 20
    assert {
        "brightness",
        "projected-quadratic",
        "spline-linear",
        "kalman-linear",
        "blocked-cv",
        "event-consensus",
    } <= set(METHODS)
