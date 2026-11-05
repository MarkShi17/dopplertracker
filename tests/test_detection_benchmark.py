import json

import numpy as np

from dopplertracker.detection import Event
from dopplertracker.detection_benchmark import match_cpa_proposals, run_detection_benchmark


def event(cpa):
    return Event((cpa - 20, cpa + 20), cpa, 100, 80, 1000)


def test_matching_is_one_to_one_and_uses_nearest_proposal():
    result = match_cpa_proposals([98, 102, 200], [event(100), event(201)], tolerance_s=5)
    assert result["matched"] == 2
    assert result["false_proposals"] == 1
    assert result["missed_events"] == 0
    assert sorted(result["errors_s"]) == [1.0, 2.0]


def test_wrong_event_is_not_counted_as_a_match():
    result = match_cpa_proposals([50, 400], [event(100)], tolerance_s=10)
    assert result["matched"] == 0
    assert result["false_proposals"] == 2
    assert result["missed_events"] == 1


def test_matching_maximizes_cardinality_before_minimizing_cpa_error():
    # Greedy closest-pair matching takes 6->4 and strands proposal 0.  The
    # maximum-cardinality assignment is 0->4 and 6->10.
    result = match_cpa_proposals([0, 6], [event(4), event(10)], tolerance_s=4)
    assert result["matched"] == 2
    assert result["false_proposals"] == 0
    assert result["missed_events"] == 0


def test_matching_counts_are_permutation_invariant():
    forward = match_cpa_proposals([0, 6, 30], [event(4), event(10)], tolerance_s=4)
    reversed_order = match_cpa_proposals([30, 6, 0], [event(10), event(4)], tolerance_s=4)
    for key in ("matched", "false_proposals", "missed_events"):
        assert forward[key] == reversed_order[key]
    assert sorted(forward["errors_s"]) == sorted(reversed_order["errors_s"])


def test_noise_only_has_explicit_empty_truth_even_when_generator_carries_events():
    report = run_detection_benchmark(seeds=(100,), scenarios=("noise_only",))
    assert report["trials"]
    assert all(row["truth_event_cpas_s"] == [] for row in report["trials"])
    assert all(row["missed_events"] == 0 for row in report["trials"])


def test_report_is_json_safe_and_detects_a_real_small_scene():
    report = run_detection_benchmark(seeds=(100,), scenarios=("clean",))
    encoded = json.dumps(report, allow_nan=False)
    assert encoded
    assert len(report["trials"]) == 3
    assert {row["method"] for row in report["trials"]} == {
        "raw-one-harmonic",
        "whitened-one-harmonic",
        "whitened-two-harmonics",
    }
    assert all(np.isfinite(row["runtime_s"]) and row["runtime_s"] >= 0 for row in report["trials"])
    # The default run stays legacy-only, but now also scores matched f0 and ranges.
    assert report["detector"]["calibrated_options"] is None
    for row in report["trials"]:
        assert len(row["matched_f0_errors_hz"]) == row["matched"]
        assert len(row["matched_ranges_cover_truth"]) == row["matched"]
        assert row["proposal_z"] == [None] * len(row["proposal_cpas_s"])


def test_matching_reports_the_matched_pairs():
    result = match_cpa_proposals([98, 300, 202], [event(100), event(201)], tolerance_s=5)
    assert sorted(result["pairs"]) == [(0, 0), (2, 1)]
