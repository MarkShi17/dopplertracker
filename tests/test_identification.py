import json

import numpy as np

from dopplertracker import identification


def test_grouping_counts_a_wrong_pair_ahead_of_true_pair():
    results = [
        {"flights": ["wrong-a", "wrong-b"], "score": 10},
        {"flights": ["wrong-a", "wrong-b"], "score": 9},
        {"flights": ["true-a", "true-b"], "score": 8},
    ]
    assert [r["score"] for r in identification._grouped(results)] == [10, 8]


def test_report_preserves_missing_true_pair_and_does_not_supply_clock(monkeypatch):
    seen = {}

    def fake_search(spec, events, tracks, points, **kwargs):
        seen.update(kwargs)
        return [
            {
                "flights": ["decoy-0", "decoy-1"],
                "score": 1.0,
                "position": [0.0, 0.0, 0.0],
                "clock": 12345.0,
                "f0": [100, 110],
            }
        ]

    monkeypatch.setattr(identification, "search", fake_search)
    monkeypatch.setattr(identification, "assess_candidate", lambda *a: {"accepted": False})
    report = identification.run_identification(seeds=(7,), distractors=2)
    assert "clock" not in seen
    positives = [t for t in report["trials"] if t["signal_present"]]
    assert all(t["correct_pair_rank"] is None and t["failure"] for t in positives)
    assert json.loads(json.dumps(report)) == report


def test_negative_control_is_excluded_from_positive_denominators(monkeypatch):
    def fake_search(spec, events, tracks, points, **kwargs):
        return [
            {
                "flights": ["decoy-0", "decoy-1"],
                "score": 1.0,
                "position": [0.0, 0.0, 0.0],
                "clock": 12345.0,
                "f0": [100, 110],
            }
        ]

    monkeypatch.setattr(identification, "search", fake_search)
    monkeypatch.setattr(identification, "assess_candidate", lambda *a: {"accepted": True})
    report = identification.run_identification(seeds=(9,), distractors=2)
    negatives = [t for t in report["trials"] if not t["signal_present"]]
    assert all(t["correct_pair_rank"] is None and t["search_error_m"] is None for t in negatives)
    assert all(t["top1"] is None and t["false_accept"] is True and t["failure"] is None for t in negatives)
    for summary in report["summaries"].values():
        assert summary["positive_trials"] == 3
        assert summary["negative_trials"] == 1
        assert summary["false_accepts"] == 1


def test_decoys_are_independently_perturbed():
    scene = identification.stress_scene("clean", 3)
    decoys = identification._distractors(scene["tracks"], 3, np.random.default_rng(4))
    assert len(decoys) == 3
    assert len({tuple(np.round(d.positions[0], 1)) for d in decoys}) == 3
