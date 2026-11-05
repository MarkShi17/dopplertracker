"""Proposal-quality benchmark for the coarse spectral event detector.

Truth CPA times are used only after detection for one-to-one scoring.  In
particular, detector windows are never derived from generator event windows.
"""

import time
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection import detect
from .stress_scenes import stress_scene

METHODS = (
    ("raw-one-harmonic", False, 1),
    ("whitened-one-harmonic", True, 1),
    ("whitened-two-harmonics", True, 2),
)
WINDOW_S = 120.0
STRIDE_S = 60.0
TOP_PROPOSALS = 4
MATCH_TOLERANCE_S = 10.0


def _cpa(value):
    return float(value.cpa if hasattr(value, "cpa") else value)


def match_cpa_proposals(proposals, truth_events, tolerance_s=MATCH_TOLERANCE_S):
    """Match CPA proposals to truth globally, each proposal/event at most once."""
    if not np.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and nonnegative")
    proposed = [_cpa(item) for item in proposals]
    truth = [_cpa(item) for item in truth_events]
    if not proposed or not truth:
        return {
            "matched": 0,
            "false_proposals": len(proposed),
            "missed_events": len(truth),
            "errors_s": [],
        }
    distances = np.abs(np.subtract.outer(proposed, truth))
    valid = distances <= tolerance_s
    # A penalty larger than every possible combination of feasible errors makes
    # assignment maximize valid-pair cardinality first, then minimize error.
    penalty = float(distances[valid].sum() + 1.0) if valid.any() else 1.0
    costs = np.where(valid, distances, penalty)
    proposal_indices, event_indices = linear_sum_assignment(costs)
    errors = [
        float(distances[proposal_index, event_index])
        for proposal_index, event_index in zip(proposal_indices, event_indices)
        if valid[proposal_index, event_index]
    ]
    return {
        "matched": len(errors),
        "false_proposals": len(proposed) - len(errors),
        "missed_events": len(truth) - len(errors),
        "errors_s": errors,
    }


def _metrics(rows):
    matched = sum(row["matched"] for row in rows)
    false = sum(row["false_proposals"] for row in rows)
    missed = sum(row["missed_events"] for row in rows)
    truth_count = matched + missed
    errors = [error for row in rows for error in row["matched_cpa_errors_s"]]
    return {
        "trials": len(rows),
        "truth_events": truth_count,
        "proposals": matched + false,
        "matched": matched,
        "precision": float(matched / (matched + false)) if matched + false else None,
        "recall": float(matched / truth_count) if truth_count else None,
        "false_proposals": false,
        "missed_events": missed,
        "median_cpa_error_s": float(np.median(errors)) if errors else None,
        "median_runtime_s": float(np.median([row["runtime_s"] for row in rows])),
    }


def run_detection_benchmark(
    seeds=(100, 101),
    scenarios=("clean", "low_snr", "broadband_interference", "noise_only"),
):
    """Benchmark actual coarse detector proposals against held-out event CPAs.

    The fixed method settings deliberately have no score/abstention threshold:
    ``top`` merely limits returned ranked proposals.  Consequently proposals in
    ``noise_only`` are counted as false proposals, not discarded.
    """
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            scene = stress_scene(scenario, int(seed))
            # Noise-only scenes retain seeded flight metadata for realism, but
            # have no acoustic events and therefore no evaluator truth events.
            truth_events = (
                []
                if scenario == "noise_only"
                else [f["true_cpa_audio_s"] for f in scene["metadata"]["flights"]]
            )
            whitened = None
            for name, use_whitening, harmonics in METHODS:
                spec = scene["spectrogram"]
                if use_whitening:
                    if whitened is None:
                        whitened = spec.whiten()
                    spec = whitened
                started = time.perf_counter()
                proposals = detect(
                    spec,
                    window=WINDOW_S,
                    stride=STRIDE_S,
                    top=TOP_PROPOSALS,
                    harmonics=harmonics,
                )
                elapsed = time.perf_counter() - started
                matching = match_cpa_proposals(proposals, truth_events)
                rows.append(
                    {
                        "scenario": str(scenario),
                        "seed": int(seed),
                        "method": name,
                        "proposal_cpas_s": [float(event.cpa) for event in proposals],
                        "truth_event_cpas_s": [float(event) for event in truth_events],
                        "matched_cpa_errors_s": matching["errors_s"],
                        "matched": matching["matched"],
                        "false_proposals": matching["false_proposals"],
                        "missed_events": matching["missed_events"],
                        "precision": float(matching["matched"] / len(proposals)) if proposals else None,
                        "recall": (float(matching["matched"] / len(truth_events)) if truth_events else None),
                        "median_cpa_error_s": (
                            float(np.median(matching["errors_s"])) if matching["errors_s"] else None
                        ),
                        "runtime_s": float(elapsed),
                    }
                )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    return {
        "validation": "synthetic stress scenes; detector proposals matched only after detection",
        "generator": "stress_scene independent forward retarded-time generator",
        "seeds": [int(seed) for seed in seeds],
        "scenarios": [str(scenario) for scenario in scenarios],
        "detector": {
            "window_s": WINDOW_S,
            "stride_s": STRIDE_S,
            "top_proposals": TOP_PROPOSALS,
            "match_tolerance_s": MATCH_TOLERANCE_S,
            "abstention_threshold": None,
        },
        "noise_only_policy": "Generator event metadata is excluded from evaluator truth; every proposal is false.",
        "truth_cpa_source": ("Generator-only unjittered heard-CPA metadata; never passed into detection."),
        "summary": {method: _metrics(method_rows) for method, method_rows in grouped.items()},
        "trials": rows,
    }
