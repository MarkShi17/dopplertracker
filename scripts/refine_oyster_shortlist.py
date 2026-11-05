"""Refine every covered published candidate, separately ranking each method.

This reuses the published regional shortlist; it is not a blind search. Each
method starts from the same brightness-calibrated position/clock/frequencies.
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rerank_oyster_bay as shortlist

from dopplertracker.advanced import ModelOptions, fit_projected
from dopplertracker.evidence import assess_candidate
from dopplertracker.physics import Trajectory
from dopplertracker.refine import refine_spectral
from dopplertracker.terrain import ConstantElevation

METHODS = ("brightness", "spline-linear", "projected-constant")


def trial(row):
    calibration = row["scores"]["brightness"]
    position = shortlist.FRAME.to_local([row["lat"], row["lon"], -30.0])
    tracks = []
    for icao, event in zip(row["flights"], shortlist.EVENTS):
        track = shortlist.choose_track(icao, event, calibration["clock"])
        tracks.append(
            Trajectory(
                track.id,
                track.times + calibration["clock"],
                track.positions,
                track.max_gap,
                track.metadata,
            )
        )
    candidate = {
        "position": position.tolist(),
        "clock": calibration["clock"],
        "f0": calibration["f0"],
        "flights": row["flights"],
    }
    results = []
    for method in METHODS:
        began = time.perf_counter()
        result = {
            "original_rank": row["original_rank"],
            "method": method,
            "flights": row["flights"],
            "result": None,
            "evidence": None,
            "error": None,
        }
        try:
            kwargs = {"frame": shortlist.FRAME, "elevation": ConstantElevation(-30.0)}
            inputs = (shortlist.SPEC, shortlist.EVENTS, tracks, candidate)
            if method == "brightness":
                fit = refine_spectral(*inputs, **kwargs)
            else:
                options = ModelOptions(
                    degree=1 if method == "spline-linear" else 0,
                    smoothing="spline" if method == "spline-linear" else "none",
                )
                fit = fit_projected(*inputs, options, **kwargs)
            result["result"] = fit
            result["evidence"] = assess_candidate(shortlist.SPEC, shortlist.EVENTS, tracks, fit)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["seconds"] = time.perf_counter() - began
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/oyster-bay/reranking/results.json")
    parser.add_argument("--out", default="outputs/oyster-bay/refined-shortlist")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    source = Path(args.input)
    original = json.loads(source.read_text())
    candidates = [row for row in original["trials"] if not row["error"]]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "validation": "Refinement of every usable published shortlist candidate, not a new blind regional search.",
        "methods": METHODS,
        "score_note": "Brightness uses spectral brightness; projected methods use local contrast. Rank within each method only.",
        "height_ellipsoid_m": -30.0,
        "sound_speed_m_s": 343.0,
        "attempted_pairs": len(candidates),
        "excluded_input_pairs": original["failures"],
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(shortlist.__file__), *Path("src/dopplertracker").glob("*.py")]
        },
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=shortlist.initialize) as executor:
        jobs = [executor.submit(trial, row) for row in candidates]
        with (output / "trials.jsonl").open("w") as stream:
            for index, job in enumerate(as_completed(jobs), 1):
                for row in job.result():
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if index % 25 == 0:
                    print(
                        f"{index}/{len(candidates)} pairs refined with all {len(METHODS)} methods", flush=True
                    )
    # Evaluation only, after all fitting. Truth never enters trial().
    shortlist.initialize()
    truth = json.loads(Path("data/oyster-bay/published/provenance.json").read_text())["evaluation_truth"]
    truth_position = shortlist.FRAME.to_local([*truth, 0.0])
    for row in rows:
        if row["result"]:
            row["error_m"] = float(
                np.linalg.norm(np.asarray(row["result"]["position"])[:2] - truth_position[:2])
            )
    summary = {}
    for method in METHODS:
        ranked = sorted(
            [row for row in rows if row["method"] == method and row["result"] is not None],
            key=lambda row: row["result"]["score"],
            reverse=True,
        )
        accepted = [row for row in ranked if row["evidence"] and row["evidence"]["accepted"]]
        summary[method] = {
            "successful_fits": len(ranked),
            "accepted_fits": len(accepted),
            "correct_pair_rank": next(
                (i + 1 for i, row in enumerate(ranked) if row["flights"] == ["a1dfeb", "ace391"]), None
            ),
            "correct_pair_rank_among_accepted": next(
                (i + 1 for i, row in enumerate(accepted) if row["flights"] == ["a1dfeb", "ace391"]), None
            ),
            "top5": ranked[:5],
        }
    rows.sort(key=lambda row: (row["original_rank"], row["method"]))
    report = {"plan": plan, "evaluation_truth": truth, "summary": summary, "trials": rows}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for method, result in summary.items():
        print(method, {k: v for k, v in result.items() if k != "top5"}, flush=True)


if __name__ == "__main__":
    main()
