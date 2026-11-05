"""Synthetic evaluation of candidate-conditioned flyover discovery.

Each stress scene has four true flights.  The method receives events 1-2 only,
all four tracks and several silent, individually perturbed decoy tracks.  Truth
is used only to score the results.  Development seeds are below 100.
"""

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .advanced import ModelOptions, fit_projected
from .discovery import DiscoveryOptions, RidgeMap, discover, discover_and_refit, icao
from .identification import _distractors
from .stress_scenes import stress_scene

DISCOVERY_SCENARIOS = (
    "clean",
    "curved_accelerating",
    "rpm_drift",
    "wind",
    "adsb_noise",
    "low_snr",
    "gaps",
    "crossing_ridge",
    "broadband_interference",
    "combined_errors",
    "noise_only",
)


def _error(result, truth):
    return float(np.linalg.norm(np.asarray(result["position"])[:2] - truth[:2]))


def _wrong_candidates(scene, rng):
    """Deliberately wrong candidates: position 1-5 km off, clock 30-120 s off, or both."""
    truth, clock = scene["truth"], scene["clock"]
    output = []
    for kind in ("position", "position", "clock", "both"):
        position = truth.copy()
        shift = 0.0
        if kind in ("position", "both"):
            angle, distance = rng.uniform(0, 2 * np.pi), rng.uniform(1000.0, 5000.0)
            position[:2] += distance * np.array([np.cos(angle), np.sin(angle)])
        if kind in ("clock", "both"):
            shift = float(rng.choice((-1, 1)) * rng.uniform(30.0, 120.0))
        output.append(
            {
                "kind": kind,
                "offset_m": float(np.linalg.norm(position[:2] - truth[:2])),
                "clock_offset_s": shift,
                "candidate": {"position": position.tolist(), "clock": clock + shift},
            }
        )
    return output


def _labels(summary, unseen):
    confirmed = summary["confirmed_flights"]
    return {
        "confirmed": confirmed,
        "true_positives": sum(f in unseen for f in confirmed),
        "false_positives": sum(f not in unseen for f in confirmed),
    }


def run_trial(scenario, seed, distractors=8, options=DiscoveryOptions(), model=ModelOptions(degree=1)):
    """One scene: two-event baseline, discovery + joint refit, all-four upper bound, controls."""
    started = time.perf_counter()
    scene = stress_scene(scenario, seed)
    spec, events, tracks, truth = scene["spectrogram"], scene["events"], scene["tracks"], scene["truth"]
    rng = np.random.default_rng(seed * 7919 + 17)
    all_tracks = [*tracks, *_distractors(tracks, distractors, rng)]
    signal = scenario != "noise_only"
    unseen = {t.id for t in tracks[2:]} if signal else set()
    used = [t.id for t in tracks[:2]]
    ridges = RidgeMap(spec, options.background_hz)
    row = {
        "scenario": scenario,
        "seed": int(seed),
        "signal_present": signal,
        "distractors": distractors,
        "unseen_true_flights": sorted(unseen),
        "exception": None,
    }
    initial = {**scene["candidate"], "f0": scene["candidate"]["f0"][:2], "flights": used}
    try:
        two = fit_projected(spec, events[:2], tracks[:2], initial, model)
        row["two_events_error_m"] = _error(two, truth) if signal else None
        pipeline = discover_and_refit(
            spec, events[:2], tracks[:2], all_tracks, two, model, options=options, ridges=ridges
        )
        joint = pipeline["result"]
        row["joint_error_m"] = _error(joint, truth) if signal else None
        row["joint_flights"] = joint["flights"]
        row["dropped_discoveries"] = joint.get("dropped_discoveries", [])
        row["first"] = _labels(pipeline["first"], unseen)
        # At the refined solution: discovered flights re-confirmed plus any new ones.
        reconfirmed = set(pipeline["second"]["used_flights_confirmed"]) - {icao(f) for f in used}
        second_confirmed = sorted(reconfirmed | set(pipeline["second"]["confirmed_flights"]))
        row["second"] = _labels({"confirmed_flights": second_confirmed}, unseen)
        row["second"]["new"] = pipeline["second"]["confirmed_flights"]
        row["first_passes"] = pipeline["first"]["passes"]
    except Exception as exc:  # Failed fits stay in the denominators.
        row["exception"] = f"{type(exc).__name__}: {exc}"
    try:
        four = fit_projected(spec, events, tracks, scene["candidate"], model)
        row["four_events_error_m"] = _error(four, truth) if signal else None
    except Exception as exc:
        row["four_events_exception"] = f"{type(exc).__name__}: {exc}"
    at_truth = discover(
        ridges,
        all_tracks,
        {"position": truth.tolist(), "clock": scene["clock"]},
        options=options,
        used_flights=used,
    )
    row["truth_confirmations"] = _labels(at_truth["summary"], unseen)
    row["truth_z_sum"] = at_truth["summary"]["z_sum"]
    row["truth_passes"] = at_truth["summary"]["passes"]
    row["wrong"] = []
    for wrong in _wrong_candidates(scene, np.random.default_rng(seed * 104729 + 3)):
        report = discover(ridges, all_tracks, wrong["candidate"], options=options, used_flights=used)[
            "summary"
        ]
        row["wrong"].append(
            {
                "kind": wrong["kind"],
                "offset_m": wrong["offset_m"],
                "clock_offset_s": wrong["clock_offset_s"],
                "confirmed": report["confirmed_independent"],
                "confirmed_flights": report["confirmed_flights"],
                "z_sum": report["z_sum"],
                "max_z": max((p["z"] for p in report["passes"] if p["z"] is not None), default=None),
            }
        )
    row["seconds"] = time.perf_counter() - started
    return row


def _job(args):
    return run_trial(*args)


def _fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _quantiles(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "median": None, "p90": None, "max": None}
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(max(values)),
    }


def summarize(rows):
    signal = [r for r in rows if r["signal_present"]]
    noise = [r for r in rows if not r["signal_present"]]
    summary = {
        "scenes": len(rows),
        "signal_scenes": len(signal),
        "exceptions": sum(bool(r["exception"]) for r in rows),
    }
    for stage in ("first", "second"):
        stats = [r[stage] for r in signal if stage in r]
        available = 2 * len(signal)
        tp = sum(s["true_positives"] for s in stats)
        fp = sum(s["false_positives"] for s in stats)
        summary[f"{stage}_recall"] = _fraction(tp, available)
        summary[f"{stage}_precision"] = _fraction(tp, tp + fp)
        summary[f"{stage}_false_positives"] = fp
    for key in ("two_events_error_m", "joint_error_m", "four_events_error_m"):
        summary[key] = _quantiles([r.get(key) for r in signal])
    truth_counts = [r["truth_confirmations"]["true_positives"] for r in signal]
    truth_false = [r["truth_confirmations"]["false_positives"] for r in rows]
    summary["truth_confirmed_mean"] = float(np.mean(truth_counts)) if truth_counts else None
    summary["truth_confirmed_distribution"] = {
        str(k): truth_counts.count(k) for k in sorted(set(truth_counts))
    }
    summary["truth_false_confirmations"] = int(sum(truth_false))
    wrong = [w for r in rows for w in r["wrong"]]
    summary["wrong_candidates"] = len(wrong)
    summary["wrong_confirmed_mean"] = float(np.mean([w["confirmed"] for w in wrong])) if wrong else None
    summary["wrong_with_any_confirmation"] = sum(w["confirmed"] > 0 for w in wrong)
    summary["wrong_confirmed_max"] = max((w["confirmed"] for w in wrong), default=None)
    by_kind = {}
    for kind in ("position", "clock", "both"):
        group = [w for w in wrong if w["kind"] == kind]
        by_kind[kind] = {"n": len(group), "with_any": sum(w["confirmed"] > 0 for w in group)}
    summary["wrong_by_kind"] = by_kind
    summary["noise_only_false_confirmations"] = int(
        sum(len(r.get("first", {}).get("confirmed", [])) for r in noise)
        + sum(r["truth_confirmations"]["false_positives"] for r in noise)
    )
    return summary


def false_pass_z(rows):
    """Every scanned z from silent tracks, wrong candidates excluded (they are summarised separately)."""
    values = []
    for row in rows:
        unseen = set(row["unseen_true_flights"])
        for name in ("first_passes", "truth_passes"):
            for p in row.get(name, []):
                if p["z"] is not None and p["flight"] not in unseen and not p["used_in_fit"]:
                    values.append(p["z"])
    return values


def run_discovery_benchmark(
    directory=None,
    seeds=(0, 1, 2, 3, 4),
    scenarios=DISCOVERY_SCENARIOS,
    distractors=8,
    options=DiscoveryOptions(),
    workers=1,
    progress=None,
):
    """Development benchmark; refuses reserved evaluation seeds (>= 100)."""
    if any(int(s) >= 100 for s in seeds):
        raise ValueError("seeds >= 100 are reserved for final evaluation")
    if not 1 <= workers <= 3:
        raise ValueError("use one to three workers")
    if any(s not in DISCOVERY_SCENARIOS for s in scenarios):
        raise ValueError("unknown discovery scenario")
    jobs = [(scenario, int(seed), distractors, options) for scenario in scenarios for seed in seeds]
    started = time.perf_counter()
    rows = []
    if workers == 1:
        for job in jobs:
            rows.append(_job(job))
            if progress:
                progress(len(rows), len(jobs), rows[-1])
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for future in as_completed([pool.submit(_job, job) for job in jobs]):
                rows.append(future.result())
                if progress:
                    progress(len(rows), len(jobs), rows[-1])
    rows.sort(key=lambda r: (list(scenarios).index(r["scenario"]), r["seed"]))
    report = {
        "validation": "synthetic development scenes; events 1-2 supplied, four true tracks plus silent decoys",
        "seeds": [int(s) for s in seeds],
        "scenarios": list(scenarios),
        "distractors": distractors,
        "options": {k: list(v) if isinstance(v, tuple) else v for k, v in vars(options).items()},
        "summary": summarize(rows),
        "by_scenario": {s: summarize([r for r in rows if r["scenario"] == s]) for s in scenarios},
        "false_pass_z": sorted(false_pass_z(rows), reverse=True)[:50],
        "seconds": time.perf_counter() - started,
        "trials": rows,
    }
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
