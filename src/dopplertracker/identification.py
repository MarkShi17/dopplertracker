"""Regional flight-identification benchmark with independently varied decoys.

This is deliberately an *identification* test: the regional search receives all
traces, not the pre-assigned tracks which generated an event.  It is synthetic
and measures conditional ranking only, not real-world flight identification.
"""

from time import perf_counter

import numpy as np

from .evidence import assess_candidate
from .physics import Trajectory
from .search import search
from .stress_scenes import stress_scene

IDENTIFICATION_SCENARIOS = ("clean", "curved_accelerating", "low_snr", "noise_only")


def _distractors(tracks, count, rng):
    """Make individually perturbed tracks, avoiding a pair-wide rigid transform."""
    output = []
    for n in range(count):
        base = tracks[n % len(tracks)]
        time = base.times
        midpoint = time.mean()
        # Each decoy gets its own clock offset, speed scaling, heading, altitude
        # and non-rigid position perturbation.
        scale = rng.uniform(0.78, 1.25)
        shift = rng.choice((-1, 1)) * rng.uniform(3.0, 15.0)
        angle = rng.uniform(-1.4, 1.4)
        c, s = np.cos(angle), np.sin(angle)
        xy = base.positions[:, :2] - base.positions[:, :2].mean(axis=0)
        rotated = xy @ np.array([[c, -s], [s, c]]) * scale
        center = base.positions[:, :2].mean(axis=0) + rng.uniform(-600, 600, 2)
        pos = np.empty_like(base.positions)
        pos[:, :2] = center + rotated + rng.normal(0, 8, rotated.shape)
        pos[:, 2] = base.positions[:, 2] + rng.uniform(-900, 900) + rng.normal(0, 18, len(time))
        decoy_time = midpoint + (time - midpoint) / scale + shift
        output.append(Trajectory(f"decoy-{n}", decoy_time, pos, max_gap=base.max_gap))
    return output


def _grouped(results):
    groups = []
    seen = set()
    for result in results:
        identity = tuple(result["flights"])
        if identity not in seen:
            seen.add(identity)
            groups.append(result)
    return groups


def _trial(scene, scenario, seed, distractors, method):
    events = scene["events"][:2]
    true_tracks = scene["tracks"][:2]
    rng = np.random.default_rng(seed * 1009 + len(scenario) * 31)
    all_tracks = [*true_tracks, *_distractors(scene["tracks"], distractors, rng)]
    axis = np.linspace(-400.0, 400.0, 5)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    started = perf_counter()
    signal_present = scenario != "noise_only"
    try:
        # Do not pass the known recording clock: two event timing differences
        # anchor it from the candidate flight pair.
        results = search(
            scene["spectrogram"],
            events,
            all_tracks,
            points,
            method=method,
            offset_tolerance=14.0,
            shift_radius=3.0,
            shift_steps=3,
            max_candidates=30_000,
        )
        groups = _grouped(results)
        truth_ids = tuple(track.id for track in true_tracks)
        rank = (
            next((i + 1 for i, result in enumerate(groups) if tuple(result["flights"]) == truth_ids), None)
            if signal_present
            else None
        )
        winner = groups[0] if groups else None
        correct = (
            next((r for r in groups if tuple(r["flights"]) == truth_ids), None) if signal_present else None
        )
        margin = None if len(groups) < 2 else float(groups[0]["score"] - groups[1]["score"])
        assessment = None
        if winner is not None:
            by_id = {track.id: track for track in all_tracks}
            assessment = assess_candidate(
                scene["spectrogram"], events, [by_id[i] for i in winner["flights"]], winner
            )
        error = (
            None
            if winner is None or not signal_present
            else float(np.linalg.norm(np.asarray(winner["position"])[:2] - scene["truth"][:2]))
        )
        accepted = None if assessment is None else bool(assessment["accepted"])
        return dict(
            scenario=scenario,
            seed=int(seed),
            method=method,
            signal_present=signal_present,
            correct_pair_rank=rank,
            top1=(rank == 1) if signal_present else None,
            top3=(rank is not None and rank <= 3) if signal_present else None,
            search_error_m=error,
            runner_up_margin=margin,
            runtime_s=float(perf_counter() - started),
            candidates=len(groups),
            winner_flights=None if winner is None else winner["flights"],
            correct_pair_score=None if correct is None else float(correct["score"]),
            accepted=accepted,
            false_accept=accepted if not signal_present else None,
            failure=None if not signal_present or rank is not None else "true_pair_not_returned",
        )
    except Exception as exc:  # Benchmarks preserve failed trials in their denominator.
        return dict(
            scenario=scenario,
            seed=int(seed),
            method=method,
            signal_present=signal_present,
            correct_pair_rank=None,
            top1=None if not signal_present else False,
            top3=None if not signal_present else False,
            search_error_m=None,
            runner_up_margin=None,
            runtime_s=float(perf_counter() - started),
            candidates=0,
            winner_flights=None,
            correct_pair_score=None,
            accepted=None,
            false_accept=None,
            failure=f"{type(exc).__name__}: {exc}",
        )


def run_identification(seeds=(100, 101, 102), distractors=6):
    """Run reproducible regional searches and return a JSON-safe report.

    The report includes every attempted trial, including cases where the true
    pair is pruned or search fails.  A positive score in ``noise_only`` is not
    interpreted as proof; its frozen-candidate evidence decision is reported as
    ``false_accept``.
    """
    if distractors < 0 or not all(isinstance(seed, (int, np.integer)) for seed in seeds):
        raise ValueError("distractors must be nonnegative and seeds must be integers")
    trials = []
    for scenario in IDENTIFICATION_SCENARIOS:
        for seed in seeds:
            scene = stress_scene(scenario, int(seed))
            for method in ("brightness", "contrast"):
                trials.append(_trial(scene, scenario, int(seed), distractors, method))
    summaries = {}
    for method in ("brightness", "contrast"):
        rows = [row for row in trials if row["method"] == method]
        positive = [row for row in rows if row["signal_present"]]
        negative = [row for row in rows if not row["signal_present"]]
        summaries[method] = dict(
            positive_trials=len(positive),
            top1_fraction=float(sum(row["top1"] is True for row in positive) / len(positive))
            if positive
            else None,
            top3_fraction=float(sum(row["top3"] is True for row in positive) / len(positive))
            if positive
            else None,
            negative_trials=len(negative),
            false_accepts=sum(row["false_accept"] is True for row in negative),
            exceptions=sum(
                row["failure"] is not None and row["failure"] != "true_pair_not_returned" for row in rows
            ),
        )
    return dict(
        validation="synthetic regional ranking; known truth used only for evaluation",
        scenarios=list(IDENTIFICATION_SCENARIOS),
        seeds=[int(s) for s in seeds],
        distractors=int(distractors),
        trials=trials,
        summaries=summaries,
        failures=sum(t["failure"] is not None for t in trials),
    )
