"""Regional search on the Oyster Bay recording: the existing search vs the fast path.

Stages, run one at a time (each writes JSON under --out):

  synthetic   search vs fast_search on decoy scenes (development seeds 0-9 only)
  baseline    existing ``search`` on a square box, the reference for equivalence
  detect      calibrated flyover detector on the whole hour, no hints
  fast        ``fast_search`` on a box or the full region for one event set:
                published   data/oyster-bay/events.json
                detected    the detections standing for the two published flyovers
                            (highest z within 20 s of each published CPA)
                auto        detections chosen by rank (--subset), no published input
  compare     baseline vs fast (capped and --per-combination 0 under all/) per box
  rank-auto   common score over every auto event set (rule below)
  refine      spline-linear and fused-velocity refinement of the top auto combinations
  evaluate    reads truth and every finished result; reports distances and ranks

Search and refinement never see the camera position or the correct flights.
Boxes are centred on the published coarse search winner.  Truth is read only
by ``evaluate``.

Fully automatic rule (``auto`` / ``rank-auto``): take the detector's accepted
detections (z >= its default min_z), at most the top three by z; search every
pair and the triple over the full region.  Every candidate must then account
for every detection: at its position and fitted clock, a detection outside its
event set is scored with the best flight whose closest approach passes that
detection's distance, speed and timing (+/- offset tolerance) filters, or 0 if
none does.  Combinations from all event sets are ranked by the mean of these
per-detection brightness scores.
"""

import argparse
import itertools
import json
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dopplertracker.advanced import ModelOptions, fit_projected
from dopplertracker.data import load_tracks, subset_runs
from dopplertracker.detection import Event, FlyoverOptions, detect_flyovers, load_events, save_events
from dopplertracker.evidence import assess_candidate
from dopplertracker.geo import LocalFrame, ground_grid
from dopplertracker.identification import _distractors
from dopplertracker.physics import Atmosphere, Trajectory, closest_approaches
from dopplertracker.scoring import joint_score, subsonic
from dopplertracker.search import fast_search, search
from dopplertracker.signal import Spectrogram
from dopplertracker.stress_scenes import stress_scene
from dopplertracker.terrain import ConstantElevation

DATA = Path("data/oyster-bay")
FULL_REGION = (40.4, 41.3, -74.4, -72.6)
HEIGHT = -30.0
OFFSET_TOLERANCE = 12.0
REFINERS = {
    "spline-linear": ModelOptions(degree=1, smoothing="spline"),
    "fused-velocity": ModelOptions(degree=1, velocity="fused"),
}


def peak_gb():
    # macOS reports bytes, Linux kilobytes.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9


def guard_memory(limit_gb):
    """Exit (code 3) as soon as resident memory exceeds ``limit_gb``; shared machines."""

    def watch():
        pid = str(os.getpid())
        while True:
            output = subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True).stdout
            if int(output) * 1024 > limit_gb * 1e9:
                print(f"Resident memory above {limit_gb} GB; stopping.", flush=True)
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


def write(path, payload):
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    print(f"-> {path}", flush=True)


def region(name):
    if name == "full":
        return FULL_REGION
    published = json.loads((DATA / "published/verified_published_baseline.json").read_text())
    row = published["oyster_bay_search_best_per_pair.csv"]["highest_scoring_row"]
    lat, lon = float(row["lat"]), float(row["lon"])
    half = float(name) * 500.0
    dlat, dlon = half / 111320.0, half / (111320.0 * np.cos(np.deg2rad(lat)))
    return (lat - dlat, lat + dlat, lon - dlon, lon + dlon)


def jets():
    return json.loads(Path("data/aircraft-types/jets.json").read_text())


def spectrogram():
    return Spectrogram.load(DATA / "spectrogram.npz").whiten()


def compact(results, limit=None):
    keys = ("position", "clock", "score", "flights", "f0", "event_scores")
    return [{k: r[k] for k in keys} for r in results[:limit]]


def by_combination(results):
    """Best result per flight combination, best first."""
    best = {}
    for result in results:
        best.setdefault(tuple(result["flights"]), result)
    return list(best.values())


# ----- synthetic --------------------------------------------------------------------------


def run_synthetic(args):
    rows = []
    axis = np.linspace(-800.0, 800.0, 9)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    common = dict(offset_tolerance=14.0, shift_radius=3.0, shift_steps=3)
    for scenario in ("clean", "curved_accelerating", "low_snr"):
        for seed in range(10):
            scene = stress_scene(scenario, seed)
            rng = np.random.default_rng(seed * 1009 + len(scenario) * 31)
            tracks = [*scene["tracks"][:2], *_distractors(scene["tracks"], args.decoys, rng)]
            events = scene["events"][:2]
            started = time.perf_counter()
            slow = search(scene["spectrogram"], events, tracks, points, max_candidates=30_000, **common)
            slow_seconds = time.perf_counter() - started
            row = dict(scenario=scenario, seed=seed, search_seconds=slow_seconds)
            for cap in (None, 48, 8):
                started = time.perf_counter()
                fast, summary = fast_search(
                    scene["spectrogram"], events, tracks, points, per_combination=cap, **common
                )
                row[f"fast{cap or 'All'}"] = dict(
                    seconds=time.perf_counter() - started,
                    same_winner=bool(slow and fast and slow[0]["flights"] == fast[0]["flights"]),
                    winner=fast[0]["flights"] if fast else None,
                    position_difference_m=float(
                        np.linalg.norm(np.subtract(fast[0]["position"], slow[0]["position"])[:2])
                    )
                    if slow and fast
                    else None,
                    score_difference=float(fast[0]["score"] - slow[0]["score"]) if slow and fast else None,
                    combinations_search=len(by_combination(slow)),
                    combinations_fast=len(by_combination(fast)),
                    scored=summary["scored"],
                    candidates=summary["candidates"],
                )
            row["search_winner"] = slow[0]["flights"] if slow else None
            # Evaluation only: generator truth, after both searches finished.
            truth_ids = [t.id for t in scene["tracks"][:2]]
            row["search_correct"] = row["search_winner"] == truth_ids
            row["fast48_correct"] = row["fast48"]["winner"] == truth_ids
            rows.append(row)
            print(
                scenario,
                seed,
                f"search {slow_seconds:.1f}s",
                {
                    k: (row[k]["same_winner"], row[k]["position_difference_m"])
                    for k in ("fastAll", "fast48", "fast8")
                },
                flush=True,
            )
    summary = {
        cap: dict(
            same_winner=sum(r[cap]["same_winner"] for r in rows),
            same_position=sum(r[cap]["position_difference_m"] == 0 for r in rows),
            trials=len(rows),
        )
        for cap in ("fastAll", "fast48", "fast8")
    }
    summary["search_correct"] = sum(r["search_correct"] for r in rows)
    summary["fast48_correct"] = sum(r["fast48_correct"] for r in rows)
    print(summary)
    write(Path(args.out) / "synthetic.json", dict(rows=rows, summary=summary, decoys=args.decoys))


# ----- real recording ----------------------------------------------------------------------


def grid_points(frame, name, spacing):
    return ground_grid(frame, region(name), spacing, ConstantElevation(HEIGHT))


def run_baseline(args):
    spec = spectrogram()
    tracks, frame = load_tracks(DATA / args.traces, exclude_types=jets())
    points = grid_points(frame, args.region, args.spacing)
    events = load_events([str(DATA / "events.json")])
    started = time.perf_counter()
    results = search(spec, events, tracks, points, Atmosphere(), "brightness")
    seconds = time.perf_counter() - started
    report = dict(
        stage="baseline",
        region=region(args.region),
        spacing=args.spacing,
        points=len(points),
        tracks=len(tracks),
        seconds=seconds,
        peak_rss_gb=peak_gb(),
        combinations=[compact([r])[0] for r in by_combination(results)],
        results=compact(results),
    )
    print(f"{len(results)} results in {seconds:.0f} s, peak {peak_gb():.2f} GB", flush=True)
    write(Path(args.out) / f"baseline-{args.region}km.json", report)


def run_detect(args):
    spec = spectrogram()
    started = time.perf_counter()
    detections = detect_flyovers(spec, FlyoverOptions())
    seconds = time.perf_counter() - started
    save_events(detections, Path(args.out) / "detections.json")
    for rank, event in enumerate(detections):
        print(f"#{rank} CPA {event.cpa:.1f}s f0 {event.f0:.2f}Hz z {event.z:.1f}", flush=True)
    write(
        Path(args.out) / "detect-run.json",
        dict(seconds=seconds, peak_rss_gb=peak_gb(), count=len(detections)),
    )


def detections(args):
    found = load_events([str(Path(args.out) / "detections.json")])
    return sorted(found, key=lambda e: e.z, reverse=True)


def event_set(args):
    if args.events == "published":
        return load_events([str(DATA / "events.json")]), "published"
    found = detections(args)
    if args.events == "detected":
        published = load_events([str(DATA / "events.json")])
        chosen = []
        for reference in published:
            nearby = [e for e in found if abs(e.cpa - reference.cpa) <= 20.0]
            chosen.append(max(nearby, key=lambda e: e.z))
        return chosen, "detected"
    subset = sorted(args.subset)
    return [found[i] for i in subset], "auto-" + "-".join(map(str, subset))


def run_fast(args):
    events, label = event_set(args)
    spec = spectrogram()
    loading = time.perf_counter()
    tracks, frame = load_tracks(DATA / args.traces, exclude_types=jets())
    points = grid_points(frame, args.region, args.spacing)
    loading = time.perf_counter() - loading
    print(
        f"{len(points)} points, {len(tracks)} segments, events {[round(e.cpa, 1) for e in events]}",
        flush=True,
    )
    last = [0.0]

    def progress(phase, i, n):
        now = time.perf_counter()
        if now - last[0] > 30:
            last[0] = now
            print(f"  {phase} {i}" + (f"/{n}" if n else ""), f"peak {peak_gb():.2f} GB", flush=True)

    results, summary = fast_search(
        spec,
        events,
        tracks,
        points,
        Atmosphere(),
        "brightness",
        offset_tolerance=OFFSET_TOLERANCE,
        per_combination=args.per_combination or None,
        progress=progress,
    )
    combinations = by_combination(results)
    summary.update(
        load_seconds=loading,
        peak_rss_gb=peak_gb(),
        positive_combinations=len(combinations),
        margin_to_runner_up=combinations[0]["score"] - combinations[1]["score"]
        if len(combinations) > 1
        else None,
    )
    print(json.dumps(summary, indent=1), flush=True)
    for result in combinations[:5]:
        print(f"  {result['score']:.3f} {result['flights']}", flush=True)
    report = dict(
        stage="fast",
        label=label,
        region=region(args.region),
        spacing=args.spacing,
        events=[asdict(e) for e in events],
        summary=summary,
        combinations=compact(combinations),
        results=compact(results, 5000),
    )
    write(Path(args.out) / f"fast-{args.region}-{label}.json", report)


def nearby_passes(tracks, observer, event, clock):
    """Tracks whose closest approach at ``observer`` passes the event's filters near its time."""
    heard_at = clock + event.cpa
    radius = event.distance_range[1]
    chosen = []
    for track in tracks:
        if track.times[0] > heard_at + 60 or track.times[-1] < heard_at - 120:
            continue
        window = (track.times > heard_at - 120) & (track.times < heard_at + 60)
        if (
            not window.any()
            or np.min(np.linalg.norm(track.positions[window, :2] - observer[:2], axis=1)) > radius
        ):
            continue
        if any(
            abs(heard - heard_at) <= OFFSET_TOLERANCE
            and event.distance_range[0] <= distance <= event.distance_range[1]
            and event.speed_range[0] <= speed <= event.speed_range[1]
            for heard, distance, speed in closest_approaches(track, observer)
        ):
            chosen.append(track)
    return chosen


def run_rank_auto(args):
    found = detections(args)[:3]
    spec = spectrogram()
    tracks, frame = load_tracks(DATA / args.traces, exclude_types=jets())
    rows = []
    for size in (2, 3):
        for subset in itertools.combinations(range(len(found)), size):
            label = "auto-" + "-".join(map(str, subset))
            path = Path(args.out) / f"fast-{args.region}-{label}.json"
            report = json.loads(path.read_text())
            # Search results list events (and flights) in closest-approach order.
            in_time = sorted(subset, key=lambda i: found[i].cpa)
            for candidate in report["combinations"][: args.top]:
                observer = np.asarray(candidate["position"])
                per_detection, flights = [], []
                for index, event in enumerate(found):
                    if index in subset:
                        per_detection.append(candidate["event_scores"][in_time.index(index)])
                        flights.append(candidate["flights"][in_time.index(index)])
                        continue
                    best, best_id = 0.0, None
                    for track in nearby_passes(tracks, observer, event, candidate["clock"]):
                        if not subsonic(track):
                            continue
                        score = joint_score(
                            spec,
                            [event],
                            [track],
                            observer,
                            candidate["clock"],
                            shift_radius=0,
                            shift_steps=1,
                        )[0]
                        if score > best:
                            best, best_id = score, track.id
                    per_detection.append(best)
                    flights.append(best_id)
                rows.append(
                    dict(
                        label=label,
                        subset=list(subset),
                        common_score=float(np.mean(per_detection)),
                        per_detection=per_detection,
                        flights_by_detection=flights,
                        search_score=candidate["score"],
                        candidate=candidate,
                    )
                )
                print(label, f"{rows[-1]['common_score']:.3f}", flights, flush=True)
    rows.sort(key=lambda r: r["common_score"], reverse=True)
    write(
        Path(args.out) / f"rank-auto-{args.region}.json",
        dict(detections=[asdict(e) for e in found], rule=__doc__.split("Fully automatic rule")[1], rows=rows),
    )


def load_flights(path, ids):
    """Only the named segments, with reported velocities (bounded memory)."""
    with np.load(path, allow_pickle=False) as z:
        frame = LocalFrame(*z["frame"])
        names = set(z.files)
        tracks = {
            item["id"]: Trajectory(
                item["id"],
                z[f"t{i}"],
                z[f"p{i}"],
                item.get("max_gap", 30.0),
                item,
                velocities=z[f"v{i}"] if f"v{i}" in names else None,
            )
            for i, item in enumerate(json.loads(str(z["metadata"])))
            if item["id"] in ids
        }
    return tracks, frame


def cut(track, clock, event):
    """The segment within +/-120 s of the event window, as in scripts/compare_oyster_bay.py."""
    mask = (track.times >= clock + event.window[0] - 120) & (track.times <= clock + event.window[1] + 120)
    metadata = dict(track.metadata)
    for key in ("position_source_runs", "vertical_rate_source_runs"):
        if key in metadata:
            metadata[key] = subset_runs(metadata[key], mask)
    velocities = None if track.velocities is None else track.velocities[mask]
    return Trajectory(track.id, track.times[mask], track.positions[mask], track.max_gap, metadata, velocities)


def run_refine(args):
    ranking = json.loads((Path(args.out) / f"rank-auto-{args.region}.json").read_text())
    found = [Event(**e) for e in ranking["detections"]]
    chosen, seen = [], set()
    for row in ranking["rows"]:
        key = (row["label"], tuple(row["candidate"]["flights"]))
        if key not in seen:
            seen.add(key)
            chosen.append(row)
        if len(chosen) == args.top:
            break
    ids = {f for row in chosen for f in row["candidate"]["flights"]}
    flights, frame = load_flights(DATA / "traces-velocity.npz", ids)
    spec = spectrogram()
    atmosphere, elevation = Atmosphere(), ConstantElevation(HEIGHT)
    fits = []
    for rank, row in enumerate(chosen, 1):
        events = sorted((found[i] for i in row["subset"]), key=lambda e: e.cpa)
        candidate = row["candidate"]
        tracks = [cut(flights[f], candidate["clock"], e) for f, e in zip(candidate["flights"], events)]
        # Shared clock and per-event frequency calibration at the search position, as in
        # scripts/compare_oyster_bay.py.
        score, clock, f0, _ = joint_score(
            spec,
            events,
            tracks,
            np.asarray(candidate["position"]),
            candidate["clock"],
            atmosphere=atmosphere,
            shift_radius=4,
            shift_steps=17,
            frequency_radius=12,
            frequency_steps=121,
        )
        start = dict(
            position=candidate["position"], clock=clock, f0=f0, flights=candidate["flights"], score=score
        )
        for name, options in REFINERS.items():
            started = time.perf_counter()
            entry = dict(rank_auto=rank, label=row["label"], flights=candidate["flights"], refiner=name)
            try:
                result = fit_projected(
                    spec, events, tracks, start, options, atmosphere, frame=frame, elevation=elevation
                )
                result["evidence"] = assess_candidate(spec, events, tracks, result, atmosphere)
                entry.update(
                    result={
                        k: result[k] for k in ("position", "clock", "score", "event_scores", "f0", "success")
                    }
                )
                entry["evidence_accepted"] = bool(result["evidence"]["accepted"])
            except (ValueError, np.linalg.LinAlgError) as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["seconds"] = time.perf_counter() - started
            fits.append(entry)
            print(
                rank,
                row["label"],
                name,
                entry.get("result", {}).get("score"),
                entry.get("error", ""),
                flush=True,
            )
    ranked = {
        name: sorted(
            (f for f in fits if f["refiner"] == name and "result" in f),
            key=lambda f: f["result"]["score"],
            reverse=True,
        )
        for name in REFINERS
    }
    write(
        Path(args.out) / f"refine-{args.region}.json", dict(fits=fits, ranked=ranked, peak_rss_gb=peak_gb())
    )


def run_compare(args):
    """Existing search vs fast path on the same box: no truth needed."""
    rows = {}
    for size in ("6", "12"):
        baseline = json.loads((Path(args.out) / f"baseline-{size}km.json").read_text())
        reference = by_combination(baseline["results"])
        for label, path in (
            ("capped", Path(args.out) / f"fast-{size}-published.json"),
            ("exhaustive", Path(args.out) / "all" / f"fast-{size}-published.json"),
        ):
            if not path.exists():
                continue
            fast = json.loads(path.read_text())
            combinations = {tuple(c["flights"]): c for c in fast["combinations"]}
            table = []
            for rank, result in enumerate(reference, 1):
                other = combinations.get(tuple(result["flights"]))
                table.append(
                    dict(
                        rank=rank,
                        flights=[f.split(":")[0] for f in result["flights"]],
                        search_score=result["score"],
                        fast_score=None if other is None else other["score"],
                        fast_rank=None
                        if other is None
                        else 1 + sorted(combinations.values(), key=lambda c: -c["score"]).index(other),
                        best_position_difference_m=None
                        if other is None
                        else float(np.linalg.norm(np.subtract(result["position"], other["position"])[:2])),
                    )
                )
            rows[f"{size}km-{label}"] = dict(
                search_seconds=baseline["seconds"],
                search_peak_rss_gb=baseline["peak_rss_gb"],
                search_results=len(baseline["results"]),
                fast_seconds=fast["summary"]["seconds"],
                fast_peak_rss_gb=fast["summary"]["peak_rss_gb"],
                fast_scored=fast["summary"]["scored"],
                fast_candidates=fast["summary"]["candidates"],
                same_winner=bool(reference)
                and bool(fast["combinations"])
                and reference[0]["flights"] == fast["combinations"][0]["flights"],
                search_combinations=len(reference),
                fast_combinations=len(combinations),
                fast_only_combinations=[
                    [f.split(":")[0] for f in k]
                    for k in combinations
                    if k not in {tuple(r["flights"]) for r in reference}
                ],
                table=table,
            )
            print(
                size, label, json.dumps({k: v for k, v in rows[f"{size}km-{label}"].items() if k != "table"})
            )
            for line in table:
                print("   ", line)
    write(Path(args.out) / "compare.json", rows)


# ----- evaluation (truth read here only) ------------------------------------------------------


def run_evaluate(args):
    truth = json.loads((DATA / "published/provenance.json").read_text())["evaluation_truth"]
    with np.load(DATA / "traces.npz", allow_pickle=False) as z:
        frame = LocalFrame(*z["frame"])
    target = frame.to_local([*truth, HEIGHT])
    correct = ("a1dfeb", "ace391")

    def icao(flights):
        return tuple(f.split(":")[0] if f else None for f in flights)

    def distance(position):
        return float(np.linalg.norm(np.asarray(position)[:2] - target[:2]))

    report = {}
    for path in sorted(Path(args.out).glob("*.json")):
        name = path.stem
        if not (name.startswith("fast-") or name.startswith("baseline-")):
            continue
        payload = json.loads(path.read_text())
        combinations = payload.get("combinations") or by_combination(payload["results"])
        rank = next(
            (i + 1 for i, c in enumerate(combinations) if set(correct) <= set(icao(c["flights"]))), None
        )
        entry = dict(
            seconds=payload.get("summary", {}).get("seconds", payload.get("seconds")),
            peak_rss_gb=payload.get("summary", {}).get("peak_rss_gb", payload.get("peak_rss_gb")),
            candidates=payload.get("summary", {}).get("candidates"),
            scored=payload.get("summary", {}).get("scored", len(payload.get("results", []))),
            combinations=len(combinations),
            top_flights=icao(combinations[0]["flights"]) if combinations else None,
            true_pair_rank=rank,
            margin_to_runner_up=combinations[0]["score"] - combinations[1]["score"]
            if len(combinations) > 1
            else None,
            top_error_m=distance(combinations[0]["position"]) if combinations else None,
            true_pair_error_m=distance(combinations[rank - 1]["position"]) if rank else None,
            true_pair_score=combinations[rank - 1]["score"] if rank else None,
        )
        report[name] = entry
        print(name, json.dumps(entry), flush=True)
    for name in ("rank-auto-full", "refine-full"):
        path = Path(args.out) / f"{name}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if name.startswith("rank"):
            rows = payload["rows"]
            report[name] = [
                dict(
                    label=r["label"],
                    common_score=r["common_score"],
                    flights=icao(r["flights_by_detection"]),
                    correct=set(correct) <= set(icao(r["candidate"]["flights"])),
                    error_m=distance(r["candidate"]["position"]),
                )
                for r in rows[:10]
            ]
        else:
            report[name] = {
                refiner: [
                    dict(
                        label=f["label"],
                        flights=icao(f["flights"]),
                        score=f["result"]["score"],
                        correct=set(correct) <= set(icao(f["flights"])),
                        error_m=distance(f["result"]["position"]),
                        evidence_accepted=f.get("evidence_accepted"),
                    )
                    for f in fits
                ]
                for refiner, fits in payload["ranked"].items()
            }
        print(name, json.dumps(report[name], indent=1), flush=True)
    write(Path(args.out) / "evaluation.json", dict(truth=truth, correct_flights=correct, report=report))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "stage",
        choices=["synthetic", "baseline", "detect", "fast", "compare", "rank-auto", "refine", "evaluate"],
    )
    parser.add_argument(
        "--region", default="6", help="Box size in km centred on the published winner, or full"
    )
    parser.add_argument("--spacing", type=float, default=200.0)
    parser.add_argument("--traces", default="traces.npz")
    parser.add_argument("--events", choices=["published", "detected", "auto"], default="published")
    parser.add_argument("--subset", nargs="+", type=int, default=[0, 1], help="Detection ranks by z (auto)")
    parser.add_argument("--per-combination", type=int, default=48)
    parser.add_argument(
        "--top", type=int, default=10, help="Combinations per set (rank-auto) or fits (refine)"
    )
    parser.add_argument("--decoys", type=int, default=6)
    parser.add_argument("--out", default="outputs/oyster-bay/search")
    parser.add_argument(
        "--max-rss-gb", type=float, default=2.4, help="Stop above this resident memory; 0 off"
    )
    args = parser.parse_args()
    if args.max_rss_gb:
        guard_memory(args.max_rss_gb)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    stages = {
        "synthetic": run_synthetic,
        "baseline": run_baseline,
        "detect": run_detect,
        "fast": run_fast,
        "rank-auto": run_rank_auto,
        "refine": run_refine,
        "compare": run_compare,
        "evaluate": run_evaluate,
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
