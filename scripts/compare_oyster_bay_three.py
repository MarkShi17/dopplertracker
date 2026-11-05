"""Oyster Bay with a third flyover: does it stabilise the location?

Same protocol and cells as scripts/compare_oyster_bay_combined.py (whose numbers
are unchanged): one shared start from the published coarse winner, joint clock /
source-frequency calibration at that start once per atmosphere and event subset,
then the eight-cell factorial (atmosphere x trajectory x measurement) plus the two
variants of the all-new cell.  Each demodulation cell is also run with the hum
lines removed from the audio instead of masked (``ModelOptions.hum``).

The third flyover is a63e21 (a Cessna Caravan heard near 782 s).  Two event sets,
built without the truth:

  published  events.json (a1dfeb, ace391) plus the discovered a63e21 event: the
             relaxed-range (12 km) discovery scan of a63e21 at the saved
             two-flyover spline-linear solution, as in scripts/discover_oyster_bay.py.
             Window = scan window (heard CPA +/- 75 s), f0 = best source frequency of
             the 60-300 Hz scan, two harmonics (DiscoveryOptions defaults).
  detector   the calibrated detector on the whole hour (default FlyoverOptions, no
             hints).  Each flight gets the detection whose CPA lies within 20 s of
             the CPA heard at the saved solution.  Window = CPA +/- 60 s, two harmonics.

Subsets: known (a1dfeb + ace391), all three, a63e21 + a1dfeb and a63e21 + ace391.
Truth is read only after every fit, and the truth-free spread summary, are saved.
"""

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_oyster_bay_atmosphere import CAMERA_ABOVE_MSL_M, GEOID_M  # noqa: E402
from compare_oyster_bay_combined import cells as combined_cells  # noqa: E402
from compare_oyster_bay_velocity import load_flights, select_tracks  # noqa: E402

from dopplertracker.advanced import fit_projected, prepare_track  # noqa: E402
from dopplertracker.atmosphere import WeatherSeries  # noqa: E402
from dopplertracker.demod import Audio  # noqa: E402
from dopplertracker.detection import FlyoverOptions, detect_flyovers, load_events  # noqa: E402
from dopplertracker.discovery import DiscoveryOptions, discover, icao  # noqa: E402
from dopplertracker.evidence import assess_candidate  # noqa: E402
from dopplertracker.physics import Atmosphere, closest_approaches, ideal_curve, predict  # noqa: E402
from dopplertracker.scoring import joint_score  # noqa: E402
from dopplertracker.signal import Spectrogram, decode_samples  # noqa: E402
from dopplertracker.terrain import ConstantElevation  # noqa: E402

KNOWN = ("a1dfeb", "ace391")
THIRD = "a63e21"
SUBSETS = {
    "known": ("a1dfeb", "ace391"),
    "three": ("a63e21", "a1dfeb", "ace391"),
    "1+3": ("a63e21", "a1dfeb"),
    "2+3": ("a63e21", "ace391"),
}
MATCH_SECONDS = 20.0
# Range-limit sensitivity of scripts/discover_oyster_bay.py, fixed a priori.
RELAXED = DiscoveryOptions(max_distance=12000.0, ground_range=12000.0)


def cells(hum):
    """The combined script's cells, then each demodulation cell with ``hum`` line removal."""
    out = combined_cells()
    for name, key, options in list(out):
        if options.ridge == "demod":
            out.append((name.replace("demod", f"demod-{hum}", 1), key, replace(options, hum=hum)))
    return out


def heard_passes(tracks, candidate, recording, atmosphere=Atmosphere(343.0)):
    """(heard recording s, slant m, speed m/s, height above observer m, flight) of every CPA in the recording."""
    observer = np.asarray(candidate["position"], dtype=float)
    out = []
    for track in tracks:
        for heard, distance, speed in closest_approaches(track, observer, atmosphere):
            heard_s = heard - float(candidate["clock"])
            if recording[0] <= heard_s <= recording[1]:
                emission = heard - distance / atmosphere.c
                position, _ = track.at(np.clip(emission, track.times[0], track.times[-1]))
                out.append((float(heard_s), distance, speed, float(position[2] - observer[2]), track.id))
    return out


def build_event_sets(spec, published_events, saved, positions_path, traces_path):
    """Discovery and detector event sets keyed by ICAO; no truth enters."""
    started = time.perf_counter()
    report = {}
    detections = detect_flyovers(spec, FlyoverOptions(), top=15, report=report)
    detector_seconds = time.perf_counter() - started
    third_tracks, _ = load_flights(positions_path, {THIRD})
    found = discover(spec, third_tracks, saved, Atmosphere(343.0), RELAXED, saved["flights"])
    rows = [p for p in found["summary"]["passes"] if p["icao"] == THIRD]
    discovered = [e for e, t in zip(found["events"], found["tracks"], strict=True) if icao(t.id) == THIRD]
    if len(discovered) != 1:
        raise SystemExit(f"relaxed discovery did not confirm exactly one {THIRD} pass: {rows}")
    published = {icao(f): e for f, e in zip(saved["flights"], published_events, strict=True)}
    published[THIRD] = discovered[0]
    # Detector assignment: CPAs heard at the saved solution (velocity-file tracks, same positions).
    flights, _ = load_flights(traces_path, set(SUBSETS["three"]))
    recording = (float(spec.times[0]), float(spec.times[-1]))
    predicted = {}
    for flight in SUBSETS["three"]:
        passes = heard_passes([t for t in flights if icao(t.id) == flight], saved, recording)
        predicted[flight] = min(passes, key=lambda p: p[1])  # the closest pass in the recording
    detector, matches = {}, {}
    for flight, (heard, *_rest) in predicted.items():
        nearby = [(i, e) for i, e in enumerate(detections) if abs(e.cpa - heard) <= MATCH_SECONDS]
        if not nearby:
            raise SystemExit(f"no detection within {MATCH_SECONDS} s of {flight} heard at {heard:.1f} s")
        rank, event = max(nearby, key=lambda item: item[1].z)
        detector[flight] = event
        matches[flight] = {
            "rank": rank + 1,
            "predicted_heard_s": heard,
            "cpa_minus_predicted_s": event.cpa - heard,
        }
    del flights, third_tracks
    gc.collect()
    record = {
        "detector_options": asdict(FlyoverOptions()),
        "detector_seconds": detector_seconds,
        "detections": [asdict(e) for e in detections],
        "detector_candidates": report["candidates"],
        "detector_matches": matches,
        "discovery_options": asdict(RELAXED),
        "discovery_candidate": {k: saved[k] for k in ("position", "clock", "f0", "flights")},
        "discovery_passes": rows,
        "predicted_at_saved_solution": {
            f: dict(zip(("heard_s", "slant_m", "speed_m_s", "height_m", "flight"), p, strict=True))
            for f, p in predicted.items()
        },
        "event_sets": {
            name: {f: asdict(e) for f, e in events.items()}
            for name, events in (("published", published), ("detector", detector))
        },
    }
    return {"published": published, "detector": detector}, record


def third_flyover_check(record, baseline, layered_fit, tracks, atmospheres):
    """Task 1.1, no truth: a63e21 heard at the two-flyover fit versus the detector, and ray fallbacks."""
    track = next(t for t in tracks if icao(t.id) == THIRD)
    detection = record["event_sets"]["detector"][THIRD]
    discovered = record["event_sets"]["published"][THIRD]
    out = {}
    for label, result in (("constant/spline/ridge", baseline), ("layered/spline/ridge", layered_fit)):
        if result is None:
            continue
        observer = np.asarray(result["position"], dtype=float)
        clock = float(result["clock"])
        atmosphere = atmospheres["layered"] if label.startswith("layered") else Atmosphere(343.0)
        smoothed = prepare_track(track, "spline")
        if label.startswith("layered"):
            atmosphere = atmosphere.for_flight(track.id) if hasattr(atmosphere, "for_flight") else atmosphere
        hits = [h for h in closest_approaches(smoothed, observer, atmosphere) if 0 <= h[0] - clock <= 3600]
        heard, slant, speed = min(hits, key=lambda h: h[1])
        emission = heard - slant / 343.0
        position, _ = smoothed.at(emission)
        height = float(position[2] - observer[2])
        horizontal = float(np.hypot(*(position[:2] - observer[:2])))
        times = np.arange(detection["window"][0], detection["window"][1] + 0.25, 0.5)
        factor, emitted = predict(smoothed, observer, clock + times, 1.0, atmosphere)
        curve = discovered["f0"] * factor
        ideal = ideal_curve(
            times, detection["f0"], detection["speed"], detection["distance"], detection["cpa"]
        )
        use = np.isfinite(curve)
        entry = {
            "heard_s": heard - clock,
            "heard_minus_detector_s": heard - clock - detection["cpa"],
            "heard_minus_discovery_s": heard - clock - discovered["cpa"],
            "slant_m": slant,
            "speed_m_s": speed,
            "height_above_observer_m": height,
            "horizontal_m": horizontal,
            "elevation_deg": float(np.degrees(np.arctan2(height, horizontal))),
            "detector": {k: detection[k] for k in ("cpa", "f0", "speed", "distance", "z")},
            "discovery": {k: discovered[k] for k in ("cpa", "f0", "speed", "distance", "z")},
            "curve_minus_detector_ideal_hz": {
                "rms": float(np.sqrt(np.mean((curve[use] - ideal[use]) ** 2))),
                "max_abs": float(np.max(np.abs(curve[use] - ideal[use]))),
                "frames": int(use.sum()),
            },
        }
        if label.startswith("layered"):
            # Rays to every emission point heard in the detector window, and to every track sample.
            source, _ = smoothed.at(emitted[use])
            _, _, solved = atmosphere.ray_travel_time(source - observer, observer[2])
            near = (track.times > clock + detection["window"][0] - 60) & (
                track.times < clock + detection["window"][1]
            )
            _, _, sample_solved = atmosphere.ray_travel_time(track.positions[near] - observer, observer[2])
            elevation = np.degrees(
                np.arctan2(source[:, 2] - observer[2], np.hypot(*(source[:, :2] - observer[:2]).T))
            )
            entry["rays"] = {
                "window_frames": int(use.sum()),
                "window_fallbacks": int((~solved).sum()),
                "window_elevation_deg_range": [float(elevation.min()), float(elevation.max())],
                "track_samples": int(near.sum()),
                "track_sample_fallbacks": int((~sample_solved).sum()),
            }
        out[label] = entry
    return out


def spread(rows, names):
    """RMS horizontal distance of the cells' estimates from their own centroid (no truth)."""
    points = np.array([r["result"]["position"][:2] for r in rows if r["cell"] in names and r["result"]])
    if len(points) < 2:
        return None
    centroid = points.mean(axis=0)
    return {
        "cells": len(points),
        "rms_from_centroid_m": float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1)))),
        "max_from_centroid_m": float(np.max(np.linalg.norm(points - centroid, axis=1))),
        "centroid": centroid.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="data/oyster-bay")
    parser.add_argument("--traces", default="data/oyster-bay/traces-velocity.npz")
    parser.add_argument("--positions", default="data/oyster-bay/traces.npz", help="Discovery's trace file")
    parser.add_argument("--refinement", default="docs/results/oyster-bay/refinement.json")
    parser.add_argument("--weather", default="data/oyster-bay/weather/open-meteo-2026-09-08.json")
    parser.add_argument("--out", default="outputs/oyster-bay/three")
    parser.add_argument("--hum", default="subtract", choices=("subtract", "notch"))
    parser.add_argument("--sets", nargs="+", default=["published", "detector"])
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    parser.add_argument("--cells", nargs="+", help="Subset of cell names (default: all)")
    args = parser.parse_args()
    directory, output = Path(args.data), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    plan_cells = [c for c in cells(args.hum) if not args.cells or c[0] in args.cells]
    spec = Spectrogram.load(directory / "spectrogram.npz").whiten()
    published_events = load_events([str(directory / "events.json")])
    published = json.loads((directory / "published/verified_published_baseline.json").read_text())
    coarse = published["oyster_bay_search_best_per_pair.csv"]["highest_scoring_row"]
    if (coarse["icao1"], coarse["icao2"]) != KNOWN:
        raise SystemExit("unexpected published flight pair")
    refinement = json.loads(Path(args.refinement).read_text())
    saved = next(
        t["result"]
        for t in refinement["trials"]
        if t["method"] == "spline-linear" and t["height_m"] == -30.0 and t["sound_speed_m_s"] == 343.0
    )
    event_sets, record = build_event_sets(spec, published_events, saved, args.positions, args.traces)
    (output / "events.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    for name, events in event_sets.items():
        print(name, {f: (round(e.cpa, 1), round(e.f0, 2), e.window) for f, e in events.items()}, flush=True)

    tracks, frame = load_flights(args.traces, set(SUBSETS["three"]))
    needs_audio = any(options.ridge == "demod" for _, _, options in plan_cells)
    audio = Audio(decode_samples(directory / "recording.webm"), 2048.0) if needs_audio else None
    weather = WeatherSeries.load(args.weather)
    initial_latlon = [float(coarse["lat"]), float(coarse["lon"])]
    heights = {"constant": -30.0, "layered": GEOID_M + CAMERA_ABOVE_MSL_M}
    plan = {
        "validation": "Real recording (development data), conditional refinement with known flight identities "
        "and the published coarse candidate; not a blind search. Truth enters only after every fit.",
        "protocol": "scripts/compare_oyster_bay_combined.py per event subset",
        "traces": args.traces,
        "weather_file": args.weather,
        "hum_removal": args.hum,
        "cells": {name: repr(options) for name, _, options in plan_cells},
        "subsets": {k: SUBSETS[k] for k in args.subsets},
        "initial_latlon": initial_latlon,
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/dopplertracker").glob("*.py")
        },
    }
    rows, contexts = [], {}
    for set_name in args.sets:
        events_by_flight = event_sets[set_name]
        # As in compare_oyster_bay.py: the coarse heard time of flight 1 against this set's event for it.
        initial_clock = float(coarse["t_heard1"]) - events_by_flight[KNOWN[0]].cpa + float(coarse["shift"])
        for subset in args.subsets:
            flights = SUBSETS[subset]
            events = [events_by_flight[f] for f in flights]
            chosen = select_tracks(
                tracks, {f"icao{i}": f for i, f in enumerate(flights, 1)}, events, initial_clock
            )
            layered = weather.for_events(events, chosen, initial_clock)
            atmospheres = {"constant": Atmosphere(343.0), "layered": layered}
            contexts[(set_name, subset)] = (events, chosen, atmospheres)
            candidates = {}
            for key in sorted({a for _, a, _ in plan_cells}):
                observer = frame.to_local([*initial_latlon, heights[key]])
                score, clock, f0, scores = joint_score(
                    spec,
                    events,
                    chosen,
                    observer,
                    initial_clock,
                    atmosphere=atmospheres[key],
                    shift_radius=4,
                    shift_steps=17,
                    frequency_radius=12,
                    frequency_steps=121,
                )
                candidates[key] = {
                    "position": observer.tolist(),
                    "clock": clock,
                    "f0": f0,
                    "flights": [t.id for t in chosen],
                    "score": score,
                }
                print(set_name, subset, "calibrated", key, "clock", clock, "f0", f0, flush=True)
            for name, key, options in plan_cells:
                atmosphere = atmospheres[key]
                start = time.perf_counter()
                row = {
                    "set": set_name,
                    "subset": subset,
                    "flights": list(flights),
                    "cell": name,
                    "atmosphere": key,
                    "height_m": heights[key],
                    "initial_clock": initial_clock,
                    "initial_candidate": candidates[key],
                    "result": None,
                    "error": None,
                }
                try:
                    result = fit_projected(
                        spec,
                        events,
                        chosen,
                        candidates[key],
                        options,
                        atmosphere,
                        frame=frame,
                        elevation=ConstantElevation(heights[key]),
                        audio=audio,
                        keep_observations=True,
                    )
                    result["evidence"] = assess_candidate(spec, events, chosen, result, atmosphere)
                    result["lat_lon_height"] = frame.to_geodetic(result["position"]).tolist()
                    row["valid_frames"] = [int(np.sum(o["valid"])) for o in result["observations"]]
                    row["frames"] = [len(o["valid"]) for o in result["observations"]]
                    del result["observations"]
                    row["result"] = result
                except Exception as exc:  # failures are outcomes
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["seconds"] = time.perf_counter() - start
                rows.append(row)
                print(
                    set_name,
                    subset,
                    name,
                    "done",
                    round(row["seconds"], 1),
                    "s",
                    row["error"] or "",
                    flush=True,
                )
            (output / "fits.json").write_text(
                json.dumps({"plan": plan, "trials": rows}, indent=2, allow_nan=False)
            )

    def find(set_name, subset, cell):
        return next(
            (r["result"] for r in rows if (r["set"], r["subset"], r["cell"]) == (set_name, subset, cell)),
            None,
        )

    check = {}
    if ("published", "three") in contexts:
        _, three_tracks, atmospheres = contexts[("published", "three")]
        check = third_flyover_check(
            record,
            find("published", "known", "constant/spline/ridge"),
            find("published", "known", "layered/spline/ridge"),
            three_tracks,
            atmospheres,
        )
    factorial = [name for name, _, _ in combined_cells()]
    removal = [name for name, _, options in plan_cells if options.hum != "mask"]
    masked_demod = [name.replace(f"demod-{args.hum}", "demod", 1) for name in removal]
    swapped = [n for n in factorial if n not in masked_demod] + removal
    spreads = {
        f"{s}/{subset}": {
            "combined_cells": spread([r for r in rows if (r["set"], r["subset"]) == (s, subset)], factorial),
            "with_removal_cells": spread(
                [r for r in rows if (r["set"], r["subset"]) == (s, subset)], swapped
            ),
        }
        for s, subset in contexts
    }
    (output / "fits.json").write_text(
        json.dumps(
            {"plan": plan, "third_flyover_check": check, "spread": spreads, "trials": rows},
            indent=2,
            allow_nan=False,
        )
    )

    # Evaluation only: fitting above never receives the camera coordinates.
    truth = json.loads((directory / "published/provenance.json").read_text())["evaluation_truth"]
    truth_position = frame.to_local([*truth, 0.0])
    for row in rows:
        result = row["result"]
        row["error_m"] = (
            float(np.linalg.norm(np.asarray(result["position"])[:2] - truth_position[:2])) if result else None
        )
    stability = {}
    for (s, subset), _ in contexts.items():
        for label, names in (("combined_cells", factorial), ("with_removal_cells", swapped)):
            errors = [
                r["error_m"]
                for r in rows
                if (r["set"], r["subset"]) == (s, subset) and r["cell"] in names and r["error_m"] is not None
            ]
            stability[f"{s}/{subset}/{label}"] = (
                {
                    "cells": len(errors),
                    "min": float(np.min(errors)),
                    "median": float(np.median(errors)),
                    "max": float(np.max(errors)),
                    "std": float(np.std(errors)),
                    **(spreads[f"{s}/{subset}"][label] or {}),
                }
                if errors
                else None
            )
    (output / "results.json").write_text(
        json.dumps(
            {
                "plan": plan,
                "evaluation_truth": truth,
                "third_flyover_check": check,
                "stability": stability,
                "trials": rows,
            },
            indent=2,
            allow_nan=False,
        )
    )
    print(json.dumps(check, indent=1))
    print(
        f"{'set/subset/cell':58s} {'error_m':>8s} {'median|r| Hz':>22s} {'rms Hz':>22s} "
        f"{'valid frames':>22s} {'evidence':>8s}"
    )
    for row in rows:
        label = f"{row['set']}/{row['subset']}/{row['cell']}"
        result = row["result"]
        if not result:
            print(f"{label:58s} FAILED {row['error']}")
            continue
        frames = "/".join(map(str, row["valid_frames"])) + " of " + "/".join(map(str, row["frames"]))
        print(
            f"{label:58s} {row['error_m']:8.1f} "
            f"{' '.join(f'{x:6.3f}' for x in result['event_residual_hz']):>22s} "
            f"{' '.join(f'{x:6.3f}' for x in result['event_residual_rms_hz']):>22s} "
            f"{frames:>22s} {'accept' if result['evidence']['accepted'] else 'reject':>8s}"
        )
    print(f"{'stability':44s} {'n':>3s} {'min':>7s} {'median':>7s} {'max':>7s} {'std':>7s} {'spread':>7s}")
    for key, value in stability.items():
        if value:
            print(
                f"{key:44s} {value['cells']:3d} {value['min']:7.1f} {value['median']:7.1f} {value['max']:7.1f} "
                f"{value['std']:7.1f} {value.get('rms_from_centroid_m', float('nan')):7.1f}"
            )


if __name__ == "__main__":
    main()
