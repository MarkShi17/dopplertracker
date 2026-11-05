"""Oyster Bay refiners under measured layered and diagnostic homogeneous atmospheres.

Same protocol as compare_oyster_bay.py (published flight pair and coarse winner,
shared clock / source-frequency calibration, the same nine methods), repeated for:

  original      ellipsoid height -30 m, homogeneous 343 m/s (compare_oyster_bay.py primary)
  layered       height -30.854 + 2 m, Open-Meteo profile interpolated to each event
  homog-336.3   height -28.854 m, homogeneous 336.3 m/s (flight-1 path harmonic mean)
  homog-344.3   height -28.854 m, homogeneous 344.3 m/s (flight-2 path harmonic mean)

The weather is queried at the region centre, never the truth. Truth is read only
after every fit has finished; nothing is selected by error.
"""

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_oyster_bay import METHODS  # noqa: E402

from dopplertracker.advanced import ModelOptions, fit_projected, select_model  # noqa: E402
from dopplertracker.atmosphere import WeatherSeries  # noqa: E402
from dopplertracker.data import load_tracks  # noqa: E402
from dopplertracker.detection import load_events  # noqa: E402
from dopplertracker.evidence import assess_candidate  # noqa: E402
from dopplertracker.physics import Atmosphere, Trajectory  # noqa: E402
from dopplertracker.refine import refine_candidate, refine_spectral  # noqa: E402
from dopplertracker.scoring import joint_score  # noqa: E402
from dopplertracker.signal import Spectrogram  # noqa: E402
from dopplertracker.terrain import ConstantElevation  # noqa: E402

GEOID_M = -30.854  # NOAA GEOID18 at 40.85, -73.5
CAMERA_ABOVE_MSL_M = 2.0


def select_tracks(directory, events, tracks):
    """compare_oyster_bay.py's flight-segment choice for the published pair."""
    published = json.loads((directory / "published/verified_published_baseline.json").read_text())
    coarse = published["oyster_bay_search_best_per_pair.csv"]["highest_scoring_row"]
    initial_clock = float(coarse["t_heard1"]) - events[0].cpa + float(coarse["shift"])
    chosen = []
    for index, event in enumerate(events, 1):
        matching = [
            t
            for t in tracks
            if t.id.startswith(coarse[f"icao{index}"] + ":")
            and t.times[0] < initial_clock + event.window[0] - 30
            and t.times[-1] > initial_clock + event.window[1]
        ]
        if len(matching) != 1:
            raise ValueError(f"Expected one covering flight segment for event {index}, got {len(matching)}")
        track = matching[0]
        mask = (track.times >= initial_clock + event.window[0] - 120) & (
            track.times <= initial_clock + event.window[1] + 120
        )
        chosen.append(
            Trajectory(track.id, track.times[mask], track.positions[mask], track.max_gap, track.metadata)
        )
    return published, coarse, initial_clock, chosen


def run_method(method, spec, events, chosen, candidate, atmosphere, frame, elevation):
    """Explicit branch per compare_oyster_bay.py method; unknown names are errors."""
    common = dict(frame=frame, elevation=elevation)
    if method in ("brightness", "contrast"):
        return refine_spectral(spec, events, chosen, candidate, method, atmosphere, **common)
    if method in ("legacy-ridge", "legacy-drift"):
        return refine_candidate(
            spec, events, chosen, candidate, atmosphere, fit_drift=method == "legacy-drift", **common
        )
    options = {
        "projected-constant": ModelOptions(degree=0),
        "projected-linear": ModelOptions(degree=1),
        "spline-linear": ModelOptions(degree=1, smoothing="spline"),
        "kalman-linear": ModelOptions(degree=1, smoothing="kalman"),
        "spline-cv": ModelOptions(degree=1, smoothing="spline"),
    }
    if method not in options:
        raise ValueError(f"unhandled method {method}")
    if method == "spline-cv":
        choices = [replace(options[method], degree=d) for d in (0, 1, 2)]
        return select_model(spec, events, chosen, candidate, choices, atmosphere, **common)
    return fit_projected(spec, events, chosen, candidate, options[method], atmosphere, **common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/oyster-bay")
    parser.add_argument("--weather", default="data/oyster-bay/weather/open-meteo-2026-09-08.json")
    parser.add_argument("--out", default="outputs/oyster-bay/atmosphere-comparison")
    parser.add_argument(
        "--conditions", nargs="+", default=["original", "layered", "homog-336.3", "homog-344.3"]
    )
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    args = parser.parse_args()
    directory, output = Path(args.data), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    spec = Spectrogram.load(directory / "spectrogram.npz").whiten()
    events = load_events([str(directory / "events.json")])
    tracks, frame = load_tracks(directory / "traces.npz")
    published, coarse, initial_clock, chosen = select_tracks(directory, events, tracks)
    del tracks  # only the two chosen segments are used; the full set is ~2 GB in memory
    gc.collect()
    weather = WeatherSeries.load(args.weather)
    # Per-event profiles at the initial (pre-calibration) heard CPAs; the +/-4 s clock
    # calibration moves them by a negligible fraction of the hourly interpolation.
    layered = weather.for_events(events, chosen, initial_clock)
    camera = GEOID_M + CAMERA_ABOVE_MSL_M
    conditions = {
        "original": (-30.0, Atmosphere(343.0)),
        "layered": (camera, layered),
        "homog-336.3": (camera, Atmosphere(336.3)),
        "homog-344.3": (camera, Atmosphere(344.3)),
    }
    plan = {
        "validation": "Real recording, conditional refinement with published flight identities and coarse candidate; not a blind search.",
        "conditions": {name: {"height_m": conditions[name][0]} for name in args.conditions},
        "methods": args.methods,
        "weather_file": args.weather,
        "weather_metadata": weather.metadata,
        "layered_profiles": layered.to_record(),
        "initial_latlon": [float(coarse["lat"]), float(coarse["lon"])],
        "initial_clock": initial_clock,
        "flights": [t.id for t in chosen],
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/dopplertracker").glob("*.py")
        },
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    rows = []
    for name in args.conditions:
        height, atmosphere = conditions[name]
        elevation = ConstantElevation(height)
        observer = frame.to_local([*plan["initial_latlon"], height])
        score, clock, f0, scores = joint_score(
            spec,
            events,
            chosen,
            observer,
            initial_clock,
            atmosphere=atmosphere,
            shift_radius=4,
            shift_steps=17,
            frequency_radius=12,
            frequency_steps=121,
        )
        candidate = {
            "position": observer.tolist(),
            "clock": clock,
            "f0": f0,
            "flights": [t.id for t in chosen],
            "score": score,
        }
        print("Calibrated", name, "clock", clock, "f0", f0, "scores", scores, flush=True)
        for method in args.methods:
            start = time.perf_counter()
            row = {"condition": name, "method": method, "height_m": height, "result": None, "error": None}
            row["initial_candidate"] = candidate
            try:
                result = run_method(method, spec, events, chosen, candidate, atmosphere, frame, elevation)
                if not isinstance(atmosphere, Atmosphere):
                    result.setdefault("atmosphere", atmosphere.to_record())
                result["evidence"] = assess_candidate(spec, events, chosen, result, atmosphere)
                result["lat_lon_height"] = frame.to_geodetic(result["position"]).tolist()
                row["result"] = result
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = time.perf_counter() - start
            rows.append(row)
            (output / "fits.json").write_text(
                json.dumps({"plan": plan, "trials": rows}, indent=2, allow_nan=False)
            )
            print(name, method, "done", round(row["seconds"], 1), "s", row["error"] or "", flush=True)
    # Evaluation only: fitting above never receives the camera coordinates.
    truth = json.loads((directory / "published/provenance.json").read_text())["evaluation_truth"]
    truth_position = frame.to_local([*truth, 0.0])
    for row in rows:
        result = row["result"]
        row["error_m"] = (
            float(np.linalg.norm(np.asarray(result["position"])[:2] - truth_position[:2])) if result else None
        )
    report = {"plan": plan, "evaluation_truth": truth, "trials": rows}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"{'method':20s}" + "".join(f"{name:>14s}" for name in args.conditions))
    for method in args.methods:
        cells = []
        for name in args.conditions:
            row = next(r for r in rows if r["condition"] == name and r["method"] == method)
            accepted = row["result"]["evidence"]["accepted"] if row["result"] else False
            cells.append(
                "failed" if row["error_m"] is None else f"{row['error_m']:.1f}{'' if accepted else '*'}"
            )
        print(f"{method:20s}" + "".join(f"{c:>14s}" for c in cells))
    print("* = evidence check not accepted")


if __name__ == "__main__":
    main()
