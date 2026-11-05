"""Compare frozen refiners on real audio, initialized from the published coarse winner.

This is explicitly conditional on the published flight pair. Truth is read only
by the evaluation stage after every fit has finished. No fit is selected by error.
"""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from dopplertracker.advanced import ModelOptions, fit_projected, select_model
from dopplertracker.data import load_tracks, save_tracks
from dopplertracker.detection import load_events
from dopplertracker.evidence import assess_candidate
from dopplertracker.physics import Atmosphere, Trajectory
from dopplertracker.refine import refine_candidate, refine_spectral
from dopplertracker.scoring import joint_score
from dopplertracker.signal import Spectrogram
from dopplertracker.terrain import ConstantElevation

METHODS = (
    "brightness",
    "contrast",
    "legacy-ridge",
    "legacy-drift",
    "projected-constant",
    "projected-linear",
    "spline-linear",
    "kalman-linear",
    "spline-cv",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/oyster-bay")
    parser.add_argument("--out", default="outputs/oyster-bay/comparison")
    parser.add_argument("--heights", nargs="+", type=float, default=[-30.0, 0.0])
    parser.add_argument("--sound-speeds", nargs="+", type=float, default=[343.0, 340.0])
    args = parser.parse_args()
    directory, output = Path(args.data), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    raw = Spectrogram.load(directory / "spectrogram.npz")
    spec = raw.whiten()
    events = load_events([str(directory / "events.json")])
    tracks, frame = load_tracks(directory / "traces.npz")
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
    save_tracks(output / "matched-tracks.npz", chosen, frame)
    plan = {
        "validation": "Real recording, conditional refinement with published flight identities and coarse candidate; not a blind search.",
        "recording_id": "4CxVxuMVkMc",
        "methods": list(METHODS),
        "heights_ellipsoid_m": args.heights,
        "sound_speeds_m_s": args.sound_speeds,
        "height_note": "-30 m approximates a sea-level shoreline with geoid correction; 0 m is a datum sensitivity case. Camera height and weather are not measured.",
        "primary_condition": {"height_m": args.heights[0], "sound_speed_m_s": args.sound_speeds[0]},
        "initial_latlon": [float(coarse["lat"]), float(coarse["lon"])],
        "initial_clock": initial_clock,
        "flights": [t.id for t in chosen],
        "initial_calibration": "At the common fixed coarse position, jointly sweep a shared +/-4 s clock shift and per-event +/-12 Hz source frequencies; no true location input.",
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/dopplertracker").glob("*.py")
        },
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    rows = []
    for height in args.heights:
        for speed in args.sound_speeds:
            atmosphere = Atmosphere(speed)
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
            print("Calibrated", height, speed, "clock", clock, "f0", f0, "scores", scores, flush=True)
            for method in METHODS:
                start = time.perf_counter()
                row = {
                    "method": method,
                    "height_m": height,
                    "sound_speed_m_s": speed,
                    "initial_candidate": candidate,
                    "result": None,
                    "error": None,
                }
                try:
                    if method in ("brightness", "contrast"):
                        result = refine_spectral(
                            spec,
                            events,
                            chosen,
                            candidate,
                            method,
                            atmosphere,
                            frame=frame,
                            elevation=elevation,
                        )
                    elif method in ("legacy-ridge", "legacy-drift"):
                        result = refine_candidate(
                            spec,
                            events,
                            chosen,
                            candidate,
                            atmosphere,
                            fit_drift=method == "legacy-drift",
                            frame=frame,
                            elevation=elevation,
                        )
                    else:
                        options = ModelOptions(
                            degree=0 if method == "projected-constant" else 1,
                            smoothing="spline"
                            if method.startswith("spline")
                            else "kalman"
                            if method.startswith("kalman")
                            else "none",
                        )
                        if method == "spline-cv":
                            result = select_model(
                                spec,
                                events,
                                chosen,
                                candidate,
                                [replace(options, degree=d) for d in (0, 1, 2)],
                                atmosphere,
                                frame=frame,
                                elevation=elevation,
                            )
                        else:
                            result = fit_projected(
                                spec,
                                events,
                                chosen,
                                candidate,
                                options,
                                atmosphere,
                                frame=frame,
                                elevation=elevation,
                            )
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
                print(method, "done", round(row["seconds"], 2), "s", row["error"] or "", flush=True)
    # Evaluation only: fitting above never receives the camera coordinates.
    truth = json.loads((directory / "published/provenance.json").read_text())["evaluation_truth"]
    truth_position = frame.to_local([*truth, 0.0])
    for row in rows:
        result = row["result"]
        row["error_m"] = (
            float(np.linalg.norm(np.asarray(result["position"])[:2] - truth_position[:2])) if result else None
        )
    report = {
        "plan": plan,
        "evaluation_truth": truth,
        "trials": rows,
        "published_refined_error_m": published["oyster_bay_refined_50m.csv"]["horizontal_error_m"],
        "published_coarse_error_m": published["oyster_bay_search_best_per_pair.csv"]["horizontal_error_m"],
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for row in rows:
        print(row["height_m"], row["sound_speed_m_s"], row["method"], "error_m", row["error_m"], flush=True)


if __name__ == "__main__":
    main()
