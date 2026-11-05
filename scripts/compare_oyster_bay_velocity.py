"""Oyster Bay comparison with aircraft-reported ADS-B velocity.

Same protocol as scripts/compare_oyster_bay.py (published flight pair, common
coarse start, shared clock/frequency calibration), but tracks come from
traces-velocity.npz, which carries per-point reported velocity.  Positions in
that file are parsed exactly as in traces.npz, so the nine original methods are
re-run as controls.  Truth is read only after every fit has finished; no fit
or parameter is selected by error.
"""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from dopplertracker.advanced import ModelOptions, fit_projected, select_model
from dopplertracker.data import save_tracks
from dopplertracker.detection import load_events
from dopplertracker.evidence import assess_candidate
from dopplertracker.geo import LocalFrame
from dopplertracker.physics import Atmosphere, Trajectory
from dopplertracker.refine import refine_candidate, refine_spectral
from dopplertracker.scoring import joint_score
from dopplertracker.signal import Spectrogram
from dopplertracker.stress_benchmark import VELOCITY_PROJECTED
from dopplertracker.terrain import ConstantElevation

ORIGINAL_METHODS = (
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
METHODS = ORIGINAL_METHODS + tuple(VELOCITY_PROJECTED)


def load_flights(path, icaos):
    """``load_tracks`` restricted to segments of the given ICAO codes (bounded memory)."""
    with np.load(path, allow_pickle=False) as z:
        frame = LocalFrame(*z["frame"])
        names = set(z.files)
        tracks = [
            Trajectory(
                item["id"],
                z[f"t{i}"],
                z[f"p{i}"],
                item.get("max_gap", 30.0),
                item,
                velocities=z[f"v{i}"] if f"v{i}" in names else None,
            )
            for i, item in enumerate(json.loads(str(z["metadata"])))
            if item["id"].split(":")[0] in icaos
        ]
    return tracks, frame


def select_tracks(tracks, coarse, events, initial_clock):
    """The covering segment per published ICAO, cut to the event window +/-120 s (velocities kept)."""
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
            Trajectory(
                track.id,
                track.times[mask],
                track.positions[mask],
                track.max_gap,
                track.metadata,
                velocities=None if track.velocities is None else track.velocities[mask],
            )
        )
    return chosen


def velocity_diagnostics(track, window):
    """Report cadence and reported-minus-differenced velocity at report points (no truth)."""
    differenced = np.concatenate(
        [np.gradient(track.positions[idx], track.times[idx], axis=0) for idx in track.segments]
    )
    times = np.concatenate([track.times[idx] for idx in track.segments])
    reported = np.concatenate([track.velocities[idx] for idx in track.segments])
    out = {"id": track.id}
    for label, mask in (
        ("track_slice", np.ones(len(times), bool)),
        ("event_window", (times >= window[0]) & (times <= window[1])),
    ):
        diff = (reported - differenced)[mask]
        horizontal = np.linalg.norm(diff[:, :2], axis=1)
        vertical = np.abs(diff[:, 2])
        intervals = np.diff(times[mask])
        out[label] = {
            "reports": int(mask.sum()),
            "interval_s": {
                "min": float(intervals.min()),
                "median": float(np.median(intervals)),
                "max": float(intervals.max()),
            },
            "horizontal_reported_minus_differenced_m_s": {
                "rms": float(np.sqrt(np.nanmean(horizontal**2))),
                "max": float(np.nanmax(horizontal)),
            },
            "vertical_reported_minus_differenced_m_s": {
                "rms": float(np.sqrt(np.nanmean(vertical**2))),
                "max": float(np.nanmax(vertical)),
            },
            "reported_ground_speed_m_s_median": float(np.median(np.linalg.norm(reported[mask, :2], axis=1))),
        }
    out["vertical_rate_source_runs"] = track.metadata.get("vertical_rate_source_runs")
    return out


def run_method(method, spec, events, chosen, candidate, atmosphere, frame, elevation):
    if method in ("brightness", "contrast"):
        return refine_spectral(
            spec, events, chosen, candidate, method, atmosphere, frame=frame, elevation=elevation
        )
    if method in ("legacy-ridge", "legacy-drift"):
        return refine_candidate(
            spec,
            events,
            chosen,
            candidate,
            atmosphere,
            fit_drift=method == "legacy-drift",
            frame=frame,
            elevation=elevation,
        )
    if method in VELOCITY_PROJECTED:
        options = VELOCITY_PROJECTED[method]
    elif method in ("projected-constant", "projected-linear"):
        options = ModelOptions(degree=0 if method == "projected-constant" else 1)
    elif method in ("spline-linear", "spline-cv"):
        options = ModelOptions(degree=1, smoothing="spline")
    elif method == "kalman-linear":
        options = ModelOptions(degree=1, smoothing="kalman")
    else:
        raise ValueError(f"unknown method {method}")
    if method == "spline-cv":
        return select_model(
            spec,
            events,
            chosen,
            candidate,
            [replace(options, degree=d) for d in (0, 1, 2)],
            atmosphere,
            frame=frame,
            elevation=elevation,
        )
    return fit_projected(
        spec, events, chosen, candidate, options, atmosphere, frame=frame, elevation=elevation
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/oyster-bay")
    parser.add_argument("--traces", default="data/oyster-bay/traces-velocity.npz")
    parser.add_argument("--out", default="outputs/oyster-bay/velocity-comparison")
    parser.add_argument("--heights", nargs="+", type=float, default=[-30.0])
    parser.add_argument("--sound-speeds", nargs="+", type=float, default=[343.0])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    directory, output = Path(args.data), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    spec = Spectrogram.load(directory / "spectrogram.npz").whiten()
    events = load_events([str(directory / "events.json")])
    published = json.loads((directory / "published/verified_published_baseline.json").read_text())
    coarse = published["oyster_bay_search_best_per_pair.csv"]["highest_scoring_row"]
    tracks, frame = load_flights(args.traces, {coarse["icao1"], coarse["icao2"]})
    initial_clock = float(coarse["t_heard1"]) - events[0].cpa + float(coarse["shift"])
    chosen = select_tracks(tracks, coarse, events, initial_clock)
    del tracks
    save_tracks(output / "matched-tracks.npz", chosen, frame)
    diagnostics = [
        velocity_diagnostics(t, (initial_clock + e.window[0], initial_clock + e.window[1]))
        for t, e in zip(chosen, events, strict=True)
    ]
    print(json.dumps(diagnostics, indent=2), flush=True)
    plan = {
        "validation": "Real recording, conditional refinement with published flight identities and coarse candidate; not a blind search.",
        "protocol": "scripts/compare_oyster_bay.py, with traces-velocity.npz and reported-velocity methods",
        "traces": args.traces,
        "methods": list(args.methods),
        "heights_ellipsoid_m": args.heights,
        "sound_speeds_m_s": args.sound_speeds,
        "initial_latlon": [float(coarse["lat"]), float(coarse["lon"])],
        "initial_clock": initial_clock,
        "flights": [t.id for t in chosen],
        "velocity_diagnostics": diagnostics,
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
            for method in args.methods:
                start = time.perf_counter()
                row = {
                    "method": method,
                    "height_m": height,
                    "sound_speed_m_s": speed,
                    "result": None,
                    "error": None,
                }
                try:
                    result = run_method(method, spec, events, chosen, candidate, atmosphere, frame, elevation)
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
    report = {"plan": plan, "evaluation_truth": truth, "trials": rows}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for row in rows:
        accepted = row["result"]["evidence"]["accepted"] if row["result"] else None
        print(
            row["height_m"],
            row["sound_speed_m_s"],
            row["method"],
            "error_m",
            row["error_m"],
            "accepted",
            accepted,
        )


if __name__ == "__main__":
    main()
