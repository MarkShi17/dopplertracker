"""Compare Doppler-measurement options on the Oyster Bay recording.

Same protocol as scripts/compare_oyster_bay.py (published flight pair, shared
coarse start, joint clock/frequency calibration at the fixed start), which is
copied here so that script's numbers are untouched.  Every method is the
``spline-linear`` baseline plus one or more measurement options.  Truth is read
only after every fit has finished; no fit is selected by error.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from dopplertracker.advanced import ModelOptions, fit_projected
from dopplertracker.data import load_tracks
from dopplertracker.demod import Audio
from dopplertracker.detection import load_events
from dopplertracker.physics import Atmosphere, Trajectory
from dopplertracker.scoring import joint_score
from dopplertracker.signal import Spectrogram, decode_samples
from dopplertracker.terrain import ConstantElevation

BASELINE = ModelOptions(degree=1, smoothing="spline")
METHODS = {
    "spline-linear": BASELINE,
    "all-frames": replace(BASELINE, max_frames=None),
    "reextract": replace(BASELINE, reextract=3),
    "robust-noise": replace(BASELINE, noise="robust"),
    "snr-noise": replace(BASELINE, noise="snr"),
    "ratio": replace(BASELINE, harmonic_mode="ratio"),
    "ratio-snr": replace(BASELINE, harmonic_mode="ratio", noise="snr"),
    "ratio-snr-h4": replace(BASELINE, harmonic_mode="ratio", noise="snr", max_harmonic=4),
    "stft-combined": replace(
        BASELINE, reextract=3, noise="snr", harmonic_mode="ratio", max_harmonic=4, max_frames=None
    ),
    "demod-single-pass": replace(BASELINE, ridge="demod"),
    "demod": replace(BASELINE, ridge="demod", reextract=2),
    "demod-snr": replace(BASELINE, ridge="demod", reextract=2, noise="snr"),
    "demod-snr-h4": replace(BASELINE, ridge="demod", reextract=2, noise="snr", max_harmonic=4),
    "demod-combined": replace(
        BASELINE, ridge="demod", reextract=2, noise="snr", max_harmonic=4, max_frames=None
    ),
}


def matched_tracks(directory, events, tracks, coarse):
    """Published flight segments around each event, exactly as compare_oyster_bay.py selects them."""
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
    return chosen, initial_clock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/oyster-bay")
    parser.add_argument("--out", default="outputs/oyster-bay/measurement")
    parser.add_argument("--height", type=float, default=-30.0)
    parser.add_argument("--sound-speed", type=float, default=343.0)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    args = parser.parse_args()
    directory, output = Path(args.data), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    spec = Spectrogram.load(directory / "spectrogram.npz").whiten()
    events = load_events([str(directory / "events.json")])
    tracks, frame = load_tracks(directory / "traces.npz")
    published = json.loads((directory / "published/verified_published_baseline.json").read_text())
    coarse = published["oyster_bay_search_best_per_pair.csv"]["highest_scoring_row"]
    chosen, initial_clock = matched_tracks(directory, events, tracks, coarse)
    needs_audio = any(METHODS[m].ridge == "demod" for m in args.methods)
    # About 30 MB for the hour at 2048 Hz; sample k is spectrogram time k / 2048 s.
    audio = Audio(decode_samples(directory / "recording.webm"), 2048.0) if needs_audio else None

    atmosphere = Atmosphere(args.sound_speed)
    elevation = ConstantElevation(args.height)
    initial_latlon = [float(coarse["lat"]), float(coarse["lon"])]
    observer = frame.to_local([*initial_latlon, args.height])
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
    print("Calibrated clock", clock, "f0", f0, "scores", scores, flush=True)
    plan = {
        "validation": "Real recording, conditional refinement with published flight identities and coarse "
        "candidate; not a blind search. Truth enters only after every fit.",
        "condition": {"height_m": args.height, "sound_speed_m_s": args.sound_speed},
        "initial_latlon": initial_latlon,
        "initial_clock": initial_clock,
        "flights": [t.id for t in chosen],
        "methods": {name: repr(METHODS[name]) for name in args.methods},
    }
    rows = []
    for method in args.methods:
        start = time.perf_counter()
        row = {"method": method, "result": None, "error": None}
        try:
            row["result"] = fit_projected(
                spec,
                events,
                chosen,
                candidate,
                METHODS[method],
                atmosphere,
                frame=frame,
                elevation=elevation,
                audio=audio,
                keep_observations=True,
            )
            row["result"]["lat_lon_height"] = frame.to_geodetic(row["result"]["position"]).tolist()
        except Exception as exc:  # failures are outcomes
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
        print(method, "done", round(row["seconds"], 1), "s", row["error"] or "", flush=True)
        (output / "fits.json").write_text(
            json.dumps({"plan": plan, "trials": rows}, indent=2, allow_nan=False)
        )
    # Evaluation only: fitting above never receives the camera coordinates.
    truth = json.loads((directory / "published/provenance.json").read_text())["evaluation_truth"]
    truth_position = frame.to_local([*truth, 0.0])
    for row in rows:
        result = row["result"]
        row["error_m"] = (
            float(np.linalg.norm(np.asarray(result["position"])[:2] - truth_position[:2])) if result else None
        )
    (output / "results.json").write_text(
        json.dumps({"plan": plan, "evaluation_truth": truth, "trials": rows}, indent=2, allow_nan=False)
    )
    print(
        f"{'method':20s} {'error_m':>8s} {'median|r| Hz':>16s} {'rms Hz':>16s} {'valid/frames':>14s} passes"
    )
    for row in rows:
        result = row["result"]
        if not result:
            print(f"{row['method']:20s} FAILED {row['error']}")
            continue
        measurement = result.get("measurement", {})
        valid = measurement.get("valid_frames")
        frames = measurement.get("frames")
        counts = "/".join(map(str, valid)) + " of " + "/".join(map(str, frames)) if valid else ""
        print(
            f"{row['method']:20s} {row['error_m']:8.1f} "
            f"{' '.join(f'{x:7.3f}' for x in result['event_residual_hz']):>16s} "
            f"{' '.join(f'{x:7.3f}' for x in result['event_residual_rms_hz']):>16s} "
            f"{counts:>14s} {measurement.get('passes', 1)}"
        )


if __name__ == "__main__":
    main()
