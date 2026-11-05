"""Candidate-conditioned flyover discovery on the Oyster Bay recording (development data).

1. At the saved two-flyover ``spline-linear`` solution (height -30 m, 343 m/s),
   enumerate every ADS-B segment's audible pass across the hour and scan it.
2. Jointly refit position and the shared clock with the two original plus the
   discovered events, then discover once more at the refined position.
3. Count independent confirmations at the ten best-scoring wrong-pair solutions
   of the refined shortlist and at the true pair.

Truth is read only after every fit and scan has finished, to report errors.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dopplertracker.advanced import ModelOptions
from dopplertracker.data import load_tracks
from dopplertracker.detection import load_events, save_events
from dopplertracker.discovery import (
    DiscoveryOptions,
    RidgeMap,
    discover,
    discover_and_refit,
    trim_track,
)
from dopplertracker.physics import Atmosphere
from dopplertracker.signal import Spectrogram
from dopplertracker.terrain import ConstantElevation

TRUE_PAIR = ["a1dfeb", "ace391"]


def listing(summary):
    keys = ("icao", "flight", "type", "registration", "heard_s", "distance_m", "altitude_m", "z", "f0")
    return [{k: p[k] for k in keys} for p in summary["passes"] if p["confirmed"] and not p["used_in_fit"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/oyster-bay")
    parser.add_argument("--results", default="docs/results/oyster-bay")
    parser.add_argument("--out", default="outputs/oyster-bay/discovery")
    parser.add_argument("--wrong", type=int, default=10)
    args = parser.parse_args()
    directory, output = Path(args.data), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    spec = Spectrogram.load(directory / "spectrogram.npz").whiten()
    events = load_events([str(directory / "events.json")])
    tracks, frame = load_tracks(directory / "traces.npz")
    lookup = {t.id: t for t in tracks}
    options = DiscoveryOptions()
    atmosphere = Atmosphere(343.0)
    model = ModelOptions(degree=1, smoothing="spline")
    ridges = RidgeMap(spec, options.background_hz)

    refinement = json.loads((Path(args.results) / "refinement.json").read_text())
    saved = next(
        t["result"]
        for t in refinement["trials"]
        if t["method"] == "spline-linear" and t["height_m"] == -30.0 and t["sound_speed_m_s"] == 343.0
    )
    candidate = {k: saved[k] for k in ("position", "clock", "f0", "flights")}
    # Same trimming as scripts/compare_oyster_bay.py: event window +/- 120 s.
    chosen = [
        trim_track(lookup[f], candidate["clock"] + e.window[0] - 120, candidate["clock"] + e.window[1] + 120)
        for f, e in zip(candidate["flights"], events, strict=True)
    ]
    print(f"Discovering at the saved two-flyover solution over {len(tracks)} segments", flush=True)
    pipeline = discover_and_refit(
        spec,
        events,
        chosen,
        tracks,
        candidate,
        model,
        atmosphere,
        options,
        frame=frame,
        elevation=ConstantElevation(-30.0),
        ridges=ridges,
    )
    save_events(pipeline["events"], output / "joint-events.json")
    for name in ("first", "second"):
        print(name, "confirmed", pipeline[name]["confirmed_flights"], flush=True)

    shortlist = json.loads((Path(args.results) / "refined-shortlist.json").read_text())
    fits = [t for t in shortlist["trials"] if t["method"] == "spline-linear" and t["result"] is not None]
    fits.sort(key=lambda t: t["result"]["score"], reverse=True)
    true_fit = next(t for t in fits if t["flights"] == TRUE_PAIR)
    wrong = [t for t in fits if t["flights"] != TRUE_PAIR][: args.wrong]
    controls = []
    for rank, trial in [(fits.index(true_fit) + 1, true_fit), *((fits.index(t) + 1, t) for t in wrong)]:
        result = trial["result"]
        report = discover(ridges, tracks, result, atmosphere, options, trial["flights"])["summary"]
        controls.append(
            {
                "flights": trial["flights"],
                "true_pair": trial["flights"] == TRUE_PAIR,
                "score_rank": rank,
                "score": result["score"],
                "evidence_accepted": bool(trial["evidence"] and trial["evidence"]["accepted"]),
                "passes_scanned": report["passes_scanned"],
                "confirmed": report["confirmed_independent"],
                "z_sum": report["z_sum"],
                "confirmed_list": listing(report),
                "used_flights_confirmed": report["used_flights_confirmed"],
            }
        )
        print(trial["flights"], "rank", rank, "confirmed", report["confirmed_flights"], flush=True)

    # Evaluation only: everything above ran without the camera coordinates.
    truth = json.loads((directory / "published/provenance.json").read_text())["evaluation_truth"]
    target = frame.to_local([*truth, 0.0])

    def error(position):
        return float(np.linalg.norm(np.asarray(position)[:2] - target[:2]))

    shortlist_errors = {tuple(t["flights"]): t.get("error_m") for t in fits}
    for row in controls:
        row["error_m"] = shortlist_errors.get(tuple(row["flights"]))
    joint = pipeline["result"]
    report = {
        "validation": "Real recording, development data; discovery conditioned on the saved two-flyover fit.",
        "options": {k: list(v) if isinstance(v, tuple) else v for k, v in vars(options).items()},
        "model": vars(model),
        "initial_candidate": candidate,
        "initial_error_m": error(candidate["position"]),
        "first_discovery": pipeline["first"],
        "first_confirmed": listing(pipeline["first"]),
        "joint_result": joint,
        "joint_flights": joint["flights"],
        "joint_error_m": error(joint["position"]),
        "joint_lat_lon_height": frame.to_geodetic(joint["position"]).tolist(),
        "second_discovery": pipeline["second"],
        "second_confirmed": listing(pipeline["second"]),
        "shortlist_controls": controls,
        "evaluation_truth": truth,
        "seconds": time.perf_counter() - began,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("initial error", report["initial_error_m"], "joint error", report["joint_error_m"])
    for row in report["first_confirmed"]:
        print(
            f"{row['icao']} {row['type']} slant {row['distance_m']:.0f} m alt {row['altitude_m']:.0f} m "
            f"z {row['z']:.1f} f0 {row['f0']:.1f}"
        )
    for row in controls:
        print(row["flights"], row["true_pair"], row["score_rank"], row["confirmed"], round(row["z_sum"], 1))


if __name__ == "__main__":
    main()
