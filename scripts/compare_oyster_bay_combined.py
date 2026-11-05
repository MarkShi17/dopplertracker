"""Oyster Bay with the atmosphere, velocity and measurement options combined.

Same protocol as scripts/compare_oyster_bay.py (published flight pair, common
coarse start, joint clock / source-frequency calibration at that start, run
once per atmosphere).  Every fit is a linear source model.  Full factorial:

  atmosphere   constant   ellipsoid height -30 m, homogeneous 343 m/s
               layered    height -28.854 m, Open-Meteo profile per event
  trajectory   spline     spline-smoothed differenced positions (spline-linear)
               fused      reported velocity fused with positions (velocity="fused")
  measurement  ridge      default spectrogram peak ridge
               demod      raw-audio demodulation, SNR noise model, 2 re-passes

plus two variants of the all-new cell: fitted sound-speed and wind offsets,
and harmonics up to 4 with every frame.  Tracks come from traces-velocity.npz,
whose positions are parsed exactly as in traces.npz.  Truth is read only after
every fit has finished and is used only to compute the location error; no
fit or option is selected by it.
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
from compare_oyster_bay_atmosphere import CAMERA_ABOVE_MSL_M, GEOID_M  # noqa: E402
from compare_oyster_bay_velocity import load_flights, select_tracks  # noqa: E402

from dopplertracker.advanced import ModelOptions, fit_projected  # noqa: E402
from dopplertracker.atmosphere import WeatherSeries  # noqa: E402
from dopplertracker.demod import Audio  # noqa: E402
from dopplertracker.detection import load_events  # noqa: E402
from dopplertracker.evidence import assess_candidate  # noqa: E402
from dopplertracker.physics import Atmosphere  # noqa: E402
from dopplertracker.scoring import joint_score  # noqa: E402
from dopplertracker.signal import Spectrogram, decode_samples  # noqa: E402
from dopplertracker.terrain import ConstantElevation  # noqa: E402

TRAJECTORIES = {
    "spline": ModelOptions(degree=1, smoothing="spline"),
    "fused": ModelOptions(degree=1, velocity="fused"),
}
MEASUREMENTS = {"ridge": {}, "demod": dict(ridge="demod", reextract=2, noise="snr")}
# Variants of the cell with every new option on (layered / fused / demod).
VARIANTS = {
    "fit-c-wind": dict(fit_sound_speed=True, fit_wind=True),
    "h4-all-frames": dict(max_harmonic=4, max_frames=None),
}


def cells():
    """(name, atmosphere key, options) for the factorial, then the variants."""
    out = []
    for atmosphere in ("constant", "layered"):
        for trajectory, base in TRAJECTORIES.items():
            for measurement, changes in MEASUREMENTS.items():
                name = f"{atmosphere}/{trajectory}/{measurement}"
                out.append((name, atmosphere, replace(base, **changes)))
    full = replace(TRAJECTORIES["fused"], **MEASUREMENTS["demod"])
    for variant, changes in VARIANTS.items():
        out.append((f"layered/fused/demod+{variant}", "layered", replace(full, **changes)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/oyster-bay")
    parser.add_argument("--traces", default="data/oyster-bay/traces-velocity.npz")
    parser.add_argument("--weather", default="data/oyster-bay/weather/open-meteo-2026-09-08.json")
    parser.add_argument("--out", default="outputs/oyster-bay/combined")
    parser.add_argument("--cells", nargs="+", help="Subset of cell names (default: all)")
    args = parser.parse_args()
    directory, output = Path(args.data), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    plan_cells = [c for c in cells() if not args.cells or c[0] in args.cells]
    spec = Spectrogram.load(directory / "spectrogram.npz").whiten()
    events = load_events([str(directory / "events.json")])
    published = json.loads((directory / "published/verified_published_baseline.json").read_text())
    coarse = published["oyster_bay_search_best_per_pair.csv"]["highest_scoring_row"]
    tracks, frame = load_flights(args.traces, {coarse["icao1"], coarse["icao2"]})
    initial_clock = float(coarse["t_heard1"]) - events[0].cpa + float(coarse["shift"])
    chosen = select_tracks(tracks, coarse, events, initial_clock)
    del tracks
    gc.collect()
    needs_audio = any(options.ridge == "demod" for _, _, options in plan_cells)
    # About 30 MB for the hour at 2048 Hz; sample k is spectrogram time k / 2048 s.
    audio = Audio(decode_samples(directory / "recording.webm"), 2048.0) if needs_audio else None
    weather = WeatherSeries.load(args.weather)
    # As in compare_oyster_bay_atmosphere.py: per-event profiles at the initial heard CPAs.
    layered = weather.for_events(events, chosen, initial_clock)
    atmospheres = {
        "constant": (-30.0, Atmosphere(343.0)),
        "layered": (GEOID_M + CAMERA_ABOVE_MSL_M, layered),
    }
    initial_latlon = [float(coarse["lat"]), float(coarse["lon"])]
    plan = {
        "validation": "Real recording, conditional refinement with published flight identities and coarse "
        "candidate; not a blind search. Truth enters only after every fit.",
        "protocol": "scripts/compare_oyster_bay.py shared start, calibrated once per atmosphere",
        "traces": args.traces,
        "weather_file": args.weather,
        "atmospheres": {k: {"height_m": h} for k, (h, _) in atmospheres.items()},
        "layered_profiles": layered.to_record(),
        "cells": {name: repr(options) for name, _, options in plan_cells},
        "initial_latlon": initial_latlon,
        "initial_clock": initial_clock,
        "flights": [t.id for t in chosen],
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/dopplertracker").glob("*.py")
        },
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    candidates = {}
    for key in sorted({a for _, a, _ in plan_cells}):
        height, atmosphere = atmospheres[key]
        observer = frame.to_local([*initial_latlon, height])
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
        candidates[key] = {
            "position": observer.tolist(),
            "clock": clock,
            "f0": f0,
            "flights": [t.id for t in chosen],
            "score": score,
        }
        print("Calibrated", key, "clock", clock, "f0", f0, "scores", scores, flush=True)
    rows = []
    for name, key, options in plan_cells:
        height, atmosphere = atmospheres[key]
        start = time.perf_counter()
        row = {"cell": name, "atmosphere": key, "height_m": height, "result": None, "error": None}
        row["initial_candidate"] = candidates[key]
        try:
            result = fit_projected(
                spec,
                events,
                chosen,
                candidates[key],
                options,
                atmosphere,
                frame=frame,
                elevation=ConstantElevation(height),
                audio=audio,
                keep_observations=True,
            )
            row["fit_seconds"] = time.perf_counter() - start
            result["evidence"] = assess_candidate(spec, events, chosen, result, atmosphere)
            result["lat_lon_height"] = frame.to_geodetic(result["position"]).tolist()
            row["valid_frames"] = [int(np.sum(o["valid"])) for o in result["observations"]]
            row["frames"] = [len(o["valid"]) for o in result["observations"]]
            row["result"] = result
        except Exception as exc:  # failures are outcomes
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
        (output / "fits.json").write_text(
            json.dumps({"plan": plan, "trials": rows}, indent=2, allow_nan=False)
        )
        print(name, "done", round(row["seconds"], 1), "s", row["error"] or "", flush=True)
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
        f"{'cell':36s} {'error_m':>8s} {'median|r| Hz':>15s} {'rms Hz':>15s} "
        f"{'valid frames':>16s} {'evidence':>8s} {'s':>6s}"
    )
    for row in rows:
        result = row["result"]
        if not result:
            print(f"{row['cell']:36s} FAILED {row['error']}")
            continue
        frames = "/".join(map(str, row["valid_frames"])) + " of " + "/".join(map(str, row["frames"]))
        print(
            f"{row['cell']:36s} {row['error_m']:8.1f} "
            f"{' '.join(f'{x:7.3f}' for x in result['event_residual_hz']):>15s} "
            f"{' '.join(f'{x:7.3f}' for x in result['event_residual_rms_hz']):>15s} "
            f"{frames:>16s} {'accept' if result['evidence']['accepted'] else 'reject':>8s} "
            f"{row['seconds']:6.1f}"
        )


if __name__ == "__main__":
    main()
