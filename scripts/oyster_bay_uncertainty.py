"""Error regions for every cell of scripts/compare_oyster_bay_combined.py.

Same protocol, cells, start and calibration as that script (imported from it;
its own numbers are unchanged): each fit additionally gets
``uncertainty=True`` (``dopplertracker.uncertainty``).  Truth is read only after
every fit and every error region has been computed, and is used only to report
the error, the normalised error d = sqrt(e^T C^-1 e), and whether the camera lies
inside the 50 % and 90 % ellipses.  Also reported: whether the cells' estimates
are mutually consistent given their ellipses (pairwise d with C_a + C_b; the
cells share their data, so this test is lenient: failing it shows a real
inconsistency, passing it does not prove agreement), and optionally a
moving-block bootstrap of the measurement part for chosen cells.
"""

import argparse
import gc
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_oyster_bay_atmosphere import CAMERA_ABOVE_MSL_M, GEOID_M  # noqa: E402
from compare_oyster_bay_combined import cells  # noqa: E402
from compare_oyster_bay_velocity import load_flights, select_tracks  # noqa: E402

from dopplertracker.advanced import fit_projected  # noqa: E402
from dopplertracker.atmosphere import WeatherSeries  # noqa: E402
from dopplertracker.demod import Audio  # noqa: E402
from dopplertracker.detection import load_events  # noqa: E402
from dopplertracker.evidence import assess_candidate  # noqa: E402
from dopplertracker.physics import Atmosphere  # noqa: E402
from dopplertracker.scoring import joint_score  # noqa: E402
from dopplertracker.signal import Spectrogram, decode_samples  # noqa: E402
from dopplertracker.terrain import ConstantElevation  # noqa: E402
from dopplertracker.uncertainty import CHI2_2, GROUPS, block_bootstrap, evaluate, predicted_uncertainty  # noqa: E402

BASELINE = "constant/spline/ridge"


def has_region(row):
    return bool(row["result"]) and "covariance_en_m2" in row["result"]["uncertainty"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/oyster-bay")
    parser.add_argument("--traces", default="data/oyster-bay/traces-velocity.npz")
    parser.add_argument("--weather", default="data/oyster-bay/weather/open-meteo-2026-09-08.json")
    parser.add_argument("--out", default="outputs/oyster-bay/uncertainty")
    parser.add_argument("--cells", nargs="+", help="Subset of cell names (default: all)")
    parser.add_argument("--bootstrap", nargs="*", default=[BASELINE], help="Cells to block-bootstrap")
    parser.add_argument("--resamples", type=int, default=100)
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
    audio = Audio(decode_samples(directory / "recording.webm"), 2048.0) if needs_audio else None
    weather = WeatherSeries.load(args.weather)
    layered = weather.for_events(events, chosen, initial_clock)
    atmospheres = {
        "constant": (-30.0, Atmosphere(343.0)),
        "layered": (GEOID_M + CAMERA_ABOVE_MSL_M, layered),
    }
    initial_latlon = [float(coarse["lat"]), float(coarse["lon"])]
    plan = {
        "validation": "Real recording, conditional refinement with published flight identities and coarse "
        "candidate; not a blind search. Truth enters only after every fit and error region.",
        "protocol": "scripts/compare_oyster_bay_combined.py cells, plus uncertainty=True",
        "cells": {name: repr(options) for name, _, options in plan_cells},
        "flights": [t.id for t in chosen],
        "altitude_sources": [t.metadata.get("altitude_sources") for t in chosen],
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/dopplertracker").glob("*.py")
        },
    }
    candidates = {}
    for key in sorted({a for _, a, _ in plan_cells}):
        height, atmosphere = atmospheres[key]
        observer = frame.to_local([*initial_latlon, height])
        score, clock, f0, _ = joint_score(
            spec, events, chosen, observer, initial_clock, atmosphere=atmosphere,
            shift_radius=4, shift_steps=17, frequency_radius=12, frequency_steps=121,
        )  # fmt: skip
        candidates[key] = {
            "position": observer.tolist(),
            "clock": clock,
            "f0": f0,
            "flights": [t.id for t in chosen],
            "score": score,
        }
    rows = []
    for name, key, options in plan_cells:
        height, atmosphere = atmospheres[key]
        start = time.perf_counter()
        row = {"cell": name, "atmosphere": key, "height_m": height, "result": None, "error": None}
        try:
            result = fit_projected(
                spec, events, chosen, candidates[key], options, atmosphere, frame=frame,
                elevation=ConstantElevation(height), audio=audio, keep_observations=True, uncertainty=True,
            )  # fmt: skip
            result["evidence"] = assess_candidate(spec, events, chosen, result, atmosphere)
            if name in (args.bootstrap or []):
                row["bootstrap"] = block_bootstrap(
                    result, events, chosen, frame, ConstantElevation(height), resamples=args.resamples
                )
                row["bootstrap"].pop("shifts_en_m")
                # What-if helper on the same design, with this fit's noise and correlation:
                # it should reproduce the fitted region when nothing is new.
                region = result["uncertainty"]
                sigma = [
                    np.median(np.asarray(o["sigma_hz"])[np.asarray(o["valid"])])
                    for o in result["observations"]
                ]
                row["what_if_same_design"] = predicted_uncertainty(
                    events, chosen, result["position"], result["clock"], result["f0"], options, atmosphere,
                    frame, ConstantElevation(height),
                    noise_hz=[s * e["noise_scale"] for s, e in zip(sigma, region["events"], strict=True)],
                    lag1_correlation=[e["lag1_correlation"] for e in region["events"]],
                )  # fmt: skip
            row["result"] = result
        except Exception as exc:  # failures are outcomes
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
        (output / "fits.json").write_text(
            json.dumps({"plan": plan, "trials": rows}, indent=2, allow_nan=False)
        )
        print(name, "done", round(row["seconds"], 1), "s", row["error"] or "", flush=True)
    # Evaluation only: fits and error regions above never receive the camera coordinates.
    truth = json.loads((directory / "published/provenance.json").read_text())["evaluation_truth"]
    truth_position = frame.to_local([*truth, 0.0])
    for row in rows:
        result = row["result"]
        if not has_region(row):
            continue
        error = np.asarray(result["position"])[:2] - truth_position[:2]
        row["error_en_m"] = error.tolist()
        row["error_m"] = float(np.linalg.norm(error))
        row["evaluation"] = evaluate(result["uncertainty"], error)
    good = [r for r in rows if has_region(r)]
    consistency = []
    for a, b in itertools.combinations(good, 2):
        delta = np.asarray(a["result"]["position"][:2]) - np.asarray(b["result"]["position"][:2])
        joint = np.asarray(a["result"]["uncertainty"]["covariance_en_m2"]) + np.asarray(
            b["result"]["uncertainty"]["covariance_en_m2"]
        )
        d = float(np.sqrt(delta @ np.linalg.pinv(joint) @ delta))
        consistency.append(
            {
                "cells": [a["cell"], b["cell"]],
                "separation_m": float(np.linalg.norm(delta)),
                "normalised_separation": d,
                "consistent_90": bool(d * d <= CHI2_2[0.9]),
            }
        )
    report = {"plan": plan, "evaluation_truth": truth, "trials": rows, "consistency": consistency}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = [
        f"{'cell':36s} {'err m':>6s} {'a90 m':>6s} {'b90 m':>6s} {'brg':>4s} {'in50':>5s} {'in90':>5s} "
        f"{'d':>5s} {'d_meas':>6s}  variance fractions ({', '.join(g[:5] for g in GROUPS)})"
    ]
    for row in rows:
        if not has_region(row):
            reason = row["error"] or row["result"]["uncertainty"].get("error")
            lines.append(f"{row['cell']:36s} FAILED {reason}")
            continue
        u = row["result"]["uncertainty"]
        e90, ev = u["ellipses"]["90"], row["evaluation"]
        fractions = " ".join(f"{u['breakdown'][g]['fraction']:.2f}" for g in GROUPS)
        lines.append(
            f"{row['cell']:36s} {row['error_m']:6.1f} {e90['semi_major_m']:6.1f} {e90['semi_minor_m']:6.1f} "
            f"{e90['major_axis_bearing_deg']:4.0f} {str(ev['total']['inside_50']):>5s} "
            f"{str(ev['total']['inside_90']):>5s} {ev['total']['normalised_error']:5.2f} "
            f"{ev['measurement']['normalised_error']:6.2f}  {fractions}"
        )
    lines.append("\nPairwise consistency (lenient: C_a + C_b); inconsistent pairs:")
    bad = [c for c in consistency if not c["consistent_90"]]
    lines.extend(
        f"  {c['cells'][0]} vs {c['cells'][1]}: {c['separation_m']:.1f} m, d {c['normalised_separation']:.2f}"
        for c in bad
    )
    lines.append(f"  {len(bad)} of {len(consistency)} pairs inconsistent at 90 %")
    for row in rows:
        if row.get("bootstrap") and has_region(row):
            b = row["bootstrap"]["ellipses"]["90"]
            m = row["result"]["uncertainty"]["measurement_ellipses"]["90"]
            lines.append(
                f"\nBootstrap {row['cell']} ({row['bootstrap']['resamples']} resamples, "
                f"{row['bootstrap']['block_seconds']:.0f} s blocks): 90 % {b['semi_major_m']:.1f} x "
                f"{b['semi_minor_m']:.1f} m at {b['major_axis_bearing_deg']:.0f} deg; linearised measurement "
                f"part {m['semi_major_m']:.1f} x {m['semi_minor_m']:.1f} m at {m['major_axis_bearing_deg']:.0f} deg"
            )
        if row.get("what_if_same_design") and has_region(row):
            w, f = (
                row["what_if_same_design"]["ellipses"]["90"],
                row["result"]["uncertainty"]["ellipses"]["90"],
            )
            lines.append(
                f"What-if helper, same design ({row['cell']}): 90 % {w['semi_major_m']:.1f} x "
                f"{w['semi_minor_m']:.1f} m at {w['major_axis_bearing_deg']:.0f} deg; fitted region "
                f"{f['semi_major_m']:.1f} x {f['semi_minor_m']:.1f} m at {f['major_axis_bearing_deg']:.0f} deg"
            )
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
