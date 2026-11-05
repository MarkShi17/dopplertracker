"""Calibration of the linearised error regions (``dopplertracker.uncertainty``) on synthetic scenes.

Development seeds only (0-9; seeds >= 100 are refused).  For every scene and
method the fit gets ``uncertainty=True``; truth is used only afterwards, to
compute the normalised error d = sqrt(e^T C^-1 e) and whether the true
position lies inside the 50 % and 90 % ellipses, both for the full region and
for its measurement part alone.  An honest region gives d ~ Rayleigh(1):
median 1.18, 90th percentile 2.15, and 50 % / 90 % coverage.

Methods: spline-linear and kalman-velocity-linear (fused velocity) on the
default 20 scenarios (without noise_only) and the extended scenarios, plus
spline-linear with the forecast-like layered profile on the layered scenes
(the measured-weather priors).  Synthetic scenes carry no ADS-B timing,
altitude or observer-height error unless the scenario injects one, so the full
region is expected to over-cover on the clean controls.

    python scripts/uncertainty_calibration.py            # run and summarise
    python scripts/uncertainty_calibration.py --summary  # re-summarise trials.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dopplertracker.advanced import ModelOptions, fit_projected
from dopplertracker.stress_scenes import (
    COMBINED_SCENARIOS,
    EXTENDED_SCENARIOS,
    LAYERED_SCENARIOS,
    STRESS_SCENARIOS,
    stress_scene,
)
from dopplertracker.uncertainty import CHI2_2, GROUPS, block_bootstrap, evaluate, mahalanobis

METHODS = {
    "spline-linear": (ModelOptions(degree=1, smoothing="spline"), None),
    "kalman-velocity-linear": (ModelOptions(degree=1, velocity="fused"), None),
    "spline-linear-forecast-profile": (ModelOptions(degree=1, smoothing="spline"), "atmosphere_forecast"),
}
SCENARIOS = tuple(s for s in STRESS_SCENARIOS if s != "noise_only") + EXTENDED_SCENARIOS
PROFILE_SCENARIOS = LAYERED_SCENARIOS + COMBINED_SCENARIOS
# Rayleigh(1) quantiles of d for an honest 2-D Gaussian region.
RAYLEIGH = {q: float(np.sqrt(-2 * np.log(1 - q))) for q in (0.5, 0.9, 0.95)}
# The cause whose prior an injected fault tests (evaluated as measurement + that cause).
FAULT_CAUSE = {
    "wind": "atmosphere",
    "sound_speed": "atmosphere",
    "layered_atmosphere": "atmosphere",
    "high_altitude_layered": "atmosphere",
    "realistic_combined": "atmosphere",
    "altitude_bias": "altitude",
    "adsb_latency": "adsb_timing",
    "adsb_noise": "velocity",
    "adsb_noise_velocity": "velocity",
    "irregular_cadence": "velocity",
    "turning_sparse": "velocity",
}
# Injected faults in the model's input units (truth minus what the fit assumes).
KNOWN_FAULTS = {
    "wind": {"wind_e": 13.0, "wind_n": -8.0},
    "sound_speed": {"sound_speed": 331.0 - 343.0},
    "altitude_bias": {f"aircraft_altitude_{i}": -180.0 for i in range(4)},
    "adsb_latency": {"track_offset_2": 0.6},
}


def trial(scene, method):
    options, key = METHODS[method]
    extra = {} if key is None else {"atmosphere": scene[key]}
    row = {"scenario": scene["scenario"], "seed": scene["seed"], "method": method, "exception": None}
    start = time.perf_counter()
    try:
        result = fit_projected(
            scene["spectrogram"], scene["events"], scene["tracks"], scene["candidate"], options,
            uncertainty=True, **extra,
        )  # fmt: skip
        u = result["uncertainty"]
        error = np.asarray(result["position"])[:2] - scene["truth"][:2]
        row.update(
            error_en_m=error.tolist(),
            error_m=float(np.linalg.norm(error)),
            success=result["success"],
            at_bounds=result["at_bounds"],
            covariance_en_m2=u["covariance_en_m2"],
            ellipse_90=u["ellipses"]["90"],
            variance_m2={g: u["breakdown"][g]["variance_m2"] for g in GROUPS},
            dominant_cause=u["dominant_cause"],
            noise_scale=[e.get("noise_scale") for e in u["events"]],
            lag1_correlation=[e.get("lag1_correlation") for e in u["events"]],
            evaluation=evaluate(u, error),
        )
        measurement = np.asarray(u["breakdown"]["measurement"]["covariance_en_m2"])
        partial = {}
        for cause in GROUPS[1:]:
            d = mahalanobis(measurement + np.asarray(u["breakdown"][cause]["covariance_en_m2"]), error)
            partial[cause] = {"normalised_error": d, "inside_90": bool(d * d <= CHI2_2[0.9])}
        row["with_one_cause"] = partial
        faults = KNOWN_FAULTS.get(scene["scenario"])
        if faults and all(k in u["considered_inputs"] for k in faults):
            # Linear prediction of the error the injected fault causes: -G kappa.
            predicted = -sum(
                np.asarray(u["considered_inputs"][k]["position_shift_en_m"])
                / u["considered_inputs"][k]["prior_sigma"]
                * value
                for k, value in faults.items()
            )
            row["fault_predicted_error_en_m"] = predicted.tolist()
    except Exception as exc:  # failures are outcomes
        row["exception"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.perf_counter() - start
    return row


def summarise(rows):
    out = {}
    numeric = [r for r in rows if r["exception"] is None]
    for part in ("total", "measurement"):
        d = np.array([r["evaluation"][part]["normalised_error"] for r in numeric])
        out[part] = {
            "inside_50": float(np.mean([r["evaluation"][part]["inside_50"] for r in numeric]))
            if numeric
            else None,
            "inside_90": float(np.mean([r["evaluation"][part]["inside_90"] for r in numeric]))
            if numeric
            else None,
            "d_median": float(np.median(d)) if len(d) else None,
            "d_p90": float(np.quantile(d, 0.9)) if len(d) else None,
            "d_p95": float(np.quantile(d, 0.95)) if len(d) else None,
            "d_max": float(np.max(d)) if len(d) else None,
        }
    causes = {FAULT_CAUSE.get(r["scenario"]) for r in numeric}
    if len(causes) == 1 and None not in causes:
        cause = causes.pop()
        d = np.array([r["with_one_cause"][cause]["normalised_error"] for r in numeric])
        out["with_fault_cause"] = {
            "cause": cause,
            "inside_90": float(np.mean(d * d <= CHI2_2[0.9])),
            "d_median": float(np.median(d)),
            "d_p90": float(np.quantile(d, 0.9)),
        }
    checked = [r for r in numeric if "fault_predicted_error_en_m" in r]
    if checked and len({r["scenario"] for r in rows}) == 1:
        predicted = np.array([r["fault_predicted_error_en_m"] for r in checked])
        actual = np.array([r["error_en_m"] for r in checked])
        out["fault_linear_check"] = {
            "median_actual_error_m": float(np.median(np.linalg.norm(actual, axis=1))),
            "median_predicted_error_m": float(np.median(np.linalg.norm(predicted, axis=1))),
            "median_actual_minus_predicted_m": float(np.median(np.linalg.norm(actual - predicted, axis=1))),
        }
    out["trials"] = len(rows)
    out["exceptions"] = len(rows) - len(numeric)
    out["median_error_m"] = float(np.median([r["error_m"] for r in numeric])) if numeric else None
    out["median_semi_major_90_m"] = (
        float(np.median([r["ellipse_90"]["semi_major_m"] for r in numeric])) if numeric else None
    )
    out["dominant_cause"] = {
        g: sum(r["dominant_cause"] == g for r in numeric)
        for g in GROUPS
        if any(r["dominant_cause"] == g for r in numeric)
    }
    return out


def table(summary, label):
    lines = [
        f"\n{label}",
        f"{'':34s} {'n':>3s} | {'total: in50 in90 d50  d90':>26s} | {'meas: in50 in90 d50   d90':>27s} | "
        f"{'err50':>6s} {'a90':>6s} {'dominant':11s} | meas+fault cause: in90 d50",
    ]
    for name, s in summary.items():
        t, m = s["total"], s["measurement"]
        if t["d_median"] is None:
            lines.append(f"{name:34s} {s['trials']:3d} | all failed")
            continue
        dominant = max(s["dominant_cause"], key=s["dominant_cause"].get) if s["dominant_cause"] else "-"
        lines.append(
            f"{name:34s} {s['trials'] - s['exceptions']:3d} | {t['inside_50']:5.2f} {t['inside_90']:4.2f} "
            f"{t['d_median']:4.2f} {t['d_p90']:5.2f} | {m['inside_50']:5.2f} {m['inside_90']:4.2f} "
            f"{m['d_median']:5.2f} {m['d_p90']:6.2f} | {s['median_error_m']:6.1f} "
            f"{s['median_semi_major_90_m']:6.1f} {dominant:11s}"
            + (
                f" | {s['with_fault_cause']['cause']}: {s['with_fault_cause']['inside_90']:.2f} "
                f"{s['with_fault_cause']['d_median']:.2f}"
                if "with_fault_cause" in s
                else ""
            )
            + (
                " | linear check: actual {median_actual_error_m:.1f} m, predicted "
                "{median_predicted_error_m:.1f} m, |diff| {median_actual_minus_predicted_m:.1f} m".format(
                    **s["fault_linear_check"]
                )
                if "fault_linear_check" in s
                else ""
            )
        )
    return "\n".join(lines)


def report(rows, output):
    summary = {"rayleigh_reference": RAYLEIGH, "methods": {}, "by_scenario": {}}
    default = set(STRESS_SCENARIOS)
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        if not group:
            continue
        summary["methods"][method] = {
            "all": summarise(group),
            "default_scenarios": summarise([r for r in group if r["scenario"] in default]),
            "extended_scenarios": summarise([r for r in group if r["scenario"] not in default]),
        }
        summary["by_scenario"][method] = {
            s: summarise([r for r in group if r["scenario"] == s])
            for s in dict.fromkeys(r["scenario"] for r in group)
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    text = [f"Rayleigh(1) reference: d50 {RAYLEIGH[0.5]:.2f}, d90 {RAYLEIGH[0.9]:.2f}; in50 0.50, in90 0.90"]
    text.append(table({m: s["all"] for m, s in summary["methods"].items()}, "All scenarios"))
    for part in ("default_scenarios", "extended_scenarios"):
        text.append(table({m: s[part] for m, s in summary["methods"].items()}, part.replace("_", " ")))
    for method, scenarios in summary["by_scenario"].items():
        text.append(table(scenarios, f"By scenario: {method}"))
    (output / "summary.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


def bootstrap(scenarios, seeds, resamples, output):
    """Moving-block bootstrap (10 s blocks) vs the linearised measurement part; no truth used."""
    options = METHODS["spline-linear"][0]
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            scene = stress_scene(scenario, seed)
            args = (scene["spectrogram"], scene["events"], scene["tracks"], scene["candidate"], options)
            result = fit_projected(*args, keep_observations=True, uncertainty=True)
            boot = block_bootstrap(result, scene["events"], scene["tracks"], resamples=resamples)
            linear = result["uncertainty"]["measurement_ellipses"]["90"]
            b = boot["ellipses"]["90"]
            rows.append({"scenario": scenario, "seed": seed, "linear_90": linear, "bootstrap_90": b})
            print(
                f"{scenario} {seed}: bootstrap 90 % {b['semi_major_m']:.2f} x {b['semi_minor_m']:.2f} m at "
                f"{b['major_axis_bearing_deg']:.0f} deg; linearised measurement {linear['semi_major_m']:.2f} x "
                f"{linear['semi_minor_m']:.2f} m at {linear['major_axis_bearing_deg']:.0f} deg",
                flush=True,
            )
    (output / "bootstrap.json").write_text(json.dumps(rows, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="outputs/uncertainty/calibration")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--summary", action="store_true", help="Only summarise an existing trials.jsonl")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Instead: block-bootstrap the measurement part (spline-linear) for the given scenes",
    )
    parser.add_argument("--resamples", type=int, default=100)
    args = parser.parse_args()
    if any(seed >= 100 or seed < 0 for seed in args.seeds):
        raise SystemExit("development calibration uses seeds 0-99 only")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if args.bootstrap:
        bootstrap(args.scenarios, args.seeds, args.resamples, output)
        return
    log = output / "trials.jsonl"
    if not args.summary:
        done = set()
        if log.exists():
            for line in log.read_text().splitlines():
                r = json.loads(line)
                done.add((r["scenario"], r["seed"], r["method"]))
        with log.open("a") as stream:
            for scenario in args.scenarios:
                for seed in args.seeds:
                    methods = [
                        m
                        for m in args.methods
                        if (scenario, seed, m) not in done
                        and (METHODS[m][1] is None or scenario in PROFILE_SCENARIOS)
                    ]
                    if not methods:
                        continue
                    scene = stress_scene(scenario, seed)
                    for method in methods:
                        row = trial(scene, method)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        e = row.get("evaluation", {}).get("total", {})
                        print(scenario, seed, method, row.get("error_m"), e.get("normalised_error"),
                              row["exception"] or "", flush=True)  # fmt: skip
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    rows = [r for r in rows if r["seed"] in args.seeds and r["scenario"] in args.scenarios]
    report(rows, output)


if __name__ == "__main__":
    main()
