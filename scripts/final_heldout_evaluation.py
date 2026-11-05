"""Final held-out synthetic evaluation: a frozen method list on fresh seeds, run once.

Seeds 300-304 (300-307 for the detector) were never used during development
(seeds 0-12).  Nothing here may be tuned on them: methods, options, scenario
lists and thresholds are frozen below and recorded, with source hashes, in
``plan.json`` before anything runs.  Every harness keeps its own seed guard;
the held-out run passes ``split="held-out"`` to the stress harnesses and
``final_evaluation=True`` to the discovery benchmark.

A. Location, the 20 default stress scenarios: brightness, spline-linear, fused velocity.
B. Location, extended scenarios: spline-linear and fused velocity on the velocity
   scenes; on the layered scenes and ``realistic_combined`` the five
   configurations spline-linear (343 m/s), + forecast profile, fused velocity
   only, fused + forecast, + fit c and wind.
C. Linearised error regions (``uncertainty=True``) for spline-linear and fused
   velocity on the A and B scenes (noise_only excluded).
D. Detection: legacy whitened two-harmonic versus the calibrated detector.
E. Flyover discovery with its default scenarios and settings.
F. Audio measurement: STFT ridge versus demodulation with SNR noise; hum
   masking versus subtraction on ``hum_crossing``.

    python scripts/final_heldout_evaluation.py            # the one held-out run
    python scripts/final_heldout_evaluation.py --smoke    # dev seed 0 only, end-to-end check
    python scripts/final_heldout_evaluation.py --summary  # re-print tables from saved parts
    python scripts/final_heldout_evaluation.py --parts E  # rerun one harness in full after a fix

Run one at a time with ``nice``; workers=1 throughout.
"""

import argparse
import hashlib
import inspect
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

import dopplertracker
from dopplertracker.advanced import fit_projected
from dopplertracker.audio_benchmark import HUM_METHODS, MEASUREMENT_METHODS, run_measurement_benchmark
from dopplertracker.detection import FlyoverOptions
from dopplertracker.detection_benchmark import CALIBRATED_METHODS, _metrics, run_detection_benchmark
from dopplertracker.detection_benchmark import METHODS as DETECTION_METHODS
from dopplertracker.discovery import DiscoveryOptions
from dopplertracker.discovery_benchmark import DISCOVERY_SCENARIOS, run_discovery_benchmark
from dopplertracker.stress_benchmark import (
    ATMOSPHERE_METHODS,
    PROJECTED,
    VELOCITY_PROJECTED,
    run_layered_benchmark,
    run_stress_benchmark,
    summarize_trials,
)
from dopplertracker.stress_scenes import (
    COMBINED_SCENARIOS,
    EXTENDED_SCENARIOS,
    LAYERED_SCENARIOS,
    STRESS_SCENARIOS,
    VELOCITY_SCENARIOS,
    stress_scene,
)
from dopplertracker.uncertainty import evaluate

# ----- Frozen plan ---------------------------------------------------------------------------------

SEEDS = (300, 301, 302, 303, 304)
DETECTION_SEEDS = tuple(range(300, 308))
SMOKE_SEEDS = (0,)
SMOKE_DISCOVERY_SCENARIOS = ("clean", "low_snr", "noise_only")

FUSED = "kalman-velocity-linear"
A_METHODS = ("brightness", "spline-linear", FUSED)
B_LAYERED_SCENARIOS = LAYERED_SCENARIOS + COMBINED_SCENARIOS
B_LAYERED_METHODS = (
    "spline-linear",
    "spline-linear-forecast-profile",
    "kalman-velocity-linear",
    "kalman-velocity-forecast-profile",
    "kalman-velocity-forecast-fit-c-wind",
)
B_VELOCITY_SCENARIOS = VELOCITY_SCENARIOS
B_VELOCITY_METHODS = ("spline-linear", FUSED)
LABELS = {
    "brightness": "brightness (old baseline)",
    "spline-linear": "spline-linear, 343",
    "spline-linear-forecast-profile": "+ forecast profile",
    "kalman-velocity-linear": "fused velocity only",
    "kalman-velocity-forecast-profile": "fused + forecast",
    "kalman-velocity-forecast-fit-c-wind": "+ fit c and wind",
}
C_METHODS = {"spline-linear": PROJECTED["spline-linear"], FUSED: VELOCITY_PROJECTED[FUSED]}
C_A_SCENARIOS = tuple(s for s in STRESS_SCENARIOS if s != "noise_only")
C_SCENARIOS = C_A_SCENARIOS + EXTENDED_SCENARIOS
D_METHODS = tuple(m for m in DETECTION_METHODS if m[0] == "whitened-two-harmonics") + CALIBRATED_METHODS
D_SCENARIOS = inspect.signature(run_detection_benchmark).parameters["scenarios"].default
F_SCENARIOS = ("fast_chirp", "low_snr_audio", "clean_audio", "noisy_audio")
F_METHODS = ("stft-peak", "demod-snr")
F_HUM_SCENARIOS = ("hum_crossing",)
F_HUM_METHODS = ("demod-snr-mask", "demod-snr-subtract")
PARTS = "ABCDEF"

# (new, replaced) pairs checked per scenario for regressions.
A_PAIRS = (("spline-linear", "brightness"), (FUSED, "spline-linear"))
B_PAIRS = (
    (FUSED, "spline-linear"),
    ("spline-linear-forecast-profile", "spline-linear"),
    ("kalman-velocity-forecast-profile", FUSED),
    ("kalman-velocity-forecast-profile", "spline-linear-forecast-profile"),
    ("kalman-velocity-forecast-fit-c-wind", "kalman-velocity-forecast-profile"),
)


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3


def say(*parts):
    print(*parts, flush=True)


def plan(seeds, detection_seeds, discovery_scenarios, held_out):
    package = Path(dopplertracker.__file__).parent
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}
    sources[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=package)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=package)
    return {
        "purpose": "final held-out synthetic evaluation" if held_out else "development smoke test",
        "git_commit": git.stdout.strip(),
        "git_dirty": dirty.stdout.splitlines(),
        "seeds": list(seeds),
        "detection_seeds": list(detection_seeds),
        "workers": 1,
        "A": {
            "harness": "stress_benchmark.run_stress_benchmark",
            "scenarios": list(STRESS_SCENARIOS),
            "methods": list(A_METHODS),
            "options": {m: vars(PROJECTED.get(m) or VELOCITY_PROJECTED[m]) for m in A_METHODS[1:]},
        },
        "B": {
            "harness": "stress_benchmark.run_layered_benchmark",
            "layered_scenarios": list(B_LAYERED_SCENARIOS),
            "layered_methods": list(B_LAYERED_METHODS),
            "velocity_scenarios": list(B_VELOCITY_SCENARIOS),
            "velocity_methods": list(B_VELOCITY_METHODS),
            "options": {
                m: {
                    "options": vars(ATMOSPHERE_METHODS[m][0]),
                    "atmosphere": ATMOSPHERE_METHODS[m][1] or "343 m/s",
                }
                for m in B_LAYERED_METHODS
            },
            "labels": LABELS,
        },
        "C": {
            "harness": "fit_projected(..., uncertainty=True) + uncertainty.evaluate, default InputPriors",
            "scenarios": list(C_SCENARIOS),
            "methods": {m: vars(o) for m, o in C_METHODS.items()},
        },
        "D": {
            "harness": "detection_benchmark.run_detection_benchmark",
            "scenarios": list(D_SCENARIOS),
            "methods": [list(m) for m in D_METHODS],
            "calibrated_options": asdict(FlyoverOptions()),
        },
        "E": {
            "harness": "discovery_benchmark.run_discovery_benchmark",
            "scenarios": list(discovery_scenarios),
            "distractors": 8,
            "options": {
                k: list(v) if isinstance(v, tuple) else v for k, v in vars(DiscoveryOptions()).items()
            },
            "final_evaluation": held_out,
        },
        "F": {
            "harness": "audio_benchmark.run_measurement_benchmark (window 4 s)",
            "scenarios": list(F_SCENARIOS),
            "methods": {m: repr(MEASUREMENT_METHODS[m]) for m in F_METHODS},
            "hum_scenarios": list(F_HUM_SCENARIOS),
            "hum_methods": {m: repr(HUM_METHODS[m]) for m in F_HUM_METHODS},
        },
        "source_hashes": sources,
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "pillow")},
        },
    }


def _progress(done, total, scenario, seed):
    say(f"  {done}/{total} {scenario} {seed}")


# ----- Parts ---------------------------------------------------------------------------------------


def part_a(out, seeds, split, **_):
    report = run_stress_benchmark(out / "A", seeds, STRESS_SCENARIOS, A_METHODS, split, 1, _progress)
    tracks = [t for s in STRESS_SCENARIOS for seed in seeds for t in stress_scene(s, seed)["tracks"]]
    return {
        "scenes": len(STRESS_SCENARIOS) * len(seeds),
        "fits": len(report["trials"]),
        "tracks": len(tracks),
        "tracks_with_velocity": sum(t.velocities is not None for t in tracks),
        "summary": report["summary"],
        "by_scenario": report["by_scenario"],
    }


def part_b(out, seeds, split, **_):
    layered = run_layered_benchmark(
        out / "B-layered", seeds, B_LAYERED_SCENARIOS, B_LAYERED_METHODS, split, 1, _progress
    )
    velocity = run_layered_benchmark(
        out / "B-velocity", seeds, B_VELOCITY_SCENARIOS, B_VELOCITY_METHODS, split, 1, _progress
    )
    rows = layered["trials"] + velocity["trials"]
    return {
        "scenes": (len(B_LAYERED_SCENARIOS) + len(B_VELOCITY_SCENARIOS)) * len(seeds),
        "fits": len(rows),
        "layered_summary": layered["summary"],
        "velocity_summary": velocity["summary"],
        "all_scenes_summary": summarize_trials([r for r in rows if r["method"] in B_VELOCITY_METHODS]),
        "by_scenario": {**layered["by_scenario"], **velocity["by_scenario"]},
    }


def _region_trial(scene, method):
    row = {"scenario": scene["scenario"], "seed": scene["seed"], "method": method, "exception": None}
    start = time.perf_counter()
    try:
        result = fit_projected(
            scene["spectrogram"], scene["events"], scene["tracks"], scene["candidate"], C_METHODS[method],
            uncertainty=True,
        )  # fmt: skip
        error = np.asarray(result["position"])[:2] - scene["truth"][:2]
        row.update(position=result["position"], error_m=float(np.linalg.norm(error)))
        u = result["uncertainty"]
        if "error" in u:
            row["region_error"] = u["error"]
        else:
            row.update(
                evaluation=evaluate(u, error),
                semi_major_90_m=u["ellipses"]["90"]["semi_major_m"],
                dominant_cause=u["dominant_cause"],
            )
    except Exception as exc:  # failures are outcomes
        row["exception"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.perf_counter() - start
    return row


def _region_summary(rows):
    scored = [r for r in rows if "evaluation" in r]
    out = {
        "trials": len(rows),
        "exceptions": sum(r["exception"] is not None for r in rows),
        "region_failures": sum("region_error" in r for r in rows),
        "scored": len(scored),
    }
    for part in ("total", "measurement"):
        d = np.array([r["evaluation"][part]["normalised_error"] for r in scored])
        out[part] = {
            "inside_50": float(np.mean([r["evaluation"][part]["inside_50"] for r in scored]))
            if scored
            else None,
            "inside_90": float(np.mean([r["evaluation"][part]["inside_90"] for r in scored]))
            if scored
            else None,
            "d_median": float(np.median(d)) if len(d) else None,
            "d_p90": float(np.quantile(d, 0.9)) if len(d) else None,
            "d_max": float(np.max(d)) if len(d) else None,
        }
    out["median_semi_major_90_m"] = (
        float(np.median([r["semi_major_90_m"] for r in scored])) if scored else None
    )
    return out


def part_c(out, seeds, **_):
    directory = out / "C"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    with (directory / "trials.jsonl").open("w") as stream:
        for scenario in C_SCENARIOS:
            for seed in seeds:
                scene = stress_scene(scenario, seed)
                for method in C_METHODS:
                    row = _region_trial(scene, method)
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                say(f"  C {scenario} {seed}")
    summary = {}
    for method in C_METHODS:
        group = [r for r in rows if r["method"] == method]
        summary[method] = {
            "A_scenes": _region_summary([r for r in group if r["scenario"] in C_A_SCENARIOS]),
            "B_scenes": _region_summary([r for r in group if r["scenario"] in EXTENDED_SCENARIOS]),
            "all": _region_summary(group),
        }
    by_scenario = {
        s: {
            m: _region_summary([r for r in rows if r["scenario"] == s and r["method"] == m])
            for m in C_METHODS
        }
        for s in C_SCENARIOS
    }
    return {
        "scenes": len(C_SCENARIOS) * len(seeds),
        "fits": len(rows),
        "summary": summary,
        "by_scenario": by_scenario,
    }


def part_d(out, detection_seeds, **_):
    report = run_detection_benchmark(seeds=detection_seeds, scenarios=D_SCENARIOS, methods=D_METHODS)
    directory = out / "D"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    trials = report["trials"]
    names = [m[0] for m in D_METHODS]
    noise = {
        name: {
            "scenes": sum(r["method"] == name and r["scenario"] == "noise_only" for r in trials),
            "scenes_with_detection": sum(
                r["method"] == name and r["scenario"] == "noise_only" and bool(r["proposal_cpas_s"])
                for r in trials
            ),
            "detections": sum(
                len(r["proposal_cpas_s"])
                for r in trials
                if r["method"] == name and r["scenario"] == "noise_only"
            ),
        }
        for name in names
    }
    signal = {
        name: _metrics([r for r in trials if r["method"] == name and r["scenario"] != "noise_only"])
        for name in names
    }
    return {
        "scenes": len(D_SCENARIOS) * len(detection_seeds),
        "summary": report["summary"],
        "signal_only_summary": signal,
        "by_scenario": {
            s: {n: _metrics([r for r in trials if r["method"] == n and r["scenario"] == s]) for n in names}
            for s in D_SCENARIOS
        },
        "noise_only": noise,
    }


def part_e(out, seeds, discovery_scenarios, held_out, **_):
    report = run_discovery_benchmark(
        out / "E",
        seeds,
        discovery_scenarios,
        progress=lambda done, total, row: say(
            f"  {done}/{total} {row['scenario']} {row['seed']} {row['seconds']:.0f} s"
        ),
        final_evaluation=held_out,
    )
    wrong = [w for r in report["trials"] for w in r["wrong"]]
    by_kind = {}
    for kind in ("position", "clock", "both"):
        group = [w for w in wrong if w["kind"] == kind]
        by_kind[kind] = {
            "n": len(group),
            "with_any": sum(w["confirmed"] > 0 for w in group),
            "confirmed_total": int(sum(w["confirmed"] for w in group)),
            "confirmed_mean": float(np.mean([w["confirmed"] for w in group])) if group else None,
        }
    exceptions = [
        {
            "scenario": r["scenario"],
            "seed": r["seed"],
            "exception": r["exception"] or r.get("four_events_exception"),
        }
        for r in report["trials"]
        if r["exception"] or r.get("four_events_exception")
    ]
    return {
        "scenes": len(report["trials"]),
        "summary": report["summary"],
        "by_scenario": report["by_scenario"],
        "wrong_by_kind": by_kind,
        "exception_rows": exceptions,
    }


def _measurement_summary(report, scenarios, methods):
    out = {}
    for scenario in scenarios:
        for method in methods:
            rows = [r for r in report["trials"] if r["scenario"] == scenario and r["method"] == method]
            done = [r for r in rows if r["error"] is None]
            entry = {"trials": len(rows), "failures": len(rows) - len(done)}
            for key in ("location_error_m", "track_rms_hz", "track_cpa_rms_hz"):
                values = [r[key] for r in done if r.get(key) is not None]
                entry[key] = {
                    "median": float(np.median(values)) if values else None,
                    "max": float(np.max(values)) if values else None,
                }
            entry["errors"] = [r["error"] for r in rows if r["error"]]
            out[f"{scenario}/{method}"] = entry
    return out


def part_f(out, seeds, **_):
    measurement = run_measurement_benchmark(
        seeds, F_SCENARIOS, F_METHODS, str(out / "F-measurement"), progress=None
    )
    hum = run_measurement_benchmark(seeds, F_HUM_SCENARIOS, F_HUM_METHODS, str(out / "F-hum"), progress=None)
    return {
        "measurement": _measurement_summary(measurement, F_SCENARIOS, F_METHODS),
        "hum": _measurement_summary(hum, F_HUM_SCENARIOS, F_HUM_METHODS),
    }


RUNNERS = {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e, "F": part_f}
# Cheapest first, so a late crash leaves the most results.
ORDER = "ABCFDE"


# ----- Tables --------------------------------------------------------------------------------------


def _f(x, digits=1):
    return "-" if x is None else f"{x:.{digits}f}"


def _pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def location_table(summary, methods):
    lines = [
        f"{'method':38s} {'n':>3s} {'median':>8s} {'p90':>8s} {'worst':>8s} {'<=100m':>7s} {'acc&<=100':>9s} "
        f"{'exc':>4s} {'noconv':>6s} {'med s':>6s}"
    ]
    for m in methods:
        if m not in summary:
            continue
        s = summary[m]
        lines.append(
            f"{m + ' [' + LABELS.get(m, m) + ']':38.38s} {s['numeric_estimates']:3d} {_f(s['median_error_m']):>8s} "
            f"{_f(s['p90_error_m']):>8s} {_f(s['worst_error_m']):>8s} {_pct(s['within_100m_fraction']):>7s} "
            f"{_pct(s['accepted_within_100m_fraction']):>9s} {s['exceptions']:4d} {s['nonconverged']:6d} "
            f"{s['median_seconds']:6.2f}"
        )
    return lines


def scenario_table(by_scenario, scenarios, methods):
    short = {m: m.replace("kalman-velocity", "kv").replace("spline-linear", "sl")[:17] for m in methods}
    lines = [f"{'scenario (median m)':24s} " + " ".join(f"{short[m]:>17s}" for m in methods)]
    for scenario in scenarios:
        cells = []
        for m in methods:
            s = by_scenario.get(scenario, {}).get(m)
            cells.append(f"{_f(s['median_error_m']) if s else '-':>17s}")
        lines.append(f"{scenario:24s} " + " ".join(cells))
    return lines


def regressions(by_scenario, scenarios, pairs):
    lines = []
    for new, old in pairs:
        for scenario in scenarios:
            a, b = by_scenario.get(scenario, {}).get(new), by_scenario.get(scenario, {}).get(old)
            if not a or not b or a["median_error_m"] is None or b["median_error_m"] is None:
                continue
            if a["median_error_m"] > b["median_error_m"]:
                lines.append(
                    f"  {scenario}: {new} {a['median_error_m']:.2f} m > {old} {b['median_error_m']:.2f} m "
                    f"(worst {_f(a['worst_error_m'])} vs {_f(b['worst_error_m'])})"
                )
    return lines


def render(out, parts):
    text = []
    meta = parts.get("_meta", {})
    for key in ORDER:
        if key in meta:
            text.append(
                f"part {key}: {meta[key]['seconds']:.0f} s, peak RSS so far {meta[key]['peak_rss_mb']:.0f} MB"
            )
    if "A" in parts:
        a = parts["A"]
        text += ["", f"A. Default 20 scenarios: {a['scenes']} scenes, {a['fits']} fits"]
        text += location_table(a["summary"], A_METHODS)
        text.append(f"tracks carrying velocity reports: {a['tracks_with_velocity']}/{a['tracks']}")
        text += scenario_table(a["by_scenario"], STRESS_SCENARIOS, A_METHODS)
        text += [
            "A regressions (new median > replaced median):",
            *regressions(a["by_scenario"], STRESS_SCENARIOS, A_PAIRS),
        ]
    if "B" in parts:
        b = parts["B"]
        text += ["", f"B. Extended scenarios: {b['scenes']} scenes, {b['fits']} fits"]
        text.append("layered scenes + realistic_combined:")
        text += location_table(b["layered_summary"], B_LAYERED_METHODS)
        text.append("velocity scenes:")
        text += location_table(b["velocity_summary"], B_VELOCITY_METHODS)
        text.append("all 7 extended scenes:")
        text += location_table(b["all_scenes_summary"], B_VELOCITY_METHODS)
        text += scenario_table(
            b["by_scenario"], B_LAYERED_SCENARIOS + B_VELOCITY_SCENARIOS, B_LAYERED_METHODS
        )
        text += [
            "B regressions (new median > replaced median):",
            *regressions(b["by_scenario"], B_LAYERED_SCENARIOS + B_VELOCITY_SCENARIOS, B_PAIRS),
        ]
    if "C" in parts:
        c = parts["C"]
        text += ["", f"C. Error regions: {c['scenes']} scenes, {c['fits']} fits (Rayleigh(1): d50 1.18)"]
        text.append(f"{'method/scope':34s} {'n':>4s} {'exc':>4s} {'nreg':>4s} {'in50':>5s} {'in90':>5s} {'d50':>6s} {'d90':>6s} "
                    f"{'meas in50':>9s} {'meas in90':>9s} {'meas d50':>8s} {'a90 m':>7s}")  # fmt: skip
        for method, scopes in c["summary"].items():
            for scope, s in scopes.items():
                t, m = s["total"], s["measurement"]
                text.append(
                    f"{method + ' ' + scope:34s} {s['scored']:4d} {s['exceptions']:4d} {s['region_failures']:4d} "
                    f"{_f(t['inside_50'], 2):>5s} {_f(t['inside_90'], 2):>5s} {_f(t['d_median'], 2):>6s} "
                    f"{_f(t['d_p90'], 2):>6s} {_f(m['inside_50'], 2):>9s} {_f(m['inside_90'], 2):>9s} "
                    f"{_f(m['d_median'], 2):>8s} {_f(s['median_semi_major_90_m']):>7s}"
                )
        text.append(f"{'scenario: total in50/in90/d50':32s} " + "  ".join(f"{m:>22s}" for m in C_METHODS))
        for scenario, methods in c["by_scenario"].items():
            cells = []
            for s in methods.values():
                t = s["total"]
                cells.append(
                    f"{_f(t['inside_50'], 2)}/{_f(t['inside_90'], 2)}/{_f(t['d_median'], 2):>5s}".rjust(22)
                )
            text.append(f"{scenario:32s} " + "  ".join(cells))
    if "D" in parts:
        d = parts["D"]
        text += ["", f"D. Detection: {d['scenes']} scenes"]
        for scope, table in (("all scenes", d["summary"]), ("signal scenes", d["signal_only_summary"])):
            for name, s in table.items():
                text.append(
                    f"{scope:13s} {name:26s} recall {_f(s['recall'], 3)} precision {_f(s['precision'], 3)} "
                    f"matched {s['matched']}/{s['truth_events']} proposals {s['proposals']} "
                    f"cpa err {_f(s['median_cpa_error_s'], 2)} s f0 err {_f(s['median_abs_f0_error_hz'], 2)} Hz "
                    f"med runtime {_f(s['median_runtime_s'])} s"
                )
        for scenario, table in d["by_scenario"].items():
            for name, s in table.items():
                text.append(
                    f"  {scenario:22s} {name:26s} recall {_f(s['recall'], 3)} precision {_f(s['precision'], 3)} "
                    f"proposals {s['proposals']} cpa {_f(s['median_cpa_error_s'], 2)} f0 {_f(s['median_abs_f0_error_hz'], 2)}"
                )
        for name, s in d["noise_only"].items():
            text.append(
                f"noise_only {name}: {s['detections']} detections in {s['scenes_with_detection']}/{s['scenes']} scenes"
            )
    if "E" in parts:
        e = parts["E"]
        s = e["summary"]
        text += [
            "",
            f"E. Discovery: {e['scenes']} scenes ({s['signal_scenes']} with signal), exceptions {s['exceptions']}",
        ]
        for stage in ("first", "second"):
            text.append(
                f"{stage} pass: recall {_f(s[f'{stage}_recall'], 3)} precision {_f(s[f'{stage}_precision'], 3)} "
                f"false positives {s[f'{stage}_false_positives']}"
            )
        for key, label in (("two_events_error_m", "two events only"), ("joint_error_m", "discovered + joint refit"),
                           ("four_events_error_m", "all four supplied")):  # fmt: skip
            q = s[key]
            text.append(
                f"{label:26s} n {q['n']:3d} median {_f(q['median'])} p90 {_f(q['p90'])} max {_f(q['max'])} m"
            )
        text.append(
            f"at truth: mean confirmed unseen flights {_f(s['truth_confirmed_mean'], 2)} of 2, distribution "
            f"{s['truth_confirmed_distribution']}, false confirmations {s['truth_false_confirmations']}"
        )
        for kind, k in e["wrong_by_kind"].items():
            text.append(
                f"wrong {kind:8s}: {k['with_any']}/{k['n']} with any confirmation, total {k['confirmed_total']}, "
                f"mean {_f(k['confirmed_mean'], 2)}"
            )
        text.append(f"noise_only false confirmations {s['noise_only_false_confirmations']}")
        for scenario, t in e["by_scenario"].items():
            text.append(
                f"  {scenario:22s} recall {_f(t['second_recall'], 2)} prec {_f(t['second_precision'], 2)} two "
                f"{_f(t['two_events_error_m']['median'])} joint {_f(t['joint_error_m']['median'])} four "
                f"{_f(t['four_events_error_m']['median'])} truth conf {_f(t['truth_confirmed_mean'], 2)} "
                f"wrong any {t['wrong_with_any_confirmation']}/{t['wrong_candidates']}"
            )
        for row in e["exception_rows"]:
            text.append(f"  exception {row['scenario']} {row['seed']}: {row['exception']}")
    if "F" in parts:
        f = parts["F"]
        text += ["", "F. Audio measurement (medians/max over seeds)"]
        text.append(
            f"{'scene/method':34s} {'fail':>4s} {'loc med':>8s} {'loc max':>8s} {'trk rms':>8s} {'trk max':>8s} {'cpa rms':>8s}"
        )
        for name, s in {**f["measurement"], **f["hum"]}.items():
            text.append(
                f"{name:34s} {s['failures']:4d} {_f(s['location_error_m']['median'], 2):>8s} "
                f"{_f(s['location_error_m']['max'], 2):>8s} {_f(s['track_rms_hz']['median'], 4):>8s} "
                f"{_f(s['track_rms_hz']['max'], 4):>8s} {_f(s['track_cpa_rms_hz']['median'], 4):>8s}"
            )
            for error in s["errors"]:
                text.append(f"    error: {error}")
    (out / "summary.txt").write_text("\n".join(text) + "\n")
    (out / "summary.json").write_text(json.dumps(parts, indent=2, allow_nan=False))
    say("\n".join(text))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true", help="Development seed 0 only (end-to-end check)")
    parser.add_argument("--parts", default=PARTS, help="Subset of ABCDEF to (re)run in full")
    parser.add_argument("--summary", action="store_true", help="Only re-print tables from saved parts")
    args = parser.parse_args()
    held_out = not args.smoke
    out = Path(args.out or ("outputs/final-heldout" if held_out else "outputs/final-heldout-smoke"))
    out.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS if held_out else SMOKE_SEEDS
    detection_seeds = DETECTION_SEEDS if held_out else SMOKE_SEEDS
    discovery_scenarios = DISCOVERY_SCENARIOS if held_out else SMOKE_DISCOVERY_SCENARIOS
    parts_file = out / "parts.json"
    parts = json.loads(parts_file.read_text()) if parts_file.exists() else {}
    if args.summary:
        render(out, parts)
        return
    selected = [p for p in ORDER if p in args.parts.upper()]
    if not selected:
        raise SystemExit("--parts must name some of ABCDEF")
    record = plan(seeds, detection_seeds, discovery_scenarios, held_out)
    record["parts"] = selected
    # The first invocation writes plan.json; a rerun of single parts after a fix gets its own plan file.
    name = (
        "plan.json"
        if not (out / "plan.json").exists()
        else f"plan-rerun-{''.join(selected)}-{int(time.time())}.json"
    )
    (out / name).write_text(json.dumps(record, indent=2))
    say(f"wrote {out / name}")
    context = dict(
        out=out,
        seeds=seeds,
        detection_seeds=detection_seeds,
        discovery_scenarios=discovery_scenarios,
        held_out=held_out,
        split="held-out" if held_out else "development",
    )
    parts.setdefault("_meta", {})
    for key in selected:
        say(f"part {key} ...")
        started = time.perf_counter()
        parts[key] = RUNNERS[key](**context)
        parts["_meta"][key] = {"seconds": time.perf_counter() - started, "peak_rss_mb": peak_rss_mb()}
        parts_file.write_text(json.dumps(parts, indent=2, allow_nan=False))
        say(f"part {key} done in {parts['_meta'][key]['seconds']:.0f} s, peak RSS {peak_rss_mb():.0f} MB")
    render(out, parts)


if __name__ == "__main__":
    main()
