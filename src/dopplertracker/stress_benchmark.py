"""Broad ablations with untouched-seed evaluation and failure-preserving reports."""

import csv
import hashlib
import html
import json
import multiprocessing
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .advanced import ModelOptions, fit_consensus, fit_projected, select_model
from .evidence import assess_candidate
from .refine import refine_candidate, refine_spectral
from .signal import Spectrogram
from .stress_scenes import EXTENDED_SCENARIOS, LAYERED_SCENARIOS, STRESS_SCENARIOS, stress_scene

PROJECTED = {
    "projected-constant": ModelOptions(degree=0),
    "projected-linear": ModelOptions(degree=1),
    "projected-quadratic": ModelOptions(degree=2),
    "viterbi-linear": ModelOptions(degree=1, ridge="viterbi"),
    "ordinary-least-squares": ModelOptions(degree=1, loss="linear", robust_source=False),
    "cauchy-linear": ModelOptions(degree=1, loss="cauchy"),
    "huber-linear": ModelOptions(degree=1, loss="huber"),
    "spline-constant": ModelOptions(degree=0, smoothing="spline"),
    "spline-linear": ModelOptions(degree=1, smoothing="spline"),
    "spline-quadratic": ModelOptions(degree=2, smoothing="spline"),
    "kalman-linear": ModelOptions(degree=1, smoothing="kalman"),
    "fit-sound-speed": ModelOptions(degree=1, fit_sound_speed=True),
    "fit-wind": ModelOptions(degree=1, fit_wind=True),
    "fit-altitude": ModelOptions(degree=1, fit_altitude=True),
    "fit-clock-rate": ModelOptions(degree=1, fit_clock_rate=True),
    "spline-joint": ModelOptions(
        degree=1,
        smoothing="spline",
        fit_sound_speed=True,
        fit_wind=True,
        fit_altitude=True,
        fit_clock_rate=True,
    ),
}
METHODS = (
    "brightness",
    "contrast",
    "legacy-ridge",
    "legacy-drift",
    *PROJECTED,
    "blocked-cv",
    "spline-cv",
    "event-consensus",
    "multistart-linear",
    "two-events",
    "three-events",
    "log-power",
)


def run_method(method, spec, events, tracks, candidate):
    """Inference entrypoint intentionally receives neither truth nor scenario labels."""
    if method in ("brightness", "contrast"):
        return refine_spectral(spec, events, tracks, candidate, method)
    if method in ("legacy-ridge", "legacy-drift"):
        return refine_candidate(spec, events, tracks, candidate, fit_drift=method == "legacy-drift")
    if method in PROJECTED:
        return fit_projected(spec, events, tracks, candidate, PROJECTED[method])
    if method in VELOCITY_PROJECTED:
        return fit_projected(spec, events, tracks, candidate, VELOCITY_PROJECTED[method])
    if method in ("blocked-cv", "spline-cv"):
        options = [
            ModelOptions(degree=d, smoothing="spline" if method == "spline-cv" else "none") for d in (0, 1, 2)
        ]
        return select_model(spec, events, tracks, candidate, options, folds=(0, 1, 2))
    if method == "event-consensus":
        return fit_consensus(spec, events, tracks, candidate)
    if method == "multistart-linear":
        attempts = []
        for shift in ([0, 0, 0], [250, 0, 0], [-250, 0, 0], [0, 250, 0], [0, -250, 0]):
            initial = {**candidate, "position": (np.array(candidate["position"]) + shift).tolist()}
            try:
                attempts.append(fit_projected(spec, events, tracks, initial, ModelOptions(degree=1)))
            except ValueError:
                continue
        if not attempts:
            raise ValueError("all deterministic starts failed")
        result = max(attempts, key=lambda r: r["score"])
        result["starts_attempted"] = 5
        result["starts_completed"] = len(attempts)
        return result
    if method in ("two-events", "three-events"):
        n = 2 if method == "two-events" else 3
        initial = {**candidate, "f0": candidate["f0"][:n], "flights": candidate["flights"][:n]}
        return fit_projected(spec, events[:n], tracks[:n], initial, ModelOptions(degree=1))
    if method == "log-power":
        compressed = Spectrogram(spec.times, spec.frequencies, np.log1p(spec.power))
        return fit_projected(compressed, events, tracks, candidate, ModelOptions(degree=1))
    raise ValueError(f"unknown method {method}")


def _run_scene(job):
    scenario, seed, methods = job
    scene = stress_scene(scenario, seed)
    rows = []
    for method in methods:
        start = time.perf_counter()
        row = {
            "scenario": scenario,
            "seed": seed,
            "method": method,
            "error_m": None,
            "success": False,
            "accepted": False,
            "exception": None,
            "at_bounds": False,
            "position": None,
            "signal_present": scenario != "noise_only",
        }
        try:
            result = run_method(
                method, scene["spectrogram"], scene["events"], scene["tracks"], scene["candidate"]
            )
            n = len(result["f0"])
            evidence = assess_candidate(
                scene["spectrogram"], scene["events"][:n], scene["tracks"][:n], result
            )
            row.update(
                position=result["position"],
                success=result["success"],
                accepted=evidence["accepted"],
                coverage=evidence["coverage"],
                frame_support=evidence["mean_frame_support"],
                at_bounds=result.get("at_bounds", False),
                clock_error_s=result["clock"] - scene["clock"],
                median_residual_hz=float(np.median(result.get("event_residual_hz", [])))
                if result.get("event_residual_hz")
                else None,
                model=result.get("options"),
                selected_degree=result.get("options", {}).get("degree"),
                excluded_event=result.get("excluded_event"),
                nfev=result.get("nfev"),
                source_frequency=result["f0"],
            )
            if scenario != "noise_only":
                row["error_m"] = float(
                    np.linalg.norm(np.asarray(result["position"])[:2] - scene["truth"][:2])
                )
        except Exception as exc:
            # The report preserves failed attempts. The test suite still raises bugs normally.
            row["exception"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
    return rows


def summarize_trials(rows):
    summary = {}
    for method in dict.fromkeys(row["method"] for row in rows):
        group = [r for r in rows if r["method"] == method]
        positive = [r for r in group if r["scenario"] != "noise_only"]
        negative = [r for r in group if r["scenario"] == "noise_only"]
        numeric = [r["error_m"] for r in positive if r["error_m"] is not None]
        summary[method] = {
            "attempts": len(group),
            "signal_trials": len(positive),
            "numeric_estimates": len(numeric),
            "median_error_m": float(np.median(numeric)) if numeric else None,
            "p90_error_m": float(np.quantile(numeric, 0.9)) if numeric else None,
            "worst_error_m": float(max(numeric)) if numeric else None,
            "within_100m_fraction": sum(r["error_m"] is not None and r["error_m"] <= 100 for r in positive)
            / max(1, len(positive)),
            "accepted_within_100m_fraction": sum(
                r["accepted"] and r["error_m"] is not None and r["error_m"] <= 100 for r in positive
            )
            / max(1, len(positive)),
            "rejected_signal_fraction": sum(not r["accepted"] for r in positive) / max(1, len(positive)),
            "exceptions": sum(r["exception"] is not None for r in group),
            "nonconverged": sum(not r["success"] for r in group),
            "negative_trials": len(negative),
            "false_accepts": sum(r["accepted"] for r in negative),
            "median_seconds": float(np.median([r["seconds"] for r in group])),
        }
    return summary


def run_stress_benchmark(
    directory,
    seeds=(100, 101, 102, 103, 104),
    scenarios=STRESS_SCENARIOS,
    methods=METHODS,
    split="held-out",
    workers=1,
    progress=None,
):
    if not seeds or not scenarios or not methods or workers < 1:
        raise ValueError("seeds, scenarios, methods and a positive worker count are required")
    if any(m not in METHODS + EXTENDED_METHODS for m in methods) or any(
        s not in STRESS_SCENARIOS + EXTENDED_SCENARIOS for s in scenarios
    ):
        raise ValueError("unknown benchmark method or scenario")
    if split not in ("development", "held-out"):
        raise ValueError("split must be development or held-out")
    if split == "held-out" and any(seed < 100 for seed in seeds):
        raise ValueError("held-out evaluation uses seeds >=100; development uses separate seeds")
    if (
        len(set(seeds)) != len(seeds)
        or len(set(methods)) != len(methods)
        or len(set(scenarios)) != len(scenarios)
    ):
        raise ValueError("duplicate trials would distort denominators")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
    }
    jobs = [(scenario, seed, list(methods)) for scenario in scenarios for seed in seeds]
    rows = []
    started = time.perf_counter()
    log = directory / "trials.jsonl"
    with log.open("w") as stream:
        if workers == 1:
            batches = map(_run_scene, jobs)
            pool = None
        else:
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            futures = [pool.submit(_run_scene, job) for job in jobs]
            batches = (future.result() for future in as_completed(futures))
        try:
            for batch in batches:
                rows.extend(batch)
                for row in batch:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if progress:
                    progress(len(rows), len(jobs) * len(methods), batch[0]["scenario"], batch[0]["seed"])
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
    rows.sort(
        key=lambda r: (
            list(scenarios).index(r["scenario"]),
            list(seeds).index(r["seed"]),
            list(methods).index(r["method"]),
        )
    )
    report = {
        "validation": "synthetic conditional localization; known flight assignments and seeded event windows",
        "split": split,
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "methods": list(methods),
        "method_options": {
            name: vars({**PROJECTED, **VELOCITY_PROJECTED}[name])
            for name in methods
            if name in PROJECTED or name in VELOCITY_PROJECTED
        },
        "source_hashes": source_hashes,
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "pillow")},
        },
        "seconds": time.perf_counter() - started,
        "workers": workers,
        "summary": summarize_trials(rows),
        "by_scenario": {
            scenario: summarize_trials([r for r in rows if r["scenario"] == scenario])
            for scenario in scenarios
        },
        "trials": rows,
        "limitations": [
            "Independent synthetic generator, not field validation.",
            "Initialization is a fixed region center, but flight identities and approximate event parameters are seeded.",
            "Numerical error summaries include every numeric estimate, even if nonconverged; exceptions remain in counts and success denominators.",
            "Evidence acceptance is heuristic, not a calibrated posterior or multiple-search probability.",
            "Methods are evaluated on the same scenes; selection based on this held-out table would require a new validation set.",
        ],
    }
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    fields = [
        "scenario",
        "seed",
        "method",
        "error_m",
        "accepted",
        "success",
        "exception",
        "seconds",
        "at_bounds",
    ]
    with (directory / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_report(report, directory / "report.html")
    return report


def write_report(report, path):
    data = json.dumps({"summary": report["summary"], "by_scenario": report["by_scenario"]}).replace(
        "<", "\\u003c"
    )
    options = "".join(f"<option>{html.escape(s)}</option>" for s in report["scenarios"])
    warnings = "".join(f"<li>{html.escape(s)}</li>" for s in report["limitations"])
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Dopplertracker extended accuracy experiments</title><style>
body{{font:15px system-ui;color:#183036;max-width:1400px;margin:32px auto;padding:0 24px}}h1{{font-size:30px}}
aside{{padding:12px 20px;background:#edf5f3}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:10px;text-align:right;border-bottom:1px solid #d7dfdf}}th:first-child,td:first-child{{text-align:left}}
thead{{position:sticky;top:0;background:white}}select,input{{padding:8px;margin:12px}}.scroll{{overflow:auto}}a{{color:#12646a}}
</style><h1>Extended accuracy experiments</h1>
<p><strong>{html.escape(report["split"])}</strong>: {len(report["methods"])} methods × {len(report["scenarios"])} scenarios × {len(report["seeds"])} seeds = {len(report["trials"])} attempted fits.</p>
<aside><strong>Synthetic, known-flight localization. Real-world accuracy is unverified.</strong><ul>{warnings}</ul></aside>
<label>Scenario<select id="scenario"><option value="all">All scenarios</option>{options}</select></label>
<label>Filter methods<input id="filter" placeholder="e.g. spline"></label>
<p>Errors: meters, lower is better. Error quantiles cover numeric estimates; failures remain in attempt counts and ≤100 m denominators. Noise-only trials have no location error.</p>
<div class="scroll"><table><thead><tr><th>Method</th><th>Median error</th><th>90th percentile</th><th>≤100 m</th><th>Accepted & ≤100 m</th><th>Rejected signal</th><th>Exceptions</th><th>Not converged</th><th>Noise accepts</th><th>Median seconds</th></tr></thead><tbody id="rows"></tbody></table></div>
<p><a href="results.json">All results and provenance</a> · <a href="trials.csv">Every trial as CSV</a></p>
<script>const data={data}; const fmt=x=>x===null?'—':x.toFixed(1); const pct=x=>(x*100).toFixed(0)+'%';
function draw(){{let s=document.getElementById('scenario').value;let f=document.getElementById('filter').value.toLowerCase();
let obj=s==='all'?data.summary:data.by_scenario[s]; let rows=Object.entries(obj).filter(([m])=>m.includes(f));
rows.sort((a,b)=>(a[1].median_error_m??Infinity)-(b[1].median_error_m??Infinity));
document.getElementById('rows').innerHTML=rows.map(([m,r])=>`<tr><td>${{m}}</td><td>${{fmt(r.median_error_m)}}</td><td>${{fmt(r.p90_error_m)}}</td><td>${{pct(r.within_100m_fraction)}}</td><td>${{pct(r.accepted_within_100m_fraction)}}</td><td>${{pct(r.rejected_signal_fraction)}}</td><td>${{r.exceptions}}/${{r.attempts}}</td><td>${{r.nonconverged}}/${{r.attempts}}</td><td>${{r.false_accepts}}/${{r.negative_trials}}</td><td>${{r.median_seconds.toFixed(2)}}</td></tr>`).join('');}}
document.getElementById('scenario').onchange=draw;document.getElementById('filter').oninput=draw;draw();</script></html>"""
    Path(path).write_text(text)


# Reported-velocity variants (opt-in; not part of the default METHODS matrix).
# Without velocity reports the "reported" variants reduce exactly to their
# differenced-position counterparts (projected-linear, spline-linear).
VELOCITY_PROJECTED = {
    "hermite-velocity-linear": ModelOptions(degree=1, velocity="reported"),
    "spline-velocity-linear": ModelOptions(degree=1, smoothing="spline", velocity="reported"),
    "kalman-velocity-linear": ModelOptions(degree=1, velocity="fused"),
    "spline-offsets-linear": ModelOptions(degree=1, smoothing="spline", fit_track_offsets=True),
    "kalman-velocity-offsets-linear": ModelOptions(degree=1, velocity="fused", fit_track_offsets=True),
}
EXTENDED_METHODS = tuple(VELOCITY_PROJECTED)


# ----- Extended layered-atmosphere matrix (not part of METHODS / STRESS_SCENARIOS) ---------------

_SPLINE = ModelOptions(degree=1, smoothing="spline")
_SPLINE_C = ModelOptions(degree=1, smoothing="spline", fit_sound_speed=True)
_SPLINE_C_WIND = ModelOptions(degree=1, smoothing="spline", fit_sound_speed=True, fit_wind=True)
# name -> (options, scene atmosphere key or None for the homogeneous 343 m/s default).
# "atmosphere_profile" is the generator's true profile (an oracle);
# "atmosphere_forecast" is biased +1.5 K with a 3 m/s wind error, like a reanalysis.
LAYERED_METHODS = {
    "spline-linear": (_SPLINE, None),
    "spline-linear-fit-c": (_SPLINE_C, None),
    "spline-linear-true-profile": (_SPLINE, "atmosphere_profile"),
    "spline-linear-forecast-profile": (_SPLINE, "atmosphere_forecast"),
    "spline-linear-forecast-fit-c": (_SPLINE_C, "atmosphere_forecast"),
    "spline-linear-forecast-fit-c-wind": (_SPLINE_C_WIND, "atmosphere_forecast"),
}
# Atmosphere and reported-velocity models together (for ``realistic_combined``).
# Not in the default LAYERED_METHODS run; run_layered_benchmark accepts both.
_FUSED = ModelOptions(degree=1, velocity="fused")
COMBINED_METHODS = {
    "kalman-velocity-linear": (_FUSED, None),
    "kalman-velocity-forecast-profile": (_FUSED, "atmosphere_forecast"),
    "kalman-velocity-forecast-fit-c-wind": (
        ModelOptions(degree=1, velocity="fused", fit_sound_speed=True, fit_wind=True),
        "atmosphere_forecast",
    ),
}
ATMOSPHERE_METHODS = {**LAYERED_METHODS, **COMBINED_METHODS}


def _run_layered_scene(job):
    scenario, seed, methods = job
    scene = stress_scene(scenario, seed)
    rows = []
    for method in methods:
        options, key = ATMOSPHERE_METHODS[method]
        start = time.perf_counter()
        row = {
            "scenario": scenario,
            "seed": seed,
            "method": method,
            "error_m": None,
            "success": False,
            "accepted": False,
            "exception": None,
            "at_bounds": False,
            "position": None,
            "signal_present": True,
        }
        try:
            extra = {} if key is None else {"atmosphere": scene[key]}
            spec, events, tracks = scene["spectrogram"], scene["events"], scene["tracks"]
            result = fit_projected(spec, events, tracks, scene["candidate"], options, **extra)
            evidence = assess_candidate(spec, events, tracks, result)
            row.update(
                position=result["position"],
                success=result["success"],
                accepted=evidence["accepted"],
                at_bounds=result["at_bounds"],
                clock_error_s=result["clock"] - scene["clock"],
                median_residual_hz=float(np.median(result["event_residual_hz"])),
                sound_speed_m_s=result["sound_speed_m_s"],
                sound_speed_offset_m_s=result.get("atmosphere", {}).get("sound_speed_offset_m_s"),
                nfev=result["nfev"],
                error_m=float(np.linalg.norm(np.asarray(result["position"])[:2] - scene["truth"][:2])),
            )
        except Exception as exc:
            row["exception"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
    return rows


def run_layered_benchmark(
    directory,
    seeds=(0, 1, 2, 3, 4),
    scenarios=LAYERED_SCENARIOS,
    methods=tuple(LAYERED_METHODS),
    split="development",
    workers=1,
    progress=None,
):
    """Spline-linear with a constant, fitted, true or forecast-like layered atmosphere."""
    if not seeds or not scenarios or not methods or workers < 1:
        raise ValueError("seeds, scenarios, methods and a positive worker count are required")
    if any(m not in ATMOSPHERE_METHODS for m in methods) or any(
        s not in EXTENDED_SCENARIOS for s in scenarios
    ):
        raise ValueError("unknown layered benchmark method or scenario")
    if split not in ("development", "held-out"):
        raise ValueError("split must be development or held-out")
    if any((seed >= 100) != (split == "held-out") for seed in seeds):
        raise ValueError("development uses seeds <100; held-out evaluation uses seeds >=100")
    seeds, scenarios, methods = list(seeds), list(scenarios), list(methods)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [(scenario, seed, list(methods)) for scenario in scenarios for seed in seeds]
    started = time.perf_counter()
    rows = []
    if workers == 1:
        batches = map(_run_layered_scene, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        batches = (f.result() for f in as_completed([pool.submit(_run_layered_scene, job) for job in jobs]))
    try:
        for batch in batches:
            rows.extend(batch)
            if progress:
                progress(len(rows), len(jobs) * len(methods), batch[0]["scenario"], batch[0]["seed"])
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    rows.sort(
        key=lambda r: (scenarios.index(r["scenario"]), seeds.index(r["seed"]), methods.index(r["method"]))
    )
    report = {
        "validation": "synthetic layered-atmosphere localization; arrival times from an independent ray tracer",
        "split": split,
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "methods": list(methods),
        "method_options": {name: vars(ATMOSPHERE_METHODS[name][0]) for name in methods},
        "method_atmosphere": {name: ATMOSPHERE_METHODS[name][1] or "homogeneous 343 m/s" for name in methods},
        "seconds": time.perf_counter() - started,
        "workers": workers,
        "summary": summarize_trials(rows),
        "by_scenario": {s: summarize_trials([r for r in rows if r["scenario"] == s]) for s in scenarios},
        "trials": rows,
        "limitations": [
            "Independent synthetic generator, not field validation.",
            "spline-linear-true-profile is an oracle: it is given the generator's exact profile.",
            "The forecast profile has one fixed error model (+1.5 K, constant 3 m/s wind vector).",
        ],
    }
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    write_report(report, directory / "report.html")
    return report
