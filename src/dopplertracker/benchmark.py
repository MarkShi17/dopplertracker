"""Reproducible ablations with an independent analytic signal generator."""

import html
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import save_tracks
from .detection import Event, save_events
from .geo import LocalFrame
from .physics import Trajectory
from .refine import refine_candidate, refine_spectral
from .scoring import joint_score
from .signal import Spectrogram

SCENARIOS = ("clean", "interference", "rpm_drift", "overhead", "sound_speed_error")


def analytic_pass(receive, velocity, origin, center, observer, f0, c, drift=0.0):
    """Independent closed-form generator, not physics.predict or its interpolator."""
    velocity = np.array(velocity, dtype=float)
    relative = np.array(origin) - observer
    speed2 = np.dot(velocity, velocity)
    t_cpa = center - np.dot(relative, velocity) / speed2
    closest = relative + (t_cpa - center) * velocity
    d2 = np.dot(closest, closest)
    r = receive - t_cpa
    a = c * c - speed2
    emission_relative = (c * c * r - np.sqrt(c * c * speed2 * r * r + a * d2)) / a
    emission = t_cpa + emission_relative
    radial = speed2 * emission_relative / np.sqrt(d2 + speed2 * emission_relative**2)
    frequency = (f0 + drift * emission_relative) / (1 + radial / c)
    return frequency, emission, t_cpa + np.sqrt(d2) / c, np.sqrt(d2)


def synthetic_scene(scenario="clean", seed=0):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}")
    rng = np.random.default_rng(seed)
    truth = np.array([135.0, -85.0, 0.0]) + np.r_[rng.uniform(-35, 35, 2), 0.0]
    c = 335.0 if scenario == "sound_speed_error" else 343.0
    times = np.arange(0.0, 640.0, 1.0)
    frequencies = np.arange(50.0, 360.0, 0.25)
    power = rng.exponential(0.15, (len(frequencies), len(times)))
    tracks, events = [], []
    geometries = [
        ([82.0, 0.0, 0.0], [0.0, 1700.0, 1000.0], 100.0, 111.0),
        ([0.0, 94.0, 0.0], [-1900.0, 0.0, 1300.0], 310.0, 127.0),
        ([67.0, -55.0, 0.0], [1100.0, 1100.0, 850.0], 520.0, 103.0),
    ]
    for i, (velocity, origin, center, f0) in enumerate(geometries):
        if scenario == "overhead":
            origin = [truth[0] + 100 * (i - 1), truth[1] + 100 * (i - 1), 4500.0 + i * 1500]
        drift = [0.013, -0.009, 0.017][i] if scenario == "rpm_drift" else 0.0
        frequency, emission, cpa, distance = analytic_pass(
            times, velocity, origin, center, truth, f0, c, drift
        )
        track_times = np.arange(center - 130, center + 131, 2.0)
        positions = np.array(origin) + (track_times - center)[:, None] * velocity
        tracks.append(Trajectory(f"flight-{i}", track_times, positions))
        # Small source jitter is not represented by the inference model.
        frequency += rng.normal(0, 0.055, len(times))
        visible = (emission >= track_times[0]) & (emission <= track_times[-1])
        for harmonic in (1, 2):
            delta = frequencies[:, None] - harmonic * frequency
            power += (7.0 / harmonic) * np.exp(-0.5 * (delta / 0.4) ** 2) * visible
        if scenario == "interference":
            # Broadband clutter and sparse strong competing tones.
            busy = rng.random(len(times)) < 0.18
            power[:, busy] += rng.uniform(2, 7, busy.sum())
            corrupted = visible & (rng.random(len(times)) < 0.2)
            wrong = frequency + rng.choice([-2.5, 2.5], len(times))
            power += 14 * np.exp(-0.5 * ((frequencies[:, None] - wrong) / 0.4) ** 2) * corrupted
        events.append(
            Event(
                (cpa - 60, cpa + 60),
                cpa + rng.normal(0, 0.25),
                f0 + rng.normal(0, 0.15),
                float(np.linalg.norm(velocity)),
                distance,
                2,
            )
        )
    return {
        "truth": truth,
        "clock": 0.0,
        "tracks": tracks,
        "events": events,
        "spectrogram": Spectrogram(times, frequencies, power.astype(np.float32)),
        "scenario": scenario,
        "seed": seed,
        "sound_speed": c,
    }


def write_demo(directory, scenario="rpm_drift", seed=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scene = synthetic_scene(scenario, seed)
    scene["spectrogram"].save(directory / "spectrogram.npz")
    save_events(scene["events"], directory / "events.json")
    frame = LocalFrame(40.0, -73.0)
    save_tracks(directory / "traces.npz", scene["tracks"], frame)
    truth = {
        "position": scene["truth"].tolist(),
        "latitude_longitude_height": frame.to_geodetic(scene["truth"]).tolist(),
        "recording_start": scene["clock"],
        "scenario": scenario,
        "seed": seed,
        "synthetic": True,
    }
    (directory / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def run_benchmark(directory, seeds=(0, 1, 2), scenarios=SCENARIOS, grid_extent=600, progress=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    axis = np.arange(-grid_extent, grid_extent + 1, 200.0)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    for scenario in scenarios:
        for seed in seeds:
            scene = synthetic_scene(scenario, seed)
            spec, events, tracks = scene["spectrogram"], scene["events"], scene["tracks"]
            best_brightness = None
            first_row = len(rows)
            for method in ("brightness", "contrast"):
                start = time.perf_counter()
                candidates = []
                for point in points:
                    score, clock, f0, individual = joint_score(
                        spec,
                        events,
                        tracks,
                        point,
                        0.0,
                        method,
                        shift_radius=1,
                        shift_steps=3,
                        frequency_radius=1,
                        frequency_steps=5,
                    )
                    candidates.append(
                        {
                            "position": point.tolist(),
                            "clock": clock,
                            "f0": f0,
                            "score": score,
                            "flights": [t.id for t in tracks],
                            "event_scores": individual,
                        }
                    )
                best = max(candidates, key=lambda r: r["score"])
                if method == "brightness":
                    best_brightness = best
                rows.append(_row(scene, method + "-grid", best, start))
                start = time.perf_counter()
                continuous = refine_spectral(spec, events, tracks, best, method)
                rows.append(_row(scene, method + "-continuous", continuous, start))
            for name, drift, sound_speed in [
                ("robust-ridge", False, False),
                ("robust-ridge-drift", True, False),
                ("robust-ridge-atmosphere", True, True),
            ]:
                start = time.perf_counter()
                refined = refine_candidate(
                    spec,
                    events,
                    tracks,
                    best_brightness,
                    radius=1000.0,
                    fit_drift=drift,
                    fit_sound_speed=sound_speed,
                )
                rows.append(_row(scene, name, refined, start))
            if progress:
                progress(scenario, seed, rows[first_row:])
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row["error_m"])
    summary = {
        method: {
            "median_error_m": float(np.median(errors)),
            "p90_error_m": float(np.quantile(errors, 0.9)),
            "max_error_m": float(np.max(errors)),
            "trials": len(errors),
        }
        for method, errors in groups.items()
    }
    report = {
        "validation": "synthetic; known flight assignments",
        "generator": "independent analytic straight-pass solution",
        "grid_spacing_m": 200.0,
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "limitations": "Tests conditional localization, not countrywide flight identification. Events are seeded from the generator. "
        "No real recordings, stratified weather, multipath or ADS-B position error in this benchmark.",
        "summary": summary,
        "trials": rows,
    }
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    _html_report(report, directory / "report.html")
    return report


def _row(scene, method, result, start):
    return {
        "scenario": scene["scenario"],
        "seed": scene["seed"],
        "method": method,
        "error_m": float(np.linalg.norm(np.array(result["position"])[:2] - scene["truth"][:2])),
        "position": result["position"],
        "truth": scene["truth"].tolist(),
        "clock_error_s": float(result["clock"] - scene["clock"]),
        "score": result["score"],
        "seconds": time.perf_counter() - start,
        "success": result.get("success", True),
        "at_bounds": result.get("at_bounds", False),
        "geometry_condition": result.get("geometry_condition"),
    }


def _html_report(report, path):
    summaries = "".join(
        f"<tr><td>{html.escape(m)}</td><td>{s['median_error_m']:.1f}</td>"
        f"<td>{s['p90_error_m']:.1f}</td><td>{s['max_error_m']:.1f}</td></tr>"
        for m, s in report["summary"].items()
    )
    rows = "".join(
        f"<tr><td>{r['scenario']}</td><td>{r['seed']}</td><td>{r['method']}</td>"
        f"<td>{r['error_m']:.1f}</td><td>{r['clock_error_s']:.3f}</td><td>{r['success']}</td></tr>"
        for r in report["trials"]
    )
    max_error = max(s["median_error_m"] for s in report["summary"].values()) or 1
    bars = "".join(
        f'<text x="0" y="{i * 38 + 21}" font-size="13">{m}</text>'
        f'<rect x="210" y="{i * 38 + 6}" width="{s["median_error_m"] / max_error * 360:.1f}" '
        f'height="22" fill="#247d80"/><text x="580" y="{i * 38 + 22}">{s["median_error_m"]:.1f} m</text>'
        for i, (m, s) in enumerate(report["summary"].items())
    )
    path.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Dopplertracker method comparison</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#173238}}
table{{border-collapse:collapse;width:100%;margin:24px 0}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #ccd8d9}}
aside{{background:#eef5f4;padding:16px}}svg{{max-width:100%;height:auto}}</style>
<h1>Dopplertracker: measured method comparison</h1><aside><strong>Synthetic benchmark — known flight assignments.</strong>
<p>{html.escape(report["limitations"])}</p></aside><h2>Median location error (lower is better)</h2>
<svg role="img" aria-label="Median localization errors by method" viewBox="0 0 700 {len(report["summary"]) * 38 + 12}">{bars}</svg>
<table><thead><tr><th>Method</th><th>Median (m)</th><th>90th percentile (m)</th><th>Worst (m)</th></tr></thead><tbody>{summaries}</tbody></table>
<h2>Every trial</h2><table><thead><tr><th>Scenario</th><th>Seed</th><th>Method</th><th>Error (m)</th><th>Clock error (s)</th><th>Converged</th></tr></thead><tbody>{rows}</tbody></table>
<p>All methods use the same scene and ground grid. Refinement starts at the brightness winner, without using truth.
Ground truth is used only to calculate the errors above. A small ridge residual is not proof of a correct location.</p></html>""")
