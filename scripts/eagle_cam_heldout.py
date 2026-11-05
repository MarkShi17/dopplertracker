"""Held-out test on the SW Florida eagle cam recording (2024-12-28), run once.

No method or option was developed on this recording; nothing here is tuned on it.
Three stages, each a separate run, in this order:

  prepare   freeze the protocol (outputs/eagle-cam/plan.json, with source hashes),
            then the preparation that involves no fitting: start candidate and
            published events from the original project's tables, the two flights'
            tracks, the layered weather, and the automatic detector on the section
            with its event assignment (preparation.json).
  fit       every fit of the plan (fits.json, rewritten after each fit).  Atmosphere
            x trajectory x measurement factorial (eight cells, error regions) for
            the published and the automatic event set, then flyover discovery at
            the primary (layered / fused / ridge) solution with default and 12 km
            range limits and a joint refit with whatever is confirmed.
  evaluate  only once fits.json is complete: read the truth, compute errors,
            region coverage and the pre-registered contrasts (results.json).

The section clock: spectrogram time 0 is recording time 17,000 s.  The clock
fitted below is the UTC of spectrogram time 0.
"""

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_oyster_bay_velocity import load_flights, select_tracks, velocity_diagnostics  # noqa: E402

from dopplertracker.advanced import ModelOptions, fit_projected  # noqa: E402
from dopplertracker.atmosphere import WeatherSeries  # noqa: E402
from dopplertracker.data import subset_runs  # noqa: E402
from dopplertracker.demod import Audio  # noqa: E402
from dopplertracker.detection import Event, FlyoverOptions, detect_flyovers  # noqa: E402
from dopplertracker.discovery import DiscoveryOptions, RidgeMap, TraceIndex, discover, icao  # noqa: E402
from dopplertracker.evidence import assess_candidate  # noqa: E402
from dopplertracker.physics import Atmosphere, Trajectory, closest_approaches  # noqa: E402
from dopplertracker.scoring import joint_score  # noqa: E402
from dopplertracker.signal import Spectrogram, decode_samples  # noqa: E402
from dopplertracker.terrain import ConstantElevation  # noqa: E402
from dopplertracker.uncertainty import evaluate  # noqa: E402

DATA = Path("data/eagle-cam")
OUT = Path("outputs/eagle-cam")
SECTION_OFFSET_S = 17000.0  # inputs.json spectrogram_audio_offset_s (checked at run time)
GEOID_M = -24.231  # data/eagle-cam/geoid.json, GEOID18 at 26.7, -81.85 (checked at run time)
OBSERVER_ABOVE_MSL_M = 5.0  # ground near sea level; nest height unknown; fixed in advance
HEIGHT_M = GEOID_M + OBSERVER_ABOVE_MSL_M
CONSTANT_C = 343.0
MATCH_SECONDS = 20.0
DETECTOR_TOP = 15  # as scripts/compare_oyster_bay_autodetect.py and compare_oyster_bay_three.py
PUBLISHED_HARMONICS = 2  # as data/oyster-bay/events.json and the detector's events
# The original project's pair filters (dopplertrack pipeline.run_search defaults).
PAIR_TOL_S, V_TOL, D_TOL = 6.0, 0.35, (0.4, 2.5)
EPOCH_TOL_S = 120.0  # implied recording start (t_heard1 - event 1 t_cpa) versus the stream's release time
TRAJECTORIES = {
    "spline": ModelOptions(degree=1, smoothing="spline"),
    "fused": ModelOptions(degree=1, velocity="fused"),
}
MEASUREMENTS = {
    "ridge": {},
    "demod": dict(ridge="demod", reextract=2, noise="snr", hum="subtract"),
}
BASELINE, PRIMARY = "constant/spline/ridge", "layered/fused/ridge"
DISCOVERY = {
    "default": DiscoveryOptions(),
    "relaxed-12km": DiscoveryOptions(max_distance=12000.0, ground_range=12000.0),
}
SETS = ("published", "automatic")


def cells():
    out = []
    for atmosphere in ("constant", "layered"):
        for trajectory, base in TRAJECTORIES.items():
            for measurement, changes in MEASUREMENTS.items():
                out.append((f"{atmosphere}/{trajectory}/{measurement}", atmosphere, replace(base, **changes)))
    return out


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dump(path, value):
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


PROTOCOL = {
    "validation": "Held-out real recording. Conditional refinement from the original project's top grid "
    "candidate and flight pair; not a blind search. Nothing is tuned on this recording. Truth "
    "(published/provenance.json evaluation_truth) is read only by the evaluate stage, after fits.json "
    "is complete.",
    "P1": "Reported velocities re-extracted from the raw archive into data/eagle-cam/traces-velocity.npz "
    "(scripts/extract_eagle_cam_velocity.py; inputs.json frame and bbox, archive SHA-256 checked).",
    "P2": "Layered weather: WeatherSeries.from_open_meteo(weather-open-meteo.json, geoid -24.231 m), saved as "
    "data/eagle-cam/weather-profile-geoid18.json; per-flight profiles at each event's heard CPA "
    "(WeatherSeries.for_events at the set's initial clock).",
    "P3": f"Observer ellipsoid height for every fit {HEIGHT_M:.3f} m (= {GEOID_M} m geoid + "
    f"{OBSERVER_ABOVE_MSL_M} m above MSL).",
    "P4": "Start candidate: highest-score row of published/swfl_search_best_per_pair.csv (lat, lon, icao1, "
    "icao2, t_heard1, shift); dist_m is not read before evaluation.",
    "P5": "Published events: the ordered pair (i, j) of published/swfl_detected.json entries that satisfies "
    "the original search's pair filters for the winning row: v_k within +/-35 % of the row's v_k, row d_k "
    "within [0.4, 2.5] x event d_k, and (t_cpa_j - t_cpa_i) within 6 s of the row's heard-time difference "
    "(t_heard2 - t_heard1, t_heard2 = icao2's acoustic CPA at the row position, 343 m/s, P3 height), "
    "and the implied recording start t_heard1 - t_cpa_i within 120 s of the live stream's release_timestamp "
    "(recording.info.json). Expected unique; if several, the smallest timing mismatch. Section clock = recording - 17000 s. "
    "Event window = [t_start, t_end], cpa = t_cpa, f0, speed = v, distance = d, harmonics 2.",
    "P6": "Automatic events: detect_flyovers(whitened section, FlyoverOptions(), top=15). For each flight, its "
    "predicted heard CPA at the start candidate (the acoustic CPA nearest the flight's published event, "
    "343 m/s, published-set initial clock); the detection nearest in heard time within 20 s. (The Oyster "
    "Bay scripts pick the highest z within 20 s; the brief says nearest; nearest is used and the "
    "highest-z choice is recorded.) A flight without a detection within 20 s is recorded and the "
    "automatic set has only what exists (one event: initial clock of the published set; none: no "
    "automatic fits).",
    "start_and_calibration": "As scripts/compare_oyster_bay.py: initial clock = t_heard1 - event1.cpa + "
    "shift; tracks = the covering segment of each ICAO trimmed to event window +/-120 s (velocity file); "
    "joint_score sweep at the fixed start (shared +/-4 s shift, 17 steps; per-event +/-12 Hz, 121 steps), "
    "once per atmosphere and event set.",
    "F1": f"BASELINE {BASELINE}: ModelOptions(degree=1, smoothing='spline'), constant 343 m/s.",
    "F2": f"PRIMARY {PRIMARY}: ModelOptions(degree=1, velocity='fused'), layered atmosphere, spectrogram "
    "ridge, uncertainty=True.",
    "F3": "Full factorial atmosphere {constant 343 m/s, layered} x trajectory {spline, fused} x measurement "
    "{ridge, demod (reextract=2, noise='snr', hum='subtract')}; every cell uncertainty=True, "
    "keep_observations, evidence via assess_candidate.",
    "F4": "At each set's F2 solution: discover() over traces-velocity.npz (TraceIndex) with DiscoveryOptions() "
    "and, separately, max_distance = ground_range = 12 km; discovery scan atmosphere constant 343 m/s (as "
    "both Oyster Bay discovery scripts). Joint refit (F2 options, layered profiles for every event, "
    "uncertainty=True) of the F2 events + confirmed independent passes, start = F2 position/clock, f0 = F2 "
    "f0 + discovered f0; the discovered segment is taken from the velocity file and trimmed to its window "
    "+/-240 s (as discovery.trim_track). If the refit raises ValueError the weakest discovery is dropped "
    "and the refit retried (as discovery.joint_refit). No confirmations: no refit (recorded).",
    "E1": "Horizontal error of every fit; inside 50 % / 90 % region; normalised error sqrt(e^T C^-1 e).",
    "E2": "F1 vs F2 per event set; for each factor the paired cells (other factors fixed) it improved.",
    "E3": "Only then: the original project's top grid candidate, two-flyover and five-flyover refinements.",
    "failures": "A failing fit is an outcome and is reported; bugs are fixed and the whole plan rerun.",
}


def published_row():
    with (DATA / "published/swfl_search_best_per_pair.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    best = max(rows, key=lambda r: float(r["score"]))
    return {k: v for k, v in best.items() if k != "dist_m"}, len(rows)


def heard_cpas(track, observer, atmosphere=Atmosphere(CONSTANT_C)):
    return [(float(h), float(d), float(s)) for h, d, s in closest_approaches(track, observer, atmosphere)]


def emission_state(track, observer, heard, distance, c=CONSTANT_C):
    emission = heard - distance / c
    position, _ = track.at(np.clip(emission, track.times[0], track.times[-1]))
    return {
        "altitude_m": float(position[2]),
        "height_above_observer_m": float(position[2] - observer[2]),
        "horizontal_m": float(np.hypot(*(position[:2] - observer[:2]))),
    }


def to_section(entry, harmonics=PUBLISHED_HARMONICS):
    return Event(
        (float(entry["t_start"]) - SECTION_OFFSET_S, float(entry["t_end"]) - SECTION_OFFSET_S),
        float(entry["t_cpa"]) - SECTION_OFFSET_S,
        float(entry["f0"]),
        float(entry["v"]),
        float(entry["d"]),
        harmonics,
        float(entry["score"]),
    )


def prepare(args):
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "plan.json"
    if plan_path.exists() and not args.force:
        raise SystemExit(f"{plan_path} exists (use --force to rewrite before any fit)")
    if (OUT / "fits.json").exists():
        raise SystemExit("fits.json exists; the plan must not change after fitting started")
    inputs = json.loads((DATA / "inputs.json").read_text())
    geoid = json.loads((DATA / "geoid.json").read_text())
    if inputs["spectrogram_audio_offset_s"] != SECTION_OFFSET_S or geoid["geoidHeight"] != GEOID_M:
        raise SystemExit("section offset or geoid differs from the frozen constants")
    notes = json.loads(Path(args.notes).read_text()) if args.notes else {}
    plan = {
        "protocol": PROTOCOL,
        "cells": {name: repr(options) for name, _, options in cells()},
        "discovery_options": {k: asdict(v) for k, v in DISCOVERY.items()},
        "detector_options": asdict(FlyoverOptions()),
        "observer_height_m": HEIGHT_M,
        "deviations": notes.get("deviations", []),
        "notes": notes.get("notes", []),
        "source_hashes": {p.name: sha256(p) for p in sorted(Path("src/dopplertracker").glob("*.py"))},
        "script_hashes": {
            p.name: sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).parent / "compare_oyster_bay_velocity.py",
                Path(__file__).parent / "extract_eagle_cam_velocity.py",
            )
        },
        "input_hashes": {
            str(p): sha256(p)
            for p in (
                DATA / "traces-velocity.npz",
                DATA / "traces.npz",
                DATA / "spectrogram-17000-24000.npz",
                DATA / "recording-17000-24000.webm",
                DATA / "weather-open-meteo.json",
                DATA / "published/swfl_search_best_per_pair.csv",
                DATA / "published/swfl_detected.json",
            )
        },
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    dump(plan_path, plan)
    print("plan written", flush=True)

    # P2: weather.
    raw = json.loads((DATA / "weather-open-meteo.json").read_text())
    weather = WeatherSeries.from_open_meteo(raw, GEOID_M)
    weather_path = DATA / "weather-profile-geoid18.json"
    if weather_path.exists():
        if WeatherSeries.load(weather_path).to_record() != weather.to_record():
            raise SystemExit(f"{weather_path} exists with different content")
    else:
        weather.save(weather_path)

    # P4: start candidate.
    row, rows = published_row()
    flights = (row["icao1"], row["icao2"])
    tracks, frame = load_flights(DATA / "traces-velocity.npz", set(flights))
    if (frame.lat, frame.lon) != tuple(inputs["frame"]):
        raise SystemExit("velocity file frame differs from inputs.json")
    start_latlon = [float(row["lat"]), float(row["lon"])]
    observer = frame.to_local([*start_latlon, HEIGHT_M])
    t_heard1 = float(row["t_heard1"])

    # P5: published events matching the winning row.
    detected = json.loads((DATA / "published/swfl_detected.json").read_text())
    release = float(json.loads((DATA / "recording.info.json").read_text())["release_timestamp"])
    alive = lambda t, when: t.times[0] - 300 <= when <= t.times[-1] + 300  # noqa: E731
    second = [t for t in tracks if icao(t.id) == flights[1]]
    first = [t for t in tracks if icao(t.id) == flights[0]]
    hits1 = [h for t in first for h in heard_cpas(t, observer) if alive(t, h[0])]
    near1 = min(hits1, key=lambda h: abs(h[0] - t_heard1))
    hits2 = [(t, h) for t in second for h in heard_cpas(t, observer)]
    pairs = []
    for i, a in enumerate(detected):
        if not (abs(float(row["v1"]) - a["v"]) <= V_TOL * a["v"]):
            continue
        if not (D_TOL[0] * a["d"] <= float(row["d1"]) <= D_TOL[1] * a["d"]):
            continue
        for j, b in enumerate(detected):
            if j == i or not abs(float(row["v2"]) - b["v"]) <= V_TOL * b["v"]:
                continue
            if not D_TOL[0] * b["d"] <= float(row["d2"]) <= D_TOL[1] * b["d"]:
                continue
            for track, (heard, distance, speed) in hits2:
                mismatch = (b["t_cpa"] - a["t_cpa"]) - (heard - t_heard1)
                epoch_offset = t_heard1 - a["t_cpa"] - release
                if abs(mismatch) <= PAIR_TOL_S and abs(epoch_offset) <= EPOCH_TOL_S:
                    pairs.append((abs(mismatch), i, j, mismatch, heard, distance, speed, epoch_offset))
    if not pairs:
        raise SystemExit("no published event pair matches the winning row")
    pairs.sort()
    _, i1, i2, mismatch, t_heard2, d2, v2, _ = pairs[0]
    published_events = [to_section(detected[i1]), to_section(detected[i2])]
    length = float(inputs["spectrogram_audio_range_s"][1] - inputs["spectrogram_audio_range_s"][0])
    in_section = all(0 <= e.window[0] and e.window[1] <= length for e in published_events)
    clock_published = t_heard1 - published_events[0].cpa + float(row["shift"])
    chosen = select_tracks(tracks, row, published_events, clock_published)

    # Flight description (no truth): altitudes and report cadence.
    described = []
    for k, (track, event) in enumerate(zip(chosen, published_events, strict=True)):
        hits = heard_cpas(track, observer)
        heard, distance, speed = min(hits, key=lambda h: abs(h[0] - clock_published - event.cpa))
        window = (clock_published + event.window[0], clock_published + event.window[1])
        try:
            cadence = velocity_diagnostics(track, window)
        except Exception as exc:  # recorded, not fatal
            cadence = {"error": f"{type(exc).__name__}: {exc}"}
        positions_file, _ = load_flights(DATA / "traces.npz", {icao(track.id)})
        original = next((t for t in positions_file if t.id == track.id), None)
        same = None
        if original is not None:
            mask = np.isin(original.times, track.times)
            same = bool(
                mask.sum() == len(track.times) and np.allclose(original.positions[mask], track.positions)
            )
        described.append(
            {
                "flight": track.id,
                "type": track.metadata.get("type"),
                "registration": track.metadata.get("registration"),
                "samples_in_slice": len(track.times),
                "slice_utc": [float(track.times[0]), float(track.times[-1])],
                "has_velocities": track.velocities is not None,
                "velocity_reports_finite": None
                if track.velocities is None
                else int(np.isfinite(track.velocities).all(axis=1).sum()),
                "heard_s_at_start": heard - clock_published,
                "slant_m_at_start": distance,
                "speed_m_s": speed,
                **emission_state(track, observer, heard, distance),
                "altitude_sources": track.metadata.get("altitude_sources"),
                "cadence": cadence,
                "positions_identical_to_traces_npz": same,
            }
        )
        del positions_file
        print("flight", k + 1, json.dumps(described[-1], default=str)[:400], flush=True)

    # P6: automatic detector on the section.
    spec = Spectrogram.load(DATA / "spectrogram-17000-24000.npz").whiten()
    report = {}
    began = time.perf_counter()
    detections = detect_flyovers(spec, FlyoverOptions(), top=DETECTOR_TOP, report=report)
    detector_seconds = time.perf_counter() - began
    print(f"{len(detections)} detections in {detector_seconds:.1f} s", flush=True)
    automatic, matches = {}, {}
    for flight, track, event in zip(flights, chosen, published_events, strict=True):
        full = [t for t in tracks if t.id == track.id]
        hits = [h for t in full for h in heard_cpas(t, observer)]
        heard = min((h[0] - clock_published for h in hits), key=lambda s: abs(s - event.cpa))
        nearby = [(rank, e) for rank, e in enumerate(detections, 1) if abs(e.cpa - heard) <= MATCH_SECONDS]
        entry = {"predicted_heard_s": heard, "published_cpa_s": event.cpa, "candidates": []}
        for rank, e in nearby:
            entry["candidates"].append({"rank": rank, "cpa": e.cpa, "z": e.z, "f0": e.f0})
        if nearby:
            rank, chosen_event = min(nearby, key=lambda item: abs(item[1].cpa - heard))
            entry["chosen_rank"] = rank
            entry["highest_z_rank"] = max(nearby, key=lambda item: item[1].z)[0]
            entry["cpa_minus_predicted_s"] = chosen_event.cpa - heard
            automatic[flight] = chosen_event
        else:
            entry["chosen_rank"] = None
        matches[flight] = entry
        print(flight, json.dumps(entry), flush=True)
    preparation = {
        "start": {
            "row": row,
            "rows_in_table": rows,
            "latlon": start_latlon,
            "position_local": observer.tolist(),
            "flights": list(flights),
            "t_heard1": t_heard1,
            "icao1_heard_at_start_utc": near1[0],
            "icao2_heard_at_start_utc": t_heard2,
            "icao2_slant_m": d2,
            "icao2_speed_m_s": v2,
        },
        "published_event_pairs": [
            {"mismatch_s": p[3], "event1_index": p[1], "event2_index": p[2], "epoch_minus_release_s": p[7]}
            for p in pairs
        ],
        "published_event_indices": [i1, i2],
        "published_event_records": [detected[i1], detected[i2]],
        "published_events_section": [asdict(e) for e in published_events],
        "published_windows_inside_section": bool(in_section),
        "recording_epoch_implied_utc": t_heard1 - float(detected[i1]["t_cpa"]),
        "initial_clock_published": clock_published,
        "flights_described": described,
        "velocity_file": json.loads((DATA / "traces-velocity.json").read_text()),
        "weather_file": str(weather_path),
        "weather_metadata": weather.metadata,
        "detector": {
            "options": asdict(FlyoverOptions()),
            "top": DETECTOR_TOP,
            "seconds": detector_seconds,
            "detections": [asdict(e) for e in detections],
            "candidates": report.get("candidates"),
            "matches": matches,
        },
        "automatic_events": {f: asdict(e) for f, e in automatic.items()},
    }
    dump(OUT / "preparation.json", preparation)
    print("preparation written", flush=True)


def events_for(preparation, set_name):
    """(flights, events, initial clock) of an event set; flights in published order."""
    start = preparation["start"]
    if set_name == "published":
        events = [Event(**e) for e in preparation["published_events_section"]]
        return list(start["flights"]), events, preparation["initial_clock_published"]
    found = preparation["automatic_events"]
    flights = [f for f in start["flights"] if f in found]
    events = [Event(**found[f]) for f in flights]
    if flights and flights[0] == start["flights"][0]:
        clock = start["t_heard1"] - events[0].cpa + float(start["row"]["shift"])
    else:
        clock = preparation["initial_clock_published"]
    return flights, events, clock


def velocity_segment(path, flight, start, end):
    """One segment of the velocity file with velocities, trimmed to UTC [start, end] (source runs sliced)."""
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
        index = next(i for i, item in enumerate(metadata) if item["id"] == flight)
        item = metadata[index]
        times, positions = z[f"t{index}"], z[f"p{index}"]
        velocities = z[f"v{index}"] if f"v{index}" in z.files else None
    mask = (times >= start) & (times <= end)
    if mask.sum() < 3:
        mask = np.ones(len(times), bool)
    item = dict(item)
    for key in ("position_source_runs", "vertical_rate_source_runs"):
        if key in item:
            item[key] = subset_runs(item[key], mask)
    return Trajectory(
        item["id"],
        times[mask],
        positions[mask],
        item.get("max_gap", 30.0),
        item,
        velocities=None if velocities is None else velocities[mask],
    )


def run_fit(spec, events, chosen, candidate, options, atmosphere, frame, audio):
    result = fit_projected(
        spec, events, chosen, candidate, options, atmosphere, frame=frame,
        elevation=ConstantElevation(HEIGHT_M), audio=audio, keep_observations=True, uncertainty=True,
    )  # fmt: skip
    result["evidence"] = assess_candidate(spec, events, chosen, result, atmosphere)
    result["lat_lon_height"] = frame.to_geodetic(result["position"]).tolist()
    result["valid_frames"] = [int(np.sum(o["valid"])) for o in result["observations"]]
    result["frames"] = [len(o["valid"]) for o in result["observations"]]
    del result["observations"]
    return result


def fit(args):
    plan = json.loads((OUT / "plan.json").read_text())
    preparation = json.loads((OUT / "preparation.json").read_text())
    current = {p.name: sha256(p) for p in sorted(Path("src/dopplertracker").glob("*.py"))}
    if current != plan["source_hashes"]:
        raise SystemExit("src/ changed since the plan was written; rerun prepare")
    spec = Spectrogram.load(DATA / "spectrogram-17000-24000.npz").whiten()
    audio = Audio(decode_samples(DATA / "recording-17000-24000.webm"), 2048.0)
    weather = WeatherSeries.load(DATA / "weather-profile-geoid18.json")
    start = preparation["start"]
    flights_all = set(start["flights"])
    tracks, frame = load_flights(DATA / "traces-velocity.npz", flights_all)
    observer = frame.to_local([*start["latlon"], HEIGHT_M])
    rows, discoveries = [], []
    output = {"plan_written_utc": plan["written_utc"], "status": "running", "trials": rows}

    def save():
        output["discoveries"] = discoveries
        dump(OUT / "fits.json", output)

    contexts = {}
    for set_name in SETS:
        flights, events, clock = events_for(preparation, set_name)
        if not events:
            rows.append({"set": set_name, "cell": None, "result": None, "error": "no automatic events"})
            save()
            continue
        coarse = {f"icao{i}": f for i, f in enumerate(flights, 1)}
        chosen = select_tracks(tracks, coarse, events, clock)
        atmospheres = {
            "constant": Atmosphere(CONSTANT_C),
            "layered": weather.for_events(events, chosen, clock),
        }
        candidates = {}
        for key, atmosphere in atmospheres.items():
            score, fitted_clock, f0, scores = joint_score(
                spec, events, chosen, observer, clock, atmosphere=atmosphere,
                shift_radius=4, shift_steps=17, frequency_radius=12, frequency_steps=121,
            )  # fmt: skip
            candidates[key] = {
                "position": observer.tolist(),
                "clock": fitted_clock,
                "f0": f0,
                "flights": [t.id for t in chosen],
                "score": score,
                "event_scores": scores,
            }
            print(set_name, "calibrated", key, fitted_clock - clock, f0, scores, flush=True)
        contexts[set_name] = (flights, events, chosen, clock, atmospheres)
        for name, key, options in cells():
            began = time.perf_counter()
            row = {
                "set": set_name,
                "cell": name,
                "role": "F1" if name == BASELINE else "F2" if name == PRIMARY else "F3",
                "flights": [t.id for t in chosen],
                "events": [asdict(e) for e in events],
                "initial_clock": clock,
                "initial_candidate": candidates[key],
                "layered_profiles": atmospheres["layered"].to_record() if key == "layered" else None,
                "result": None,
                "error": None,
            }
            try:
                row["result"] = run_fit(
                    spec, events, chosen, candidates[key], options, atmospheres[key], frame, audio
                )
            except Exception as exc:  # failures are outcomes
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = time.perf_counter() - began
            rows.append(row)
            save()
            print(set_name, name, "done", round(row["seconds"], 1), "s", row["error"] or "", flush=True)
            gc.collect()
    del tracks
    gc.collect()

    # F4: discovery at each set's F2 solution, then a joint refit with the confirmed passes.
    index = TraceIndex(DATA / "traces-velocity.npz")
    ridges = RidgeMap(spec, DiscoveryOptions().background_hz)
    primary_options = next(o for n, _, o in cells() if n == PRIMARY)
    for set_name, (flights, events, chosen, clock, atmospheres) in contexts.items():
        base = next(r for r in rows if r["set"] == set_name and r["cell"] == PRIMARY)
        for label, options in DISCOVERY.items():
            began = time.perf_counter()
            entry = {
                "set": set_name,
                "discovery": label,
                "options": asdict(options),
                "result": None,
                "error": None,
            }
            discoveries.append(entry)
            if base["result"] is None:
                entry["error"] = "no F2 solution"
                save()
                continue
            solution = base["result"]
            candidate = {k: solution[k] for k in ("position", "clock", "f0", "flights")}
            try:
                found = discover(
                    ridges, index, candidate, Atmosphere(CONSTANT_C), options, candidate["flights"]
                )
            except Exception as exc:
                entry["error"] = f"discovery {type(exc).__name__}: {exc}"
                save()
                continue
            summary = found["summary"]
            entry["summary"] = {k: v for k, v in summary.items() if k != "passes"}
            entry["passes"] = summary["passes"]
            entry["confirmed"] = [
                {k: p.get(k) for k in ("flight", "icao", "type", "registration", "heard_s", "distance_m",
                                       "altitude_m", "height_above_observer_m", "z", "f0", "window")}
                for p in summary["passes"]
                if p["confirmed"] and not p["used_in_fit"]
            ]  # fmt: skip
            entry["independent"] = summary["confirmed_flights"]
            print(set_name, label, "confirmed", summary["confirmed_flights"], flush=True)
            extra = []
            for event, track in zip(found["events"], found["tracks"], strict=True):
                if icao(track.id) in {icao(f) for f in candidate["flights"]}:
                    continue
                segment = velocity_segment(
                    DATA / "traces-velocity.npz",
                    track.id,
                    candidate["clock"] + event.window[0] - 240,
                    candidate["clock"] + event.window[1] + 240,
                )
                extra.append((event, segment))
            entry["discovered_events"] = [asdict(e) for e, _ in extra]
            if not extra:
                entry["refit"] = "no confirmed independent passes; no refit"
                save()
                continue
            dropped = []
            while True:
                combined = sorted(
                    [*zip(events, chosen, solution["f0"], strict=True), *((e, t, e.f0) for e, t in extra)],
                    key=lambda item: item[0].cpa,
                )
                all_events, all_tracks, f0 = map(list, zip(*combined, strict=True))
                initial = {**candidate, "f0": f0, "flights": [t.id for t in all_tracks]}
                layered = weather.for_events(all_events, all_tracks, solution["clock"])
                try:
                    entry["result"] = run_fit(
                        spec, all_events, all_tracks, initial, primary_options, layered, frame, None
                    )
                    entry["layered_profiles"] = layered.to_record()
                    entry["error"] = None
                    break
                except ValueError as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    if not extra:
                        break
                    weakest = min(range(len(extra)), key=lambda i: extra[i][0].score)
                    dropped.append(extra.pop(weakest)[1].id)
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    break
            entry["dropped_discoveries"] = dropped
            entry["joint_flights"] = [t.id for t in all_tracks]
            entry["joint_events"] = [asdict(e) for e in all_events]
            entry["seconds"] = time.perf_counter() - began
            save()
            print(
                set_name, label, "joint refit", entry["error"] or "ok", round(entry["seconds"], 1), flush=True
            )
            gc.collect()
    output["status"] = "complete"
    output["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    print("all fits saved", flush=True)


def describe(result, truth_position):
    """Error and region checks of one fit (evaluation only)."""
    error = np.asarray(result["position"])[:2] - truth_position[:2]
    out = {
        "error_m": float(np.linalg.norm(error)),
        "error_en_m": error.tolist(),
        "residual_median_abs_hz": result["event_residual_hz"],
        "residual_rms_hz": result["event_residual_rms_hz"],
        "valid_frames": result.get("valid_frames"),
        "frames": result.get("frames"),
        "evidence_accepted": bool(result["evidence"]["accepted"]),
        "success": result["success"],
        "at_bounds": result["at_bounds"],
        "clock_shift_from_start_s": None,
    }
    region = result.get("uncertainty", {})
    if "covariance_en_m2" in region:
        e90 = region["ellipses"]["90"]
        checks = evaluate(region, error)
        out.update(
            region_90_semi_axes_m=[e90["semi_major_m"], e90["semi_minor_m"]],
            region_90_bearing_deg=e90["major_axis_bearing_deg"],
            inside_50=checks["total"]["inside_50"],
            inside_90=checks["total"]["inside_90"],
            normalised_error=checks["total"]["normalised_error"],
            measurement_only=checks["measurement"],
        )
    else:
        out["region_error"] = region.get("error", "no region")
    return out


def evaluation(args):
    fits = json.loads((OUT / "fits.json").read_text())
    if fits.get("status") != "complete":
        raise SystemExit("fits.json is not complete; truth stays unread")
    preparation = json.loads((OUT / "preparation.json").read_text())
    # Evaluation only: nothing above received the truth.
    truth = json.loads((DATA / "published/provenance.json").read_text())["evaluation_truth"]
    _, frame = load_flights(DATA / "traces-velocity.npz", set())
    truth_position = frame.to_local([*truth, HEIGHT_M])

    def horizontal(latlon):
        return float(np.linalg.norm(frame.to_local([*latlon, HEIGHT_M])[:2] - truth_position[:2]))

    table = []
    for row in fits["trials"]:
        item = {"set": row["set"], "cell": row["cell"], "role": row.get("role"), "error": row["error"]}
        if row["result"]:
            item.update(describe(row["result"], truth_position))
            item["clock_shift_from_start_s"] = row["result"]["clock"] - row["initial_clock"]
            item["lat_lon"] = row["result"]["lat_lon_height"][:2]
        table.append(item)
    for entry in fits["discoveries"]:
        item = {
            "set": entry["set"],
            "cell": f"F4 {PRIMARY} + discovery {entry['discovery']}",
            "role": "F4",
            "error": entry["error"],
            "independent": entry.get("independent"),
            "joint_flights": entry.get("joint_flights"),
            "dropped": entry.get("dropped_discoveries"),
            "refit": entry.get("refit"),
        }
        if entry["result"]:
            item.update(describe(entry["result"], truth_position))
            item["lat_lon"] = entry["result"]["lat_lon_height"][:2]
        table.append(item)
    start_error = horizontal(preparation["start"]["latlon"])

    def error_of(set_name, cell):
        return next((t.get("error_m") for t in table if t["set"] == set_name and t["cell"] == cell), None)

    contrast = {s: {"F1": error_of(s, BASELINE), "F2": error_of(s, PRIMARY)} for s in SETS}
    factors = {
        "atmosphere": (0, "constant", "layered"),
        "trajectory": (1, "spline", "fused"),
        "measurement": (2, "ridge", "demod"),
    }
    factorial = {}
    for factor, (position, old, new) in factors.items():
        pairs = []
        for s in SETS:
            for name, _, _ in cells():
                parts = name.split("/")
                if parts[position] != old:
                    continue
                parts[position] = new
                a, b = error_of(s, name), error_of(s, "/".join(parts))
                pairs.append({"set": s, "from": name, "to": "/".join(parts), "from_m": a, "to_m": b})
        valid = [p for p in pairs if p["from_m"] is not None and p["to_m"] is not None]
        factorial[factor] = {
            "change": f"{old} -> {new}",
            "pairs": pairs,
            "improved": sum(p["to_m"] < p["from_m"] for p in valid),
            "valid_pairs": len(valid),
            "median_change_m": float(np.median([p["to_m"] - p["from_m"] for p in valid])) if valid else None,
        }
    # E3: the original project's numbers, read only now.
    with (DATA / "published/swfl_search_best_per_pair.csv").open() as stream:
        search = list(csv.DictReader(stream))
    top = max(search, key=lambda r: float(r["score"]))
    with (DATA / "published/swfl_refined_50m.csv").open() as stream:
        refined = list(csv.DictReader(stream))
    best_refined = max(refined, key=lambda r: float(r["score"]))
    published = {
        "top_grid_candidate": {
            "latlon": [float(top["lat"]), float(top["lon"])],
            "dist_m_column": float(top["dist_m"]),
            "error_m_our_frame": horizontal([float(top["lat"]), float(top["lon"])]),
        },
        "refined_50m_highest_score": {
            "row": best_refined,
            "error_m_our_frame": horizontal([float(best_refined["lat"]), float(best_refined["lon"])]),
        },
    }
    report = {
        "evaluation_truth": truth,
        "start_candidate_error_m": start_error,
        "table": table,
        "contrast_F1_F2": contrast,
        "factorial": factorial,
        "published": published,
    }
    dump(OUT / "results.json", report)
    print(f"start candidate error {start_error:.1f} m")
    header = (
        f"{'set':9s} {'cell':44s} {'err m':>7s} {'a90':>7s} {'b90':>7s} {'in50':>5s} {'in90':>5s} "
        f"{'d':>6s} {'med|r| Hz':>22s} {'evid':>6s}"
    )
    print(header)
    for t in table:
        if "error_m" not in t:
            print(f"{t['set']:9s} {str(t['cell']):44s} FAILED {t['error'] or t.get('refit')}")
            continue
        axes = t.get("region_90_semi_axes_m", [float("nan")] * 2)
        print(
            f"{t['set']:9s} {t['cell']:44s} {t['error_m']:7.1f} {axes[0]:7.1f} {axes[1]:7.1f} "
            f"{str(t.get('inside_50')):>5s} {str(t.get('inside_90')):>5s} {t.get('normalised_error', float('nan')):6.2f} "
            f"{' '.join(f'{x:.3f}' for x in t['residual_median_abs_hz']):>22s} "
            f"{'acc' if t['evidence_accepted'] else 'rej':>6s}"
        )
    print(json.dumps(contrast))
    for factor, value in factorial.items():
        print(
            factor,
            value["change"],
            f"improved {value['improved']}/{value['valid_pairs']}",
            value["median_change_m"],
        )
    print(json.dumps(published, indent=1))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=("prepare", "fit", "evaluate"))
    parser.add_argument(
        "--force", action="store_true", help="prepare: rewrite the plan (only before any fit)"
    )
    parser.add_argument("--notes", help="prepare: JSON file with plan notes and protocol deviations")
    args = parser.parse_args()
    {"prepare": prepare, "fit": fit, "evaluate": evaluation}[args.stage](args)


if __name__ == "__main__":
    main()
