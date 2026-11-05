"""Rerank the published 328-pair shortlist; this is not a new blind regional search."""

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from dopplertracker.data import load_tracks
from dopplertracker.detection import load_events
from dopplertracker.physics import Atmosphere, Trajectory, predict
from dopplertracker.signal import Spectrogram


def initialize():
    global SPEC, EVENTS, FRAME, TRACKS
    SPEC = Spectrogram.load("data/oyster-bay/spectrogram.npz").whiten()
    EVENTS = load_events(["data/oyster-bay/events.json"])
    tracks, FRAME = load_tracks("data/oyster-bay/traces.npz")
    TRACKS = {}
    for track in tracks:
        TRACKS.setdefault(track.id.split(":")[0], []).append(track)


def choose_track(icao, event, clock):
    matching = [
        t
        for t in TRACKS.get(icao, [])
        if t.times[-1] >= clock + event.window[0] - 120 and t.times[0] <= clock + event.window[1] + 120
    ]
    times, positions = [], []
    for track in matching:
        mask = (track.times >= clock + event.window[0] - 120) & (track.times <= clock + event.window[1] + 120)
        times.extend(track.times[mask])
        positions.extend(track.positions[mask])
    if len(times) < 3:
        raise ValueError(f"Insufficient geometric altitude data for {icao}")
    t, indices = np.unique(times, return_index=True)
    return Trajectory(icao, t - clock, np.asarray(positions)[indices], max_gap=30)


def curve_scores(spec, times, factor, frequencies, harmonics, method):
    hz = frequencies[:, None] * factor[None, :]
    values = np.zeros(hz.shape)
    weight = 0.0
    for harmonic in range(1, harmonics + 1):
        center = spec.sample(times[None, :], hz * harmonic)
        if method == "contrast":
            background = 0.5 * (
                spec.sample(times[None, :], hz * harmonic - 3)
                + spec.sample(times[None, :], hz * harmonic + 3)
            )
            center = np.maximum(center - background, 0)
        values += center / np.sqrt(harmonic)
        weight += 1 / np.sqrt(harmonic)
    return np.mean(values, axis=1) / weight


def trial(row):
    began = time.perf_counter()
    output = {
        "original_rank": int(row["rank"]),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "flights": [row["icao1"], row["icao2"]],
        "error": None,
        "scores": {},
    }
    try:
        clock = float(row["t_heard1"]) - EVENTS[0].cpa + float(row["shift"])
        tracks = [choose_track(icao, event, clock) for icao, event in zip(output["flights"], EVENTS)]
        position = FRAME.to_local([output["lat"], output["lon"], -30.0])
        best = {method: {"score": -1.0} for method in ("brightness", "contrast")}
        for shift in np.linspace(-4, 4, 17):
            per_method = {method: [] for method in best}
            for event, track in zip(EVENTS, tracks):
                times = event.sample_times(SPEC)
                factor, _ = predict(track, position, times + shift, atmosphere=Atmosphere(343))
                f0 = np.linspace(event.f0 - 12, event.f0 + 12, 121)
                for method in best:
                    scores = curve_scores(SPEC, times, factor, f0, event.harmonics, method)
                    index = int(np.argmax(scores))
                    per_method[method].append((float(scores[index]), float(f0[index])))
            for method, values in per_method.items():
                score = float(np.mean([value[0] for value in values]))
                if score > best[method]["score"]:
                    best[method] = {
                        "score": score,
                        "clock": clock + float(shift),
                        "f0": [value[1] for value in values],
                        "event_scores": [value[0] for value in values],
                    }
        output["scores"] = best
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
    output["seconds"] = time.perf_counter() - began
    return output


def main():
    directory = Path("outputs/oyster-bay/reranking")
    directory.mkdir(parents=True, exist_ok=True)
    with Path("data/oyster-bay/published/oyster_bay_search_best_per_pair.csv").open() as stream:
        candidates = list(csv.DictReader(stream))
    rows = []
    with ProcessPoolExecutor(max_workers=4, initializer=initialize) as executor:
        jobs = [executor.submit(trial, row) for row in candidates]
        with (directory / "trials.jsonl").open("w") as stream:
            for job in as_completed(jobs):
                row = job.result()
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if len(rows) % 20 == 0:
                    print(f"{len(rows)}/{len(candidates)} flight-pair candidates scored", flush=True)
    rows.sort(key=lambda r: r["original_rank"])
    summary = {}
    for method in ("brightness", "contrast"):
        ranked = sorted(
            [r for r in rows if not r["error"]], key=lambda r: r["scores"][method]["score"], reverse=True
        )
        truth_rank = next((i + 1 for i, r in enumerate(ranked) if r["flights"] == ["a1dfeb", "ace391"]), None)
        summary[method] = {
            "correct_pair_rank": truth_rank,
            "scored_pairs": len(ranked),
            "top5": [{**r, "rank": i + 1} for i, r in enumerate(ranked[:5])],
        }
    report = {
        "validation": "Reranking published candidate positions; known event windows, no new blind regional search.",
        "height_ellipsoid_m": -30.0,
        "sound_speed_m_s": 343.0,
        "attempts": len(rows),
        "failures": sum(r["error"] is not None for r in rows),
        "summary": summary,
        "trials": rows,
    }
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
