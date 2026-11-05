"""Corridor-pruned coarse search with shared-clock multi-event scoring."""

import json
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from .physics import Atmosphere, closest_approaches
from .scoring import joint_score


def search(
    spec,
    events,
    tracks,
    points,
    atmosphere=Atmosphere(),
    method="brightness",
    clock=None,
    clock_tolerance=5.0,
    offset_tolerance=12.0,
    silence_radius=0.0,
    max_candidates=100_000,
    shift_radius=3.0,
    shift_steps=7,
    progress=None,
):
    if not events or (len(events) == 1 and clock is None):
        raise ValueError("a single event requires the recording start clock (UTC epoch seconds)")
    events = sorted(events, key=lambda event: event.cpa)
    if any(b.cpa <= a.cpa for a, b in zip(events, events[1:])):
        raise ValueError("events must have strictly increasing CPA times")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or len(points) == 0:
        raise ValueError("points must be a nonempty finite Nx3 array")
    if offset_tolerance <= 0 or clock_tolerance < 0 or max_candidates < 1:
        raise ValueError("invalid search tolerances or candidate budget")
    radius = max(max(e.distance_range) for e in events)
    radius = max(radius, silence_radius)
    tree = cKDTree(points[:, :2])
    passes = {}
    for index, track in enumerate(tracks):
        possible = set()
        # Densify chords so sparse samples cannot miss a ground corridor.
        for a, b, ta, tb in zip(track.positions[:-1], track.positions[1:], track.times[:-1], track.times[1:]):
            if tb - ta > track.max_gap:
                continue
            count = max(2, int(np.ceil(np.linalg.norm(b[:2] - a[:2]) / (radius * 0.5))) + 1)
            samples = np.linspace(a[:2], b[:2], count)
            for neighbors in tree.query_ball_point(samples, radius * 1.3):
                possible.update(neighbors)
        for point_id in possible:
            for heard, distance, speed in closest_approaches(track, points[point_id], atmosphere):
                if distance <= radius:
                    passes.setdefault(point_id, []).append((index, heard, distance, speed))
        if progress:
            progress("geometry", index + 1, len(tracks))
    results = []
    scored = 0
    for point_id, hits in passes.items():
        pools = [
            [
                hit
                for hit in hits
                if e.distance_range[0] <= hit[2] <= e.distance_range[1]
                and e.speed_range[0] <= hit[3] <= e.speed_range[1]
            ]
            for e in events
        ]
        for first in pools[0]:
            anchor_clock = first[1] - events[0].cpa
            if clock is not None and abs(anchor_clock - clock) > clock_tolerance:
                continue
            later = [
                [
                    hit
                    for hit in pool
                    if abs((hit[1] - first[1]) - (e.cpa - events[0].cpa)) <= offset_tolerance
                ]
                for e, pool in zip(events[1:], pools[1:])
            ]
            for rest in product(*later):
                selected = [first, *rest]
                if len({(h[0], h[1]) for h in selected}) != len(selected):
                    continue
                if silence_radius and any(
                    first[1] + 15 < h[1] < selected[-1][1] - 15
                    and h[2] < silence_radius
                    and h not in selected
                    for h in hits
                ):
                    continue
                scored += 1
                if scored > max_candidates:
                    raise ValueError(
                        "candidate budget exceeded; narrow region/ranges or increase --max-candidates"
                    )
                chosen = [tracks[h[0]] for h in selected]
                base_clock = anchor_clock if clock is None else clock
                allowed_shift = shift_radius if clock is None else min(shift_radius, clock_tolerance)
                score, fitted_clock, f0, individual = joint_score(
                    spec,
                    events,
                    chosen,
                    points[point_id],
                    base_clock,
                    method,
                    atmosphere,
                    shift_radius=allowed_shift,
                    shift_steps=shift_steps,
                )
                if score <= 0 or any(value <= 0 for value in individual):
                    continue
                results.append(
                    {
                        "position": points[point_id].tolist(),
                        "clock": fitted_clock,
                        "score": score,
                        "flights": [t.id for t in chosen],
                        "f0": f0,
                        "event_scores": individual,
                        "method": method,
                        "sound_speed_m_s": atmosphere.c,
                        "wind_enu_m_s": list(atmosphere.wind),
                    }
                )
                if progress and scored % 100 == 0:
                    progress("scoring", scored, None)
    return sorted(results, key=lambda r: r["score"], reverse=True)


def distinct_candidates(results, top=10, separation=300.0):
    """Retain competing flight assignments AND spatial modes within an assignment."""
    selected = []
    for result in results:
        if all(
            result["flights"] != other["flights"]
            or np.linalg.norm(np.array(result["position"])[:2] - np.array(other["position"])[:2]) > separation
            for other in selected
        ):
            selected.append(result)
        if len(selected) >= top:
            break
    return selected


def save_results(path, results, frame=None):
    payload = {"schema": 1, "frame": None if frame is None else [frame.lat, frame.lon], "candidates": results}
    if frame:
        for result in results:
            result["latitude"], result["longitude"], result["ellipsoid_height_m"] = map(
                float, frame.to_geodetic(result["position"])
            )
    with open(path, "w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
