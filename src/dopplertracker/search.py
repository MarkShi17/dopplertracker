"""Corridor-pruned coarse search with shared-clock multi-event scoring.

``search`` is the reference: one scalar closest-approach solve per (grid point,
track).  ``fast_search`` is an opt-in regional path that computes the same
closest approaches on resampled piecewise-linear tracks, vectorised over grid
points, joins event timings by sorted keys, and scores a bounded number of
candidates per flight combination.
"""

import json
from itertools import product
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .physics import Atmosphere, closest_approaches, predict
from .scoring import joint_score, subsonic


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


def _delay(r, c, wind):
    """Homogeneous-medium travel time for source-minus-observer offsets ``r`` (..., 3)."""
    squared = np.einsum("...i,...i->...", r, r)
    if not np.any(wind):
        return np.sqrt(squared) / c
    a = c * c - wind @ wind
    rw = r @ wind
    return (rw + np.sqrt(rw * rw + a * squared)) / a


def _vertex(h0, h2, y0, y1, y2):
    """Offset of the parabola vertex through (h0, y0), (0, y1), (h2, y2); h0 < 0 < h2.

    Clipped to [h0, h2]; zero where the samples are not convex.
    """
    a = ((y0 - y1) / h0 - (y2 - y1) / h2) / (h0 - h2)
    b = (y0 - y1) / h0 - a * h0
    vertex = np.divide(-b, 2 * a, out=np.zeros_like(a), where=a > 0)
    return np.clip(vertex, h0, h2)


class _Samples:
    """Tracks resampled on their own interpolants, kept only where a pass can matter.

    Each segment is sampled every ``step`` seconds plus at its raw sample times, so
    the polyline passes through every reported position.  Each run is a contiguous
    kept stretch of one segment; ``open_start``/``open_end`` mark run ends that are
    real segment ends, where ``closest_approaches`` also reports endpoint
    approaches.  Cut ends carry ``pad`` samples beyond the kept range so a minimum
    just inside the range keeps both neighbours.  Raw samples of retained
    segments are kept too: ``closest_approaches`` detects minima on them.
    """

    def __init__(self, tracks, box, step, speed_window=None, time_window=None, pad=2):
        xmin, xmax, ymin, ymax = box
        times, positions, speeds, intervals, runs = [], [], [], [], []
        raw_times, raw_positions = [], []
        count = raw_count = 0
        for index, track in enumerate(tracks):
            for piece, ((start, end, spline, derivative), idx) in enumerate(
                zip(track.pieces, track.segments)
            ):
                raw = track.positions[idx]
                if (
                    raw[:, 0].max() < xmin
                    or raw[:, 0].min() > xmax
                    or raw[:, 1].max() < ymin
                    or raw[:, 1].min() > ymax
                ):
                    continue
                if time_window is not None and (end < time_window[0] or start > time_window[1]):
                    continue
                raw_rel = track.times[idx] - start
                grid = np.arange(0.0, end - start, step)
                nearest = np.clip(np.searchsorted(raw_rel, grid), 1, len(raw_rel) - 1)
                gap = np.minimum(np.abs(grid - raw_rel[nearest - 1]), np.abs(grid - raw_rel[nearest]))
                rel = np.union1d(raw_rel, grid[gap > 1e-3])
                xyz = spline(rel)
                xyz[np.searchsorted(rel, raw_rel)] = raw
                speed = np.linalg.norm(derivative(rel), axis=1)
                keep = (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
                if speed_window is not None:
                    keep &= (speed >= speed_window[0]) & (speed <= speed_window[1])
                if time_window is not None:
                    keep &= (start + rel >= time_window[0]) & (start + rel <= time_window[1])
                if not keep.any():
                    continue
                # Dilate by pad samples ("same" would pad inputs shorter than the kernel).
                keep = np.convolve(keep, np.ones(2 * pad + 1))[pad : pad + len(keep)] > 0.5
                edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(np.int8), 0]))
                interval = raw_count + np.clip(np.searchsorted(raw_rel, rel, "right") - 1, 0, len(idx) - 2)
                for a, b in zip(edges[::2], edges[1::2]):
                    if b - a < 2:
                        continue
                    runs.append(
                        (
                            index,
                            count,
                            count + b - a,
                            a == 0,
                            b == len(rel),
                            raw_count,
                            raw_count + len(idx),
                            piece,
                        )
                    )
                    times.append(start + rel[a:b])
                    positions.append(xyz[a:b])
                    speeds.append(speed[a:b])
                    intervals.append(interval[a:b])
                    count += b - a
                raw_times.append(track.times[idx])
                raw_positions.append(raw)
                raw_count += len(idx)
        self.t = np.concatenate(times) if times else np.empty(0)
        self.xyz = np.concatenate(positions) if positions else np.empty((0, 3))
        self.speed = np.concatenate(speeds) if speeds else np.empty(0)
        self.interval = np.concatenate(intervals) if intervals else np.empty(0, dtype=np.int64)
        self.raw_t = np.concatenate(raw_times) if raw_times else np.empty(0)
        self.raw_xyz = np.concatenate(raw_positions) if raw_positions else np.empty((0, 3))
        runs = np.array(runs, dtype=np.int64).reshape(-1, 8)
        self.run_track, self.run_start, self.run_stop = runs[:, 0], runs[:, 1], runs[:, 2]
        self.open_start, self.open_end = runs[:, 3].astype(bool), runs[:, 4].astype(bool)
        self.run_raw_start, self.run_raw_stop, self.run_piece = runs[:, 5], runs[:, 6], runs[:, 7]
        self.sample_run = np.repeat(np.arange(len(runs)), self.run_stop - self.run_start)
        # A minimum refined within its two adjacent pieces lies at most the longer
        # piece's length closer than the sample itself (``slack``).
        piece = np.linalg.norm(np.diff(self.xyz, axis=0), axis=1)
        piece[self.run_stop[:-1] - 1] = 0.0  # no piece joins consecutive runs
        self.slack = np.maximum(np.r_[0.0, piece], np.r_[piece, 0.0]) if len(self.t) else np.empty(0)

    def windows(self, size):
        """Consecutive sample blocks of each run, with one neighbour on each side.

        Columns: run, a, b (own samples), lo, hi (with neighbours), centre x, y,
        horizontal extent about the centre, and the largest slack.
        """
        rows = []
        for run, (start, stop) in enumerate(zip(self.run_start, self.run_stop)):
            for a in range(start, stop, size):
                b = min(a + size, stop)
                lo, hi = max(a - 1, start), min(b + 1, stop)
                xy = self.xyz[lo:hi, :2]
                centre = xy.mean(axis=0)
                extent = np.sqrt(((xy - centre) ** 2).sum(axis=1).max())
                rows.append((run, a, b, lo, hi, centre[0], centre[1], extent, self.slack[lo:hi].max()))
        return np.array(rows, dtype=float).reshape(-1, 9)


def _metric(g, q, c, wind):
    """Travel time, or squared distance in still air (same ordering), observers x sources."""
    if np.any(wind):
        return _delay(q[None, :, :] - g[:, None, :], np.float32(c), np.asarray(wind, np.float32))
    # One matrix product; centred float32 coordinates keep the cancellation harmless.
    return (g * g).sum(axis=1)[:, None] + (q * q).sum(axis=1)[None, :] - 2.0 * (g @ q.T)


def _window_minima(samples, window, points, c, wind, radius):
    """Local travel-time minima over one window, for every observer in ``points``.

    Returns (observer row, sample index, endpoint flag) for minima that can lie
    within ``radius`` once refined.  Interior minima need a neighbour on both
    sides.  Endpoint approaches exist only at real segment ends and, as in
    ``closest_approaches``, compare the end with its raw neighbour.
    """
    run, a, b, lo, hi = (int(v) for v in window[:5])
    centre = samples.xyz[lo:hi].mean(axis=0)
    g = (points - centre).astype(np.float32)
    q = (samples.xyz[lo:hi] - centre).astype(np.float32)
    delay = _metric(g, q, c, wind)
    minimum = np.zeros(delay.shape, dtype=bool)
    first, last = max(a - lo, 1), min(b - lo, hi - lo - 1)
    if first < last:
        middle = delay[:, first:last]
        minimum[:, first:last] = (middle <= delay[:, first - 1 : last - 1]) & (
            middle < delay[:, first + 1 : last + 1]
        )
    endpoint = np.zeros(hi - lo, dtype=bool)
    for column, closed, neighbour in (
        (0, a == samples.run_start[run] and samples.open_start[run], samples.run_raw_start[run] + 1),
        (-1, b == samples.run_stop[run] and samples.open_end[run], samples.run_raw_stop[run] - 2),
    ):
        if closed:
            raw = (samples.raw_xyz[neighbour : neighbour + 1] - centre).astype(np.float32)
            minimum[:, column] = delay[:, column] <= _metric(g, raw, c, wind)[:, 0]
            endpoint[column] = True
    rows, cols = np.nonzero(minimum)
    reach = radius + np.where(endpoint[cols], 0.0, samples.slack[lo + cols]) + 1.0
    near = np.einsum("ij,ij->i", g[rows] - q[cols], g[rows] - q[cols]) <= reach * reach
    return rows[near], lo + cols[near], endpoint[cols[near]]


def _refine_minima(samples, observers, index, endpoint, c, wind):
    """Heard time, slant distance, speed, travel time, raw-minimum id and validity.

    An interior minimum at sample j moves to the vertex of the parabola through
    the squared travel times at j - 1, j, j + 1: exact for a straight
    constant-speed pass in still air, and second order in the step on curves.
    As in ``closest_approaches`` an interior approach must lie in the bracket of
    a raw sample that is no farther than its raw neighbours; others (interpolant
    wiggles between raw samples) are invalid.  The id names that raw sample (or
    the segment end), so one approach per bracket can be kept.
    """
    t, xyz, speed = samples.t, samples.xyz, samples.speed
    lower = index.copy()
    fraction = np.zeros(len(index))
    inner = ~endpoint
    if inner.any():
        j, p = index[inner], observers[inner]
        y0, y1, y2 = (_delay(xyz[n] - p, c, wind) ** 2 for n in (j - 1, j, j + 1))
        offset = _vertex(t[j - 1] - t[j], t[j + 1] - t[j], y0, y1, y2)
        before = offset < 0
        piece = np.where(before, j - 1, j)
        lower[inner] = piece
        fraction[inner] = (t[j] + offset - t[piece]) / (t[piece + 1] - t[piece])
    # Endpoints use fraction zero at their own sample; interior minima interpolate within a piece.
    upper = np.where(endpoint, lower, lower + 1)
    emit = t[lower] + fraction * (t[upper] - t[lower])
    position = xyz[lower] + fraction[:, None] * (xyz[upper] - xyz[lower])
    offset = position - observers
    delay = _delay(offset, c, wind)
    at_speed = speed[lower] + fraction * (speed[upper] - speed[lower])
    # Samples include every raw time, so a piece lies inside one raw interval [k, k + 1].
    run = samples.sample_run[index]
    first, final = samples.run_raw_start[run], samples.run_raw_stop[run] - 1
    k = samples.interval[lower]

    def raw_minimum(i):
        interior = (i > first) & (i < final)
        i = np.where(interior, i, first + 1)
        here, before, after = (
            _delay(samples.raw_xyz[np.minimum(n, len(samples.raw_xyz) - 1)] - observers, c, wind)
            for n in (i, i - 1, i + 1)
        )
        return interior & (here <= before) & (here <= after)

    left, right = raw_minimum(k), raw_minimum(k + 1)
    raw_id = np.where(
        endpoint, np.where(index == samples.run_start[run], first, final), np.where(left, k, k + 1)
    )
    valid = endpoint | left | right
    return emit + delay, np.linalg.norm(offset, axis=1), at_speed, delay, raw_id, valid


def _polish(samples, tracks, run, emit, raw_id, endpoint, observers, c, wind, iterations=4, h=0.02):
    """Heard time, slant distance and speed of each approach on its track's own interpolant.

    Starting from the sampled estimate, Newton steps on finite differences of the
    travel time move each interior approach to the interpolant's local minimum,
    kept inside the raw bracket of its raw minimum as in ``closest_approaches``.
    Endpoint approaches stay at the segment end.  One evaluation per run and step.
    """
    heard, distance, speed, delay = (np.empty(len(run)) for _ in range(4))
    order = np.argsort(run, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(run[order]) != 0, True]) if len(run) else []
    last = len(samples.raw_t) - 1
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = order[a:b]
        r = run[rows[0]]
        start, end, spline, derivative = tracks[samples.run_track[r]].pieces[samples.run_piece[r]]
        span, p, inner, i = end - start, observers[rows], ~endpoint[rows], raw_id[rows]
        t = np.clip(emit[rows] - start, 0.0, span)
        lo = np.where(inner, samples.raw_t[np.maximum(i - 1, 0)] - start, t)
        hi = np.where(inner, samples.raw_t[np.minimum(i + 1, last)] - start, t)
        for _ in range(iterations if inner.any() else 0):
            probe = np.clip(t[None, :] + np.array([[-h], [0.0], [h]]), 0.0, span)
            tau = _delay(spline(probe) - p[None], c, wind)
            width = probe[2] - probe[0]
            slope = np.divide(tau[2] - tau[0], width, out=np.zeros_like(width), where=width > 0)
            curve = (tau[2] - 2 * tau[1] + tau[0]) / (h * h)
            step = np.divide(-slope, curve, out=np.zeros_like(curve), where=curve > 0)
            t = np.clip(t + step, lo, hi)
        offset = spline(t) - p
        delay[rows] = _delay(offset, c, wind)
        heard[rows] = start + t + delay[rows]
        distance[rows] = np.linalg.norm(offset, axis=1)
        speed[rows] = np.linalg.norm(derivative(t), axis=1)
    return heard, distance, speed, delay


def _tile_passes(samples, tracks, windows, points, tree, radius, c, wind, chunk):
    """Every closest approach within ``radius`` of a tile's points, in bounded memory.

    Minima are refined, polished and deduplicated in chunks of about ``chunk``
    rows, then once more across chunks.  Returns compactly typed (point row, run,
    heard time, slant distance, speed).
    """
    parts, pending, size = [], [], 0

    def flush():
        row, index, endpoint = (np.concatenate(column) for column in zip(*pending))
        heard, distance, _, delay, raw_id, valid = _refine_minima(
            samples, points[row], index, endpoint, c, wind
        )
        keep = valid & (distance <= radius + samples.slack[index] + 1.0)
        keep[keep] = _deduplicate(row[keep], raw_id[keep], delay[keep])
        row, index, endpoint, raw_id = row[keep], index[keep], endpoint[keep], raw_id[keep]
        run = samples.sample_run[index]
        emit = (heard - delay)[keep]
        heard, distance, speed, delay = _polish(
            samples, tracks, run, emit, raw_id, endpoint, points[row], c, wind
        )
        keep = distance <= radius
        parts.append(
            (
                row[keep].astype(np.int32),
                run[keep].astype(np.int32),
                heard[keep],
                distance[keep].astype(np.float32),
                speed[keep].astype(np.float32),
                raw_id[keep].astype(np.int32),
                delay[keep].astype(np.float32),
            )
        )
        pending.clear()

    for w in windows:
        neighbours = tree.query_ball_point(w[5:7], radius + w[8] + w[7] + 1.0)
        if not neighbours:
            continue
        neighbours = np.asarray(neighbours)
        r, j, e = _window_minima(samples, w, points[neighbours], c, wind, radius)
        pending.append((neighbours[r], j, e))
        size += len(j)
        if size >= chunk:
            flush()
            size = 0
    if pending:
        flush()
    if not parts:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, np.empty(0), np.empty(0, np.float32), np.empty(0, np.float32)
    row, run, heard, distance, speed, raw_id, delay = (np.concatenate(column) for column in zip(*parts))
    if len(parts) > 1:
        keep = _deduplicate(row, raw_id, delay)
        row, run, heard, distance, speed = row[keep], run[keep], heard[keep], distance[keep], speed[keep]
    return row, run, heard, distance, speed


def _deduplicate(point, raw_id, delay):
    """One approach per (observer, raw minimum): the one with the shortest travel time."""
    keep = np.zeros(len(point), dtype=bool)
    if not len(point):
        return keep
    order = np.lexsort((delay, raw_id, point))
    first = np.r_[True, (point[order][1:] != point[order][:-1]) | (raw_id[order][1:] != raw_id[order][:-1])]
    keep[order[first]] = True
    return keep


def _combinations(point, heard, distance, matches, events, clock, clock_tolerance, offset_tolerance, silence):
    """Sorted-key timing join: rows of pass indices, one column per event.

    Keys place every point's passes on its own stretch of a single number line, so
    one ``searchsorted`` finds each anchor's later events within the offset tolerance.
    """
    span = 1e6
    origin = heard.min() if len(heard) else 0.0
    key = point * span + (heard - origin)
    combos = np.flatnonzero(matches[0])[:, None]
    if clock is not None:
        combos = combos[np.abs(heard[combos[:, 0]] - events[0].cpa - clock) <= clock_tolerance]
    for k, event in enumerate(events[1:], 1):
        pool = np.flatnonzero(matches[k])
        shifted = key[pool] - (event.cpa - events[0].cpa)
        order = np.argsort(shifted, kind="stable")
        shifted, pool = shifted[order], pool[order]
        anchor = key[combos[:, 0]]
        lo = np.searchsorted(shifted, anchor - offset_tolerance, "left")
        hi = np.searchsorted(shifted, anchor + offset_tolerance, "right")
        counts = hi - lo
        rows = np.repeat(np.arange(len(combos)), counts)
        within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        combos = np.column_stack((combos[rows], pool[np.repeat(lo, counts) + within]))
    # A single pass cannot explain two events.
    distinct = np.ones(len(combos), dtype=bool)
    for i in range(combos.shape[1]):
        for j in range(i + 1, combos.shape[1]):
            distinct &= combos[:, i] != combos[:, j]
    combos = combos[distinct]
    if silence and len(combos):
        quiet = np.sort(key[distance < silence])
        first, final = key[combos[:, 0]], key[combos[:, -1]]
        between = np.searchsorted(quiet, final - 15, "left") - np.searchsorted(quiet, first + 15, "right")
        for column in combos.T:
            between -= (distance[column] < silence) & (key[column] > first + 15) & (key[column] < final - 15)
        combos = combos[between <= 0]
    return combos


class _Scorer:
    """``joint_score`` batched over clock shifts and source frequencies.

    The same receive times, frequency options and selection order are used, so the
    result matches ``joint_score``; shifted receive times that coincide are
    propagated once.
    """

    def __init__(self, spec, events, method, atmosphere, shift_radius, shift_steps, radius=2.0, steps=9):
        if method not in ("brightness", "contrast"):
            raise ValueError("method must be brightness or contrast")
        self.spec, self.events, self.method, self.atmosphere = spec, events, method, atmosphere
        self.times = [event.sample_times(spec) for event in events]
        self.shifts = np.linspace(-shift_radius, shift_radius, shift_steps)
        self.options = [np.linspace(e.f0 - radius, e.f0 + radius, steps) for e in events]

    def _curves(self, k, factor):
        harmonics = self.events[k].harmonics
        frequency = factor[:, None, :] * self.options[k][None, :, None]
        times = np.broadcast_to(self.times[k], frequency.shape)
        values = np.zeros(frequency.shape)
        weight_sum = 0.0
        for h in range(1, harmonics + 1):
            weight = 1 / np.sqrt(h)
            center = self.spec.sample(times, frequency * h)
            if self.method == "contrast":
                background = 0.5 * (
                    self.spec.sample(times, frequency * h - 3) + self.spec.sample(times, frequency * h + 3)
                )
                center = np.maximum(center - background, 0)
            values += weight * center
            weight_sum += weight
        return values.mean(axis=-1) / weight_sum

    def __call__(self, tracks, observer, clock):
        tables = []
        for k, track in enumerate(tracks):
            receive = ((self.times[k] + clock)[None, :] + self.shifts[:, None]).ravel()
            _, first, inverse = np.unique(np.round(receive, 6), return_index=True, return_inverse=True)
            factor, _ = predict(track, observer, receive[first], atmosphere=self.atmosphere)
            tables.append(self._curves(k, factor[inverse.ravel()].reshape(len(self.shifts), -1)))
        best = (-np.inf, clock, [], [])
        for s, shift in enumerate(self.shifts):
            picks = [int(np.argmax(table[s])) for table in tables]
            scores = [float(table[s, i]) for table, i in zip(tables, picks)]
            total = float(np.mean(scores))
            if total > best[0]:
                fitted = [float(options[i]) for options, i in zip(self.options, picks)]
                best = total, float(clock + shift), fitted, scores
        return best


def _cells(xy, cell):
    """Indices of one row per occupied square cell: the row nearest the cell centre."""
    ids = np.floor(xy / cell).astype(np.int64)
    _, inverse = np.unique(ids, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    offset = np.linalg.norm(xy - (ids + 0.5) * cell, axis=1)
    order = np.lexsort((offset, inverse))
    return order[np.r_[True, inverse[order][1:] != inverse[order][:-1]]]


def _thin(xy, spacing, cap):
    """At most ``cap`` rows, one per cell, the cell size doubling from ``spacing``."""
    cell = spacing
    while True:
        cell *= 2
        keep = _cells(xy, cell)
        if len(keep) <= cap:
            return keep, cell


def _climb(xy, tree, evaluate, start, cell, spacing):
    """Coarse-to-fine local search from row ``start`` (``evaluate`` is memoised).

    Around the current best, rows within ``radius`` are thinned to cells of half
    that radius and scored; the radius halves once nothing improves, down to the
    immediate grid neighbours.
    """
    current, radius, finest = start, cell, 1.5 * spacing
    while True:
        near = np.asarray(tree.query_ball_point(xy[current], radius), dtype=int)
        if radius / 2 > spacing:
            near = near[_cells(xy[near], radius / 2)]
        top = max([current, *near.tolist()], key=evaluate)
        if evaluate(top) > evaluate(current):
            current = top
        elif radius <= finest:
            return current
        else:
            radius = max(radius / 2, finest)


def fast_search(
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
    shift_radius=3.0,
    shift_steps=7,
    exclude_types=(),
    per_combination=48,
    refine_combinations=50,
    seeds=4,
    step=1.0,
    window=32,
    tile_size=10_000.0,
    chunk=400_000,
    speed_margin=5.0,
    progress=None,
):
    """Regional ``search`` with vectorised closest approaches; returns (results, summary).

    ``exclude_types`` drops tracks by ``metadata["type"]`` (e.g. jets).  Each track
    segment is resampled every ``step`` seconds (and at its raw times) on its own
    interpolant and kept only near the region, near the events' speed ranges
    (unless a silence filter needs every pass) and, with a known clock, near the
    events' times.  Closest approaches are local travel-time minima over these
    samples, refined to sub-sample precision, for all grid points of a tile at
    once.  Filters and the timing rule are those of ``search``.  Per flight combination
    at most ``per_combination`` spatially thinned candidates are scored (``None``
    scores all), then the ``refine_combinations`` best combinations are
    hill-climbed at full resolution from their ``seeds`` best candidates.  A
    combination containing a track the propagator rejects as supersonic is skipped
    and counted.  ``atmosphere`` must be homogeneous.
    """
    started = perf_counter()
    if not isinstance(atmosphere, Atmosphere):
        raise ValueError("fast_search supports a homogeneous atmosphere only")
    if not events or (len(events) == 1 and clock is None):
        raise ValueError("a single event requires the recording start clock (UTC epoch seconds)")
    events = sorted(events, key=lambda event: event.cpa)
    if any(b.cpa <= a.cpa for a, b in zip(events, events[1:])):
        raise ValueError("events must have strictly increasing CPA times")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or len(points) == 0:
        raise ValueError("points must be a nonempty finite Nx3 array")
    if offset_tolerance <= 0 or clock_tolerance < 0 or step <= 0 or window < 1 or tile_size <= 0:
        raise ValueError("invalid search tolerances, step, window or tile size")
    if per_combination is not None and per_combination < 1:
        raise ValueError("per_combination must be positive or None")
    exclude_types = set(exclude_types)
    supplied = len(tracks)
    tracks = [track for track in tracks if track.metadata.get("type") not in exclude_types]
    c, wind = atmosphere.c, np.asarray(atmosphere.wind, dtype=float)
    radius = max(max(max(e.distance_range) for e in events), silence_radius)
    summary = dict(tracks=len(tracks), excluded_tracks=supplied - len(tracks), points=len(points))

    lower, upper = points[:, :2].min(axis=0) - radius, points[:, :2].max(axis=0) + radius
    speeds = None
    if not silence_radius:
        speeds = (
            min(e.speed_range[0] for e in events) - speed_margin,
            max(e.speed_range[1] for e in events) + speed_margin,
        )
    times = None
    if clock is not None:
        times = (
            clock + events[0].cpa - clock_tolerance - radius / (c - np.linalg.norm(wind)) - 2 * step,
            clock + events[-1].cpa + clock_tolerance + offset_tolerance + 2 * step,
        )
    samples = _Samples(tracks, (lower[0], upper[0], lower[1], upper[1]), step, speeds, times)
    windows = samples.windows(window)
    summary.update(samples=len(samples.t), runs=len(samples.run_track), windows=len(windows))

    # Tiles bound memory; every stage up to scoring is per point, so tiling is exact.
    tile_ids = np.floor((points[:, :2] - points[:, :2].min(axis=0)) / tile_size).astype(np.int64)
    _, tile_of = np.unique(tile_ids, axis=0, return_inverse=True)
    tile_of = tile_of.ravel()
    tile_order = np.argsort(tile_of, kind="stable")
    tile_bounds = np.flatnonzero(np.r_[True, np.diff(tile_of[tile_order]) != 0, True])
    passes = 0
    found = []
    for n, (a, b) in enumerate(zip(tile_bounds[:-1], tile_bounds[1:])):
        members = tile_order[a:b]
        local = points[members]
        tree = cKDTree(local[:, :2])
        lo, hi = local[:, :2].min(axis=0), local[:, :2].max(axis=0)
        margin = radius + windows[:, 8] + windows[:, 7] + 1.0
        near = np.flatnonzero(
            (windows[:, 5] >= lo[0] - margin)
            & (windows[:, 5] <= hi[0] + margin)
            & (windows[:, 6] >= lo[1] - margin)
            & (windows[:, 6] <= hi[1] + margin)
        )
        row, run, heard, distance, speed = _tile_passes(
            samples, tracks, windows[near], local, tree, radius, c, wind, chunk
        )
        if progress:
            progress("geometry", n + 1, len(tile_bounds) - 1)
        passes += len(row)
        track = samples.run_track[run]
        matches = [
            (e.distance_range[0] <= distance)
            & (distance <= e.distance_range[1])
            & (e.speed_range[0] <= speed)
            & (speed <= e.speed_range[1])
            for e in events
        ]
        combos = _combinations(
            row,
            heard,
            distance,
            matches,
            events,
            clock,
            clock_tolerance,
            offset_tolerance,
            silence_radius,
        )
        if len(combos):
            found.append((members[row[combos[:, 0]]], track[combos], heard[combos[:, 0]]))
    summary.update(passes=passes, geometry_seconds=perf_counter() - started)

    if found:
        point_of = np.concatenate([f[0] for f in found])
        combination = np.concatenate([f[1] for f in found])
        anchor = np.concatenate([f[2] for f in found])
    else:
        point_of, combination, anchor = np.empty(0, int), np.empty((0, len(events)), int), np.empty(0)
    base = anchor - events[0].cpa if clock is None else np.full(len(anchor), float(clock))
    allowed_shift = shift_radius if clock is None else min(shift_radius, clock_tolerance)
    scorer = _Scorer(spec, events, method, atmosphere, allowed_shift, shift_steps)
    keys, group = np.unique(combination, axis=0, return_inverse=True)
    group = group.ravel()
    order = np.argsort(group, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(group[order]) != 0, True]) if len(group) else np.zeros(1, int)
    if len(points) > 1:
        sample = points[:: max(1, len(points) // 2000), :2]
        spacing = float(np.median(cKDTree(points[:, :2]).query(sample, k=2)[0][:, 1]))
    else:
        spacing = 1.0
    supersonic = {}
    scored = {}
    skipped = 0

    def score(row):
        if row not in scored:
            chosen = [tracks[i] for i in combination[row]]
            scored[row] = scorer(chosen, points[point_of[row]], base[row])
        return scored[row][0]

    plans = []
    for g, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        rows = order[a:b]
        for i in keys[g]:
            if i not in supersonic:
                supersonic[i] = not subsonic(tracks[i], atmosphere)
        if any(supersonic[i] for i in keys[g]):
            skipped += len(rows)
            continue
        xy = points[point_of[rows], :2]
        if per_combination is None or len(rows) <= per_combination:
            chosen, cell = np.arange(len(rows)), None
        else:
            chosen, cell = _thin(xy, spacing, per_combination)
        values = [score(int(rows[i])) for i in chosen]
        plans.append((max(values), rows, xy, cell))
        if progress:
            progress("scoring", len(scored), None)
    plans.sort(key=lambda plan: plan[0], reverse=True)
    for best, rows, xy, cell in plans[:refine_combinations]:
        if cell is None:
            continue
        tree = cKDTree(xy)
        coarse = [i for i, r in enumerate(rows) if int(r) in scored]
        for seed in sorted(coarse, key=lambda i: scored[int(rows[i])][0], reverse=True)[:seeds]:
            _climb(xy, tree, lambda i: score(int(rows[i])), seed, cell, spacing)
        if progress:
            progress("refining", len(scored), None)

    results = []
    for row, (value, fitted_clock, f0, individual) in scored.items():
        if value <= 0 or any(v <= 0 for v in individual):
            continue
        results.append(
            {
                "position": points[point_of[row]].tolist(),
                "clock": fitted_clock,
                "score": value,
                "flights": [tracks[i].id for i in combination[row]],
                "f0": f0,
                "event_scores": individual,
                "method": method,
                "sound_speed_m_s": atmosphere.c,
                "wind_enu_m_s": list(atmosphere.wind),
            }
        )
    summary.update(
        candidates=len(point_of),
        combinations=len(keys),
        scored=len(scored),
        skipped_supersonic=skipped,
        supersonic_tracks=int(sum(supersonic.values())),
        results=len(results),
        seconds=perf_counter() - started,
    )
    return sorted(results, key=lambda r: r["score"], reverse=True), summary


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
