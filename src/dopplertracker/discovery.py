"""Candidate-conditioned flyover discovery and independent confirmation.

Given a candidate (observer position and the UTC recording clock), every ADS-B
track predicts the complete shape of its Doppler curve; only the source
frequency (and optionally a small linear drift) is unknown.  ``discover``
enumerates audible passes, scans each predicted ratio curve against the
spectrogram's local contrast, and compares the best score with the same curve
misaligned in time (data shifted by tens of seconds) and mirrored in time.

Flights that were not used to fit the candidate but appear where it predicts
are independent evidence for it.  The z-scores are empirical heuristics, not
calibrated probabilities.

Two-step command-line use with ``refine``::

    dopplertrack discover spec.npz fit.json --traces traces.npz \\
        --events events.json --out discovered.json --confirmation confirmation.json
    dopplertrack refine spec.npz fit.json --events events.json --extra discovered.json \\
        --traces traces.npz --method projected --source-degree 1 --smooth spline \\
        --elevation 0 --out joint.json

``discover --refit`` performs the joint refit and the second discovery pass in one step.
"""

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .advanced import ModelOptions, fit_projected, smooth_trajectory
from .detection import Event
from .geo import LocalFrame
from .physics import Atmosphere, Trajectory, closest_approaches, predict


def _null_shifts():
    steps = np.arange(20.0, 121.0, 10.0)
    return tuple(np.r_[-steps[::-1], steps].tolist())


@dataclass(frozen=True)
class DiscoveryOptions:
    """Pass enumeration, scan and acceptance settings.

    The slant-range limit for a pass is ``min(max_distance, ground_range +
    height)``, with ``height`` the aircraft height above the observer at closest
    approach.  Near-grazing paths suffer ground absorption and refraction shadow
    zones, while sound from high aircraft arrives on a direct path: a PC-12 at
    5.5 km altitude was audible at 6-9 km slant range in the Oyster Bay clip.
    """

    band: tuple = (60.0, 300.0)
    harmonics: int = 2
    max_distance: float = 10000.0
    ground_range: float = 6000.0
    half_window: float | None = None
    window_factor: float = 1.5
    min_half_window: float = 30.0
    max_half_window: float = 75.0
    min_sweep: float = 0.03
    min_coverage: float = 0.5
    f0_step: float = 0.1
    drift_radius: float = 0.0
    drift_steps: int = 1
    background_hz: float = 3.0
    null_shifts: tuple = field(default_factory=_null_shifts)
    mirror_shifts: tuple = (0.0, -60.0, -30.0, 30.0, 60.0)
    min_nulls: int = 8
    threshold: float = 8.0
    shadow_hz: float = 1.0
    smoothing: str = "spline"

    def __post_init__(self):
        if not (0 < self.band[0] < self.band[1]) or self.harmonics < 1 or self.f0_step <= 0:
            raise ValueError("band must be increasing and positive; harmonics and f0 step positive")
        if min(self.max_distance, self.ground_range, self.min_half_window, self.window_factor) <= 0:
            raise ValueError("distance and window limits must be positive")
        if self.max_half_window < self.min_half_window or (
            self.half_window is not None and self.half_window <= 0
        ):
            raise ValueError("window limits must be positive and ordered")
        if self.drift_steps < 1 or self.drift_radius < 0 or (self.drift_steps > 1 and self.drift_radius == 0):
            raise ValueError("drift search needs a positive radius when more than one step is used")
        if not 0 < self.min_coverage <= 1 or self.min_sweep < 0 or self.min_nulls < 3:
            raise ValueError("coverage in (0, 1], nonnegative sweep and at least three nulls are required")
        if self.smoothing not in ("none", "spline", "kalman"):
            raise ValueError("smoothing must be none, spline or kalman")
        if any(abs(s) < 1e-9 for s in self.null_shifts):
            raise ValueError("time-shifted nulls must not include zero shift")


def icao(flight):
    """Aircraft identity: trace segment ids are ``icao:start``, refined fits use bare ICAO."""
    return str(flight).split(":")[0].lower()


class RidgeMap:
    """Local narrow-band contrast on the spectrogram grid, computed once per spectrogram."""

    def __init__(self, spec, background_hz=3.0):
        self.spec = spec
        self.times = np.asarray(spec.times, dtype=float)
        self.frequencies = np.asarray(spec.frequencies, dtype=float)
        self.background_hz = float(background_hz)
        power = np.asarray(spec.power, dtype=np.float32)
        background = np.zeros_like(power)
        for sign in (-1.0, 1.0):
            position = np.interp(
                self.frequencies + sign * background_hz, self.frequencies, np.arange(len(self.frequencies))
            )
            low = np.minimum(np.floor(position).astype(int), len(self.frequencies) - 2)
            fraction = (position - low).astype(np.float32)[:, None]
            background += 0.5 * (power[low] * (1 - fraction) + power[low + 1] * fraction)
        contrast = np.maximum(power - background, 0)
        edge = (self.frequencies - background_hz < self.frequencies[0]) | (
            self.frequencies + background_hz > self.frequencies[-1]
        )
        contrast[edge] = 0
        self.contrast = contrast
        self.hop = float(np.median(np.diff(self.times)))

    def copy(self):
        clone = object.__new__(RidgeMap)
        clone.__dict__.update(self.__dict__)
        clone.contrast = self.contrast.copy()
        return clone

    def mask(self, frames, frequency, harmonics, width):
        """Remove an explained ridge (all harmonics, +/- ``width`` Hz) from this map in place."""
        for frame, hz in zip(frames, frequency, strict=True):
            if not np.isfinite(hz):
                continue
            for harmonic in range(1, harmonics + 1):
                low, high = np.searchsorted(self.frequencies, [harmonic * hz - width, harmonic * hz + width])
                self.contrast[low:high, frame] = 0

    def frames(self, start, end):
        return np.flatnonzero((self.times >= start) & (self.times <= end))

    def shifted(self, frames, shift):
        """Nearest frame indices ``shift`` seconds later; -1 outside the recording."""
        target = self.times[frames] + shift
        index = np.clip(np.searchsorted(self.times, target), 1, len(self.times) - 1)
        index -= (target - self.times[index - 1]) < (self.times[index] - target)
        inside = (target >= self.times[0] - self.hop / 2) & (target <= self.times[-1] + self.hop / 2)
        return np.where(inside, index, -1)

    def scores(self, frames, frequency, harmonics):
        """Mean harmonic-weighted contrast; ``frequency`` is (..., frames) in Hz for harmonic one."""
        frequency = np.asarray(frequency, dtype=float)
        low_limit = self.frequencies[0] + self.background_hz
        high_limit = self.frequencies[-1] - self.background_hz
        total = np.zeros(frequency.shape)
        weight = np.zeros(frequency.shape)
        axis = np.arange(len(self.frequencies))
        for harmonic in range(1, harmonics + 1):
            hz = frequency * harmonic
            valid = np.isfinite(hz) & (hz >= low_limit) & (hz <= high_limit)
            if not valid.any():
                break
            position = np.interp(np.where(valid, hz, low_limit), self.frequencies, axis)
            low = np.minimum(np.floor(position).astype(int), len(axis) - 2)
            fraction = position - low
            values = self.contrast[low, frames] * (1 - fraction) + self.contrast[low + 1, frames] * fraction
            w = 1 / np.sqrt(harmonic)
            total += w * np.where(valid, values, 0.0)
            weight += w * valid
        per_frame = np.divide(total, weight, out=np.zeros_like(total), where=weight > 0)
        return per_frame.mean(axis=-1)


def _ridge_map(spec, options):
    if isinstance(spec, RidgeMap):
        return spec
    return RidgeMap(spec, options.background_hz)


def pass_limit(height, options=DiscoveryOptions()):
    """Altitude-aware maximum slant distance for a pass, in metres."""
    return float(min(options.max_distance, options.ground_range + max(0.0, float(height))))


class TraceIndex:
    """Raw trace arrays (the ``save_tracks`` format) with lazy trajectory construction.

    Building every interpolating ``Trajectory`` of a full-day regional archive
    costs gigabytes; discovery only needs the few segments that are inside the
    recording and come within range of the candidate.
    """

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.frame = LocalFrame(*z["frame"])
            self.metadata = json.loads(str(z["metadata"]))
            self.times = [z[f"t{i}"] for i in range(len(self.metadata))]
            self.positions = [z[f"p{i}"] for i in range(len(self.metadata))]
        if not self.metadata:
            raise ValueError("trace file contains no flight segments")
        self.ids = {item["id"]: i for i, item in enumerate(self.metadata)}
        self.cache = {}

    def __len__(self):
        return len(self.metadata)

    def track(self, index):
        if isinstance(index, str):
            index = self.ids[index]
        if index not in self.cache:
            item = self.metadata[index]
            self.cache[index] = Trajectory(
                item["id"], self.times[index], self.positions[index], item.get("max_gap", 30.0), item
            )
        return self.cache[index]

    def select(self, candidate, recording, atmosphere=Atmosphere(), options=None):
        """Segments overlapping the recording in time whose samples come within range."""
        options = DiscoveryOptions() if options is None else options
        observer = np.asarray(candidate["position"], dtype=float)
        clock = float(candidate["clock"])
        delay = options.max_distance / (atmosphere.c - np.linalg.norm(atmosphere.wind))
        start, end = clock + recording[0] - delay - 1, clock + recording[1] + 1
        chosen = []
        for i, (times, positions) in enumerate(zip(self.times, self.positions, strict=True)):
            if len(times) < 2 or times[-1] < start or times[0] > end:
                continue
            inside = (times >= start - 60) & (times <= end + 60)
            if inside.sum() < 2:
                continue
            local = positions[inside]
            # Half the largest sample step bounds how far the interpolated path can come closer.
            slack = 0.5 * float(np.max(np.linalg.norm(np.diff(local, axis=0), axis=1), initial=0.0))
            if float(np.min(np.linalg.norm(local - observer, axis=1))) - slack <= options.max_distance:
                chosen.append(self.track(i))
        return chosen


def _candidate_tracks(tracks, candidate, recording, atmosphere, options):
    if isinstance(tracks, TraceIndex):
        return tracks.select(candidate, recording, atmosphere, options)
    return tracks


def enumerate_passes(tracks, candidate, recording, atmosphere=Atmosphere(), options=DiscoveryOptions()):
    """All closest approaches heard inside ``recording`` = (start, end) recording seconds.

    Returns JSON-friendly dicts (plus the ``track`` object) sorted by heard time.
    """
    observer = np.asarray(candidate["position"], dtype=float)
    clock = float(candidate["clock"])
    start, end = map(float, recording)
    delay = options.max_distance / (atmosphere.c - np.linalg.norm(atmosphere.wind))
    passes = []
    for track in tracks:
        if track.times[-1] < clock + start - delay - 1 or track.times[0] > clock + end + 1:
            continue
        # Gaps split a pass into segments whose endpoints are also local minima;
        # keep only the closest approach within one maximum half-window.
        hits = []
        for heard, distance, speed in sorted(
            closest_approaches(track, observer, atmosphere), key=lambda h: h[1]
        ):
            if all(abs(heard - other[0]) > options.max_half_window for other in hits):
                hits.append((heard, distance, speed))
        for heard, distance, speed in hits:
            heard_recording = heard - clock
            if not start <= heard_recording <= end or distance > options.max_distance:
                continue
            emission = heard - distance / atmosphere.c
            position, _ = track.at(np.clip(emission, track.times[0], track.times[-1]))
            if not np.isfinite(position).all():
                position = track.positions[int(np.argmin(np.abs(track.times - emission)))]
            height = float(position[2] - observer[2])
            limit = pass_limit(height, options)
            if distance > limit:
                continue
            passes.append(
                {
                    "track": track,
                    "flight": track.id,
                    "icao": icao(track.id),
                    "type": track.metadata.get("type"),
                    "registration": track.metadata.get("registration"),
                    "heard_s": float(heard_recording),
                    "distance_m": float(distance),
                    "speed_m_s": float(speed),
                    "altitude_m": float(position[2]),
                    "height_above_observer_m": height,
                    "distance_limit_m": limit,
                }
            )
    return sorted(passes, key=lambda p: (p["heard_s"], p["flight"]))


def _half_window(item, options):
    if options.half_window is not None:
        return float(options.half_window)
    natural = options.window_factor * item["distance_m"] / max(item["speed_m_s"], 1.0)
    return float(np.clip(natural, options.min_half_window, options.max_half_window))


def _best(ridges, frames, factor, emission, f0s, drifts, harmonics):
    """Maximum over source frequency and drift; returns (score, f0, drift)."""
    best = (-np.inf, np.nan, 0.0)
    for drift in drifts:
        source = f0s[:, None] + drift * emission[None, :]
        values = ridges.scores(frames, source * factor[None, :], harmonics)
        index = int(np.argmax(values))
        if values[index] > best[0]:
            best = (float(values[index]), float(f0s[index]), float(drift))
    return best


def scan_pass(spec, item, candidate, atmosphere=Atmosphere(), options=DiscoveryOptions()):
    """Score one enumerated pass against time-shifted and time-mirrored null curves."""
    ridges = _ridge_map(spec, options)
    track = item["track"]
    observer = np.asarray(candidate["position"], dtype=float)
    clock = float(candidate["clock"])
    cpa = item["heard_s"]
    half = _half_window(item, options)
    window = (max(cpa - half, float(ridges.times[0])), min(cpa + half, float(ridges.times[-1])))
    row = {
        **{k: v for k, v in item.items() if k != "track"},
        "window": [float(window[0]), float(window[1])],
        "status": "ok",
        "z": None,
        "score": None,
        "f0": None,
        "drift_hz_s": None,
        "accepted": False,
    }
    frames = ridges.frames(*window)
    nominal = max(1, int(round(2 * half / ridges.hop)) + 1)
    if len(frames) < max(10, options.min_coverage * nominal):
        row["status"] = "outside_recording"
        return row
    times = ridges.times[frames]
    if options.smoothing != "none":
        # Smooth only the trimmed neighbourhood of the pass, as the refit would.
        margin = options.max_half_window + 120
        local = trim_track(track, clock + window[0] - margin, clock + window[1] + margin)
        track = smooth_trajectory(local, options.smoothing)
    try:
        factor, emission = predict(track, observer, clock + times, 1.0, atmosphere)
        mirror, mirror_emission = predict(track, observer, clock + 2 * cpa - times, 1.0, atmosphere)
    except ValueError:
        row["status"] = "prediction_error"
        return row
    covered = np.isfinite(factor)
    row["coverage"] = float(covered.mean())
    if row["coverage"] < options.min_coverage:
        row["status"] = "track_coverage"
        return row
    sweep = float((np.nanmax(factor) - np.nanmin(factor)) / np.nanmedian(factor))
    row["sweep"] = sweep
    if sweep < options.min_sweep:
        row["status"] = "flat_curve"
        return row
    reference = clock + cpa - item["distance_m"] / atmosphere.c
    f0s = np.arange(options.band[0], options.band[1] + options.f0_step / 2, options.f0_step)
    drifts = (
        np.linspace(-options.drift_radius, options.drift_radius, options.drift_steps)
        if options.drift_steps > 1
        else np.zeros(1)
    )

    def evaluate(curve, emitted, shift):
        use = np.isfinite(curve)
        shifted = ridges.shifted(frames, shift) if shift else frames
        use &= shifted >= 0
        if use.mean() < options.min_coverage:
            return None
        return _best(
            ridges, shifted[use], curve[use], emitted[use] - reference, f0s, drifts, options.harmonics
        )

    score, f0, drift = evaluate(factor, emission, 0.0)
    nulls = [evaluate(factor, emission, s) for s in options.null_shifts]
    nulls += [evaluate(mirror, mirror_emission, s) for s in options.mirror_shifts]
    null_scores = np.array([n[0] for n in nulls if n is not None])
    row["nulls"] = int(len(null_scores))
    if len(null_scores) < options.min_nulls:
        row["status"] = "too_few_nulls"
        return row
    center = float(np.median(null_scores))
    spread = 1.4826 * float(np.median(np.abs(null_scores - center)))
    scale = max(spread, 0.25 * float(np.std(null_scores)), 1e-9)
    z = (score - center) / scale
    # The explained ridge extends beyond the scan window; mask it wherever the track predicts it.
    span = ridges.frames(window[0] - options.max_half_window, window[1] + options.max_half_window)
    wide, wide_emission = predict(track, observer, clock + ridges.times[span], 1.0, atmosphere)
    fitted = (f0 + drift * (wide_emission - reference)) * wide
    row.update(
        z=float(z),
        score=score,
        null_median=center,
        null_scale=scale,
        null_max=float(null_scores.max()),
        f0=f0,
        drift_hz_s=drift,
        accepted=bool(z >= options.threshold),
        _frames=span,
        _fitted=fitted,
    )
    return row


def _public(row):
    return {k: v for k, v in row.items() if not k.startswith("_")}


def trim_track(track, start, end):
    """Copy of ``track`` limited to UTC [start, end], keeping metadata (as in the Oyster Bay protocol)."""
    mask = (track.times >= start) & (track.times <= end)
    if mask.sum() < 3:
        return track
    return Trajectory(track.id, track.times[mask], track.positions[mask], track.max_gap, track.metadata)


def discover(
    spec,
    tracks,
    candidate,
    atmosphere=Atmosphere(),
    options=DiscoveryOptions(),
    used_flights=None,
    progress=None,
):
    """Enumerate, scan and accept passes at a frozen candidate.

    ``tracks`` is a list of trajectories or a ``TraceIndex`` (pre-filtered per candidate).
    ``used_flights`` (default: ``candidate["flights"]``) are the flights that fitted
    the candidate; they are scanned and reported but never count as independent
    confirmation.  Returns a JSON-safe report plus non-serialised ``events`` and
    ``tracks`` lists (accepted, unshadowed, independent passes) for ``fit_projected``.
    """
    ridges = _ridge_map(spec, options)
    used = {icao(f) for f in (candidate.get("flights", []) if used_flights is None else used_flights)}
    recording = (float(ridges.times[0]), float(ridges.times[-1]))
    tracks = _candidate_tracks(tracks, candidate, recording, atmosphere, options)
    passes = enumerate_passes(tracks, candidate, recording, atmosphere, options)
    rows = []
    for i, item in enumerate(passes):
        row = scan_pass(ridges, item, candidate, atmosphere, options)
        row["used_in_fit"] = item["icao"] in used
        row["_track"], row["_item"] = item["track"], item
        rows.append(row)
        if progress:
            progress(i + 1, len(passes))
    # Explaining away: strongest accepted ridge first; weaker passes are rescanned
    # after removing every kept ridge, so a silent track crossing a real ridge
    # cannot borrow its energy.
    kept, masked = [], None
    for row in sorted((r for r in rows if r["accepted"]), key=lambda r: r["z"], reverse=True):
        row["z_unmasked"] = row["z"]
        row["shadowed_by"] = None
        if masked is not None:
            rescored = scan_pass(masked, row["_item"], candidate, atmosphere, options)
            for key in ("z", "score", "null_median", "null_scale", "null_max", "f0", "drift_hz_s"):
                row[key] = rescored.get(key)
            row["_frames"], row["_fitted"] = rescored.get("_frames"), rescored.get("_fitted")
            if not rescored["accepted"]:
                row["accepted"] = False
                row["shadowed_by"] = [
                    k["flight"]
                    for k in kept
                    if k["window"][0] < row["window"][1] and row["window"][0] < k["window"][1]
                ]
                continue
        kept.append(row)
        masked = ridges.copy() if masked is None else masked
        masked.mask(row["_frames"], row["_fitted"], options.harmonics, options.shadow_hz)
    for row in rows:
        row["confirmed"] = bool(row["accepted"])
    independent, seen = [], set()
    for row in sorted(kept, key=lambda r: r["z"], reverse=True):
        if row["used_in_fit"] or row["icao"] in seen:
            continue
        seen.add(row["icao"])
        independent.append(row)
    independent.sort(key=lambda r: r["heard_s"])
    events, chosen = [], []
    clock = float(candidate["clock"])
    for row in independent:
        events.append(
            Event(
                tuple(row["window"]),
                row["heard_s"],
                row["f0"],
                row["speed_m_s"],
                row["distance_m"],
                options.harmonics,
                float(row["z"]),
            )
        )
        chosen.append(
            trim_track(row["_track"], clock + row["window"][0] - 240, clock + row["window"][1] + 240)
        )
    summary = {
        "position": [float(v) for v in candidate["position"]],
        "clock": clock,
        "used_flights": sorted(used),
        "passes_enumerated": len(rows),
        "passes_scanned": sum(r["z"] is not None for r in rows),
        "accepted": sum(r["accepted"] for r in rows),
        "confirmed_independent": len(independent),
        "confirmed_flights": [r["icao"] for r in independent],
        "z_sum": float(sum(r["z"] for r in independent)),
        "used_flights_confirmed": sorted({r["icao"] for r in kept if r["used_in_fit"]}),
        "options": asdict(options),
        "passes": [_public(r) for r in rows],
        "note": "Empirical time-shift/mirror nulls; z is a heuristic, not a calibrated probability.",
    }
    return {"summary": summary, "events": events, "tracks": chosen}


def confirmation(
    spec, tracks, candidate, atmosphere=Atmosphere(), options=DiscoveryOptions(), used_flights=None
):
    """Compact confirmation summary: count of independent confirmed passes, sum of z, and the list."""
    summary = discover(spec, tracks, candidate, atmosphere, options, used_flights)["summary"]
    listed = [
        {k: p[k] for k in ("flight", "icao", "type", "heard_s", "distance_m", "altitude_m", "z", "f0")}
        for p in summary["passes"]
        if p["confirmed"] and not p["used_in_fit"]
    ]
    return {
        "confirmed_independent": summary["confirmed_independent"],
        "z_sum": summary["z_sum"],
        "confirmed": listed,
        "passes_scanned": summary["passes_scanned"],
    }


def joint_refit(
    spec,
    events,
    tracks,
    candidate,
    discovered,
    model=ModelOptions(degree=1),
    atmosphere=Atmosphere(),
    frame=None,
    elevation=None,
):
    """Fit position and the shared clock with the original plus discovered events.

    ``discovered`` is the output of ``discover``.  If a discovered event yields too
    few ridge samples for the fit, the weakest discovered event is dropped and the
    fit is retried; original events are never dropped.
    """
    extra = list(zip(discovered["events"], discovered["tracks"], strict=True))
    used = {icao(t.id) for t in tracks}
    extra = [(e, t) for e, t in extra if icao(t.id) not in used]
    dropped = []
    while True:
        combined = sorted(
            [*zip(events, tracks, candidate["f0"], strict=True), *((e, t, e.f0) for e, t in extra)],
            key=lambda item: item[0].cpa,
        )
        all_events, all_tracks, f0 = map(list, zip(*combined, strict=True))
        initial = {**candidate, "f0": f0, "flights": [t.id for t in all_tracks]}
        try:
            result = fit_projected(
                spec, all_events, all_tracks, initial, model, atmosphere, frame=frame, elevation=elevation
            )
            break
        except ValueError:
            if not extra:
                raise
            weakest = min(range(len(extra)), key=lambda i: extra[i][0].score)
            dropped.append(extra.pop(weakest)[1].id)
    result["discovered_flights"] = [t.id for _, t in extra]
    result["dropped_discoveries"] = dropped
    return result, all_events, all_tracks


def discover_and_refit(
    spec,
    events,
    tracks,
    all_tracks,
    candidate,
    model=ModelOptions(degree=1),
    atmosphere=Atmosphere(),
    options=DiscoveryOptions(),
    frame=None,
    elevation=None,
    progress=None,
    ridges=None,
):
    """Discover at ``candidate``, jointly refit, then discover once more at the refined solution.

    ``events``/``tracks`` are the fitted pairs behind ``candidate``; ``all_tracks``
    are every ADS-B segment to search.  The second discovery treats every fitted
    flight as used, so its confirmation count only includes flights outside the fit.
    """
    ridges = _ridge_map(spec, options) if ridges is None else ridges
    first = discover(ridges, all_tracks, candidate, atmosphere, options, [t.id for t in tracks], progress)
    if first["events"]:
        result, joint_events, joint_tracks = joint_refit(
            spec, events, tracks, candidate, first, model, atmosphere, frame, elevation
        )
    else:
        result, joint_events, joint_tracks = dict(candidate), list(events), list(tracks)
        result["discovered_flights"], result["dropped_discoveries"] = [], []
    second = discover(ridges, all_tracks, result, atmosphere, options, [t.id for t in joint_tracks], progress)
    return {
        "result": result,
        "events": joint_events,
        "tracks": joint_tracks,
        "first": first["summary"],
        "second": second["summary"],
        "second_events": second["events"],
        "second_tracks": second["tracks"],
    }
