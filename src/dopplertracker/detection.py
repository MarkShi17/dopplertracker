"""Ideal-pass fitting seeds coarse ranges; actual flight geometry determines ranking.

Two detectors live here.  ``detect``/``fit_event`` are the original brightness
fits on sliding windows and remain unchanged so documented numbers reproduce.
``detect_flyovers``/``fit_flyover`` are the calibrated detector: sideband
contrast with flat portions down-weighted, a minimum sweep, harmonic
consistency, a log-frequency matched-filter bank over the whole recording,
curve-overlap suppression, local refinement, and a z-score against
time-reversed and frequency-shifted null curves so that it can abstain.
"""

import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy.ndimage import maximum_filter, median_filter
from scipy.optimize import differential_evolution, minimize

from .physics import ideal_curve
from .scoring import ridge_contrast, score_curve, slope_weights
from .signal import Spectrogram


@dataclass
class Event:
    window: tuple
    cpa: float
    f0: float
    speed: float
    distance: float
    harmonics: int = 1
    score: float = 0.0
    speed_range: tuple | None = None
    distance_range: tuple | None = None
    # Set by the calibrated detector only; None for published or legacy events.
    z: float | None = None
    cpa_range: tuple | None = None
    f0_range: tuple | None = None

    def __post_init__(self):
        if (
            len(self.window) != 2
            or not np.isfinite(self.window).all()
            or not self.window[0] < self.window[1]
            or not self.window[0] <= self.cpa <= self.window[1]
            or not np.isfinite([self.cpa, self.f0, self.speed, self.distance]).all()
            or min(self.f0, self.speed, self.distance) <= 0
            or self.harmonics < 1
        ):
            raise ValueError("invalid event window or curve parameters")
        if self.speed_range is None:
            self.speed_range = (max(10.0, self.speed * 0.65), self.speed * 1.4)
        if self.distance_range is None:
            self.distance_range = (max(100.0, self.distance * 0.45), self.distance * 1.8)
        for limits in (self.speed_range, self.distance_range):
            if len(limits) != 2 or not np.isfinite(limits).all() or not 0 <= limits[0] < limits[1]:
                raise ValueError("event filter ranges must be finite increasing pairs")
        if self.z is not None and not np.isfinite(self.z):
            raise ValueError("event z must be finite")
        for value, limits in ((self.cpa, self.cpa_range), (self.f0, self.f0_range)):
            if limits is not None and (
                len(limits) != 2 or not np.isfinite(limits).all() or not limits[0] <= value <= limits[1]
            ):
                raise ValueError("event CPA and f0 ranges must be finite pairs containing the estimate")

    def sample_times(self, spec, maximum=200):
        t = spec.times[(spec.times >= self.window[0]) & (spec.times <= self.window[1])]
        if len(t) < 5:
            raise ValueError("event window must cover at least five spectrogram frames")
        return t[:: max(1, int(np.ceil(len(t) / maximum)))]


def fit_event(spec, window, band=(60, 300), harmonics=1, seed=0, iterations=60):
    times = spec.times[(spec.times >= window[0]) & (spec.times <= window[1])]
    if len(times) < 5 or band[0] >= band[1]:
        raise ValueError("fit window needs at least five frames and an increasing frequency band")
    times = times[:: max(1, len(times) // 160)]
    bounds = [(band[0], band[1]), (30.0, 180.0), (300.0, 12000.0), (float(times[0]), float(times[-1]))]

    def objective(p):
        return -score_curve(spec, times, ideal_curve(times, *p), "brightness", harmonics)

    result = differential_evolution(
        objective, bounds, seed=seed, maxiter=iterations, popsize=12, polish=False, tol=0.005
    )
    polished = minimize(
        objective, result.x, method="Nelder-Mead", bounds=bounds, options={"maxiter": 500, "xatol": 0.001}
    )
    p = polished.x if polished.fun < result.fun else result.x
    return Event(
        tuple(window), float(p[3]), float(p[0]), float(p[1]), float(p[2]), harmonics, float(-objective(p))
    )


def detect(spec, band=(60, 300), window=120.0, stride=60.0, top=10, harmonics=1, progress=None):
    if window <= 0 or stride <= 0 or top < 1:
        raise ValueError("window, stride and top must be positive")
    events = []
    starts = np.arange(spec.times[0], spec.times[-1] - window + 1e-8, stride)
    for i, start in enumerate(starts):
        event = fit_event(spec, (float(start), float(start + window)), band, harmonics, seed=i, iterations=30)
        if event.score > 0:
            events.append(event)
        if progress:
            progress(i + 1, len(starts))
    kept = []
    for event in sorted(events, key=lambda e: e.score, reverse=True):
        if all(abs(event.cpa - other.cpa) > window * 0.4 for other in kept):
            kept.append(event)
        if len(kept) >= top:
            break
    return kept


@dataclass(frozen=True)
class FlyoverOptions:
    """Settings of the calibrated detector; frequencies are relative unless in Hz.

    ``span`` seconds of curve centred on the heard closest approach are scored.
    ``slope_scale`` is the relative sweep rate (1/s) at which a sample carries
    full weight; ``min_sweep`` is the least ``(max - min) / f0`` inside the span.
    ``tone_seconds`` subtracts a per-frequency running median of that length, so
    tones lasting longer are suppressed (0 disables).  Detections need
    ``z >= min_z`` and at most ``max_fragment`` of their evidence inside any
    fifth of the span (1 disables).  Speed and distance ranges are the profile
    region within ``range_sigma`` null deviations of the best score, widened by
    the relative model-error margins.  ``candidates`` coarse peaks are kept
    once a proposal's slope-weighted curve overlaps a stronger one by less than
    ``overlap``; each is refined from itself and its ``starts`` best nearby
    bank shapes, and a refined curve is dropped when more than ``shared`` of its
    evidence lies on a stronger detection's ridge or harmonic.  Candidates are
    refined in order of coarse score until ``patience`` in a row fail the
    acceptance tests (0 refines all).
    """

    band: tuple = (60.0, 300.0)
    harmonics: int = 2
    span: float = 120.0
    sidebands: tuple = (0.03, 0.06)
    clip: tuple = (-1.0, 4.0)
    slope_scale: float = 0.002
    min_sweep: float = 0.08
    tone_seconds: float = 60.0
    speeds: tuple = (45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 140.0, 160.0)
    time_scales: tuple = (5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 56.0, 80.0)
    speed_bounds: tuple = (30.0, 180.0)
    distance_bounds: tuple = (300.0, 12000.0)
    log_step: float = 0.002
    time_step: float = 1.0
    candidates: int = 24
    starts: int = 3
    patience: int = 8
    overlap: float = 0.5
    shared: float = 0.5
    min_z: float = 12.0
    max_fragment: float = 0.7
    range_sigma: float = 2.0
    speed_margin: float = 0.15
    distance_margin: float = 0.3

    def __post_init__(self):
        values = [
            *self.band,
            *self.sidebands,
            *self.speeds,
            *self.time_scales,
            *self.speed_bounds,
            *self.distance_bounds,
            self.span,
            self.slope_scale,
            self.log_step,
            self.time_step,
        ]
        if (
            not np.isfinite(values).all()
            or min(values) <= 0
            or not self.band[0] < self.band[1]
            or not self.speed_bounds[0] < self.speed_bounds[1] < 343.0
            or not self.distance_bounds[0] < self.distance_bounds[1]
            or not self.clip[0] < self.clip[1]
            or self.harmonics < 1
            or self.candidates < 1
            or not 0 <= self.min_sweep < 1
            or self.starts < 0
            or self.patience < 0
            or not 0 < self.overlap <= 1
            or not 0 < self.shared <= 1
            or not 0 < self.max_fragment <= 1
            or self.tone_seconds < 0
            or min(self.range_sigma, self.speed_margin, self.distance_margin) < 0
        ):
            raise ValueError("invalid calibrated detector options")


@dataclass
class _Pass:
    """One candidate curve during the calibrated search (internal)."""

    f0: float
    speed: float
    distance: float
    cpa: float
    score: float = 0.0
    extras: dict = field(default_factory=dict)
    evidence: np.ndarray | None = None

    @property
    def params(self):
        return np.array([self.f0, self.speed, self.distance, self.cpa])


def suppress_tones(spec, seconds=60.0, step=1.0):
    """Subtract each frequency's running median over ``seconds`` of time.

    Whitening removes only tones present for most of the recording; this removes
    those lasting longer than about half of ``seconds`` while a Doppler sweep,
    which crosses a frequency bin quickly, is kept.  The median is computed on
    ``step``-second pooled frames and interpolated back to the original frames.
    """
    if seconds <= 0:
        return spec
    power = np.asarray(spec.power, dtype=np.float32)
    k = max(1, int(round(step / float(np.median(np.diff(spec.times))))))
    n = len(spec.times) // k
    if n < 3:
        return spec
    pooled = power[:, : n * k].reshape(len(power), n, k).mean(axis=2)
    pooled_times = spec.times[: n * k].reshape(n, k).mean(axis=1)
    size = max(3, int(round(seconds / (k * float(np.median(np.diff(spec.times)))))) | 1)
    background = median_filter(pooled, size=(1, size), mode="nearest")
    index = np.interp(spec.times, pooled_times, np.arange(n))
    low = np.clip(np.floor(index).astype(int), 0, n - 1)
    high = np.minimum(low + 1, n - 1)
    fraction = (index - low).astype(np.float32)
    interpolated = background[:, low] * (1 - fraction) + background[:, high] * fraction
    return Spectrogram(spec.times, spec.frequencies, power - interpolated)


def _log_map(spec, options):
    """Pool to ``time_step`` seconds and resample onto a log-frequency grid.

    On a log axis a source frequency is a shift, so one sum along a normalised
    curve scores every f0 at once.  Three sub-samples per log bin avoid aliasing
    narrow ridges where log bins are wider than linear ones.
    """
    k = max(1, int(round(options.time_step / float(np.median(np.diff(spec.times))))))
    n = len(spec.times) // k
    power = np.asarray(spec.power, dtype=np.float32)[:, : n * k].reshape(len(spec.frequencies), n, k)
    power = power.mean(axis=2)
    times = spec.times[: n * k].reshape(n, k).mean(axis=1)
    frequencies = spec.frequencies
    grid = np.arange(np.log(frequencies[0]), np.log(frequencies[-1]), options.log_step)
    values = np.zeros((len(grid), n), dtype=np.float32)
    for offset in (-1 / 3, 0.0, 1 / 3):
        index = np.interp(np.exp(grid + offset * options.log_step), frequencies, np.arange(len(frequencies)))
        low = np.clip(np.floor(index).astype(int), 0, len(frequencies) - 2)
        fraction = (index - low).astype(np.float32)[:, None]
        values += ((1 - fraction) * power[low] + fraction * power[low + 1]) / 3
    return times, grid, values


def _contrast_map(values, options):
    """Clipped log-domain contrast, then 1/sqrt(h) harmonic stacking at +ln(h)."""
    offsets = [max(1, int(round(np.log1p(s) / options.log_step))) for s in options.sidebands]
    pad = max(offsets)
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    rows = len(values)
    background = sum(padded[pad - o : pad - o + rows] + padded[pad + o : pad + o + rows] for o in offsets) / (
        2 * len(offsets)
    )
    contrast = np.clip(values - background, *options.clip).astype(np.float32)
    stacked = np.zeros_like(contrast)
    weight = np.zeros((rows, 1), dtype=np.float32)
    for h in range(1, options.harmonics + 1):
        shift = int(round(np.log(h) / options.log_step))
        if shift >= rows:
            break
        stacked[: rows - shift] += contrast[shift:] / np.sqrt(h)
        weight[: rows - shift] += 1 / np.sqrt(h)
    return stacked / np.maximum(weight, 1e-9)


def _shape(speed, distance, options):
    """One normalised ideal pass as integer (time, log-frequency) steps, or None if too flat."""
    if not (
        options.distance_bounds[0] <= distance <= options.distance_bounds[1]
        and options.speed_bounds[0] <= speed <= options.speed_bounds[1]
    ):
        return None
    tau = np.arange(-options.span / 2, options.span / 2 + 1e-9, options.time_step)
    curve = ideal_curve(tau, 1.0, speed, distance, 0.0)
    if curve[0] - curve[-1] < options.min_sweep:
        return None
    weights = slope_weights(tau, curve, 1.0, options.slope_scale)
    offsets = np.round(np.log(curve) / options.log_step).astype(int)
    steps = np.round(tau / options.time_step).astype(int)
    keep = weights > 0
    return float(speed), float(distance), steps[keep], offsets[keep], weights[keep]


def _shape_bank(options):
    """Global bank: every speed x time scale (distance / speed) that sweeps enough."""
    bank = [_shape(v, v * s, options) for v in options.speeds for s in options.time_scales]
    bank = [shape for shape in bank if shape is not None]
    if not bank:
        raise ValueError("no ideal pass in the shape bank satisfies the minimum sweep")
    return bank


def _local_bank(speed, distance, options):
    """Dense bank around one coarse shape: real ridges are narrow and sweeps are wide."""
    scale = distance / speed
    bank = [
        _shape(v, v * s, options)
        for v in speed * np.geomspace(0.75, 1 / 0.75, 11)
        for s in scale * np.geomspace(0.6, 1 / 0.6, 17)
    ]
    return [shape for shape in bank if shape is not None]


@dataclass
class _BankMap:
    """Zero-padded harmonic contrast map plus the shape bank that is summed over it."""

    times: np.ndarray
    grid: np.ndarray
    padded: np.ndarray
    bank: list
    first: int
    last: int
    row_pad: int
    time_pad: int
    samples: int

    @classmethod
    def build(cls, spec, options):
        times, grid, values = _log_map(spec, options)
        bank = _shape_bank(options)
        first = int(np.searchsorted(grid, np.log(options.band[0])))
        last = int(np.searchsorted(grid, np.log(options.band[1]), side="right"))
        if last <= first:
            raise ValueError("detection band lies outside the spectrogram frequencies")
        doppler = max(-np.log1p(-options.speed_bounds[1] / 343.0), np.log1p(options.speed_bounds[1] / 343.0))
        row_pad = int(np.ceil(doppler / options.log_step)) + 64
        time_pad = int(np.ceil(options.span / 2 / options.time_step)) + 64
        padded = np.pad(_contrast_map(values, options), ((row_pad, row_pad), (time_pad, time_pad)))
        samples = len(np.arange(-options.span / 2, options.span / 2 + 1e-9, options.time_step))
        return cls(times, grid, padded, bank, first, last, row_pad, time_pad, samples)

    def response(self, shape, rows, columns):
        """Weighted mean contrast of one bank shape for f0 rows x CPA columns (slices)."""
        _, _, steps, offsets, weights = shape
        total = np.zeros((rows.stop - rows.start, columns.stop - columns.start), dtype=np.float32)
        for step, offset, weight in zip(steps, offsets, weights):
            r = self.row_pad + self.first + offset
            c = self.time_pad + step
            total += (
                np.float32(weight)
                * self.padded[r + rows.start : r + rows.stop, c + columns.start : c + columns.stop]
            )
        return total / self.samples

    def best(self):
        """Best response over the bank for every (f0 row in band, CPA frame)."""
        rows, columns = slice(0, self.last - self.first), slice(0, len(self.times))
        best = np.full((rows.stop, columns.stop), -np.inf, dtype=np.float32)
        best_shape = np.zeros(best.shape, dtype=np.int32)
        for index, shape in enumerate(self.bank):
            total = self.response(shape, rows, columns)
            better = total > best
            best[better] = total[better]
            best_shape[better] = index
        return best, best_shape

    def starts(self, row, column, bank, count, row_radius, column_radius):
        """Parameters of the ``count`` best ``bank`` shapes near (row, column), one per shape."""
        rows = slice(
            max(row - row_radius, -self.first), min(row + row_radius + 1, len(self.grid) - self.first)
        )
        columns = slice(max(column - column_radius, 0), min(column + column_radius + 1, len(self.times)))
        options = []
        for shape in bank:
            speed, distance = shape[:2]
            total = self.response(shape, rows, columns)
            r, c = np.unravel_index(int(np.argmax(total)), total.shape)
            f0 = float(np.exp(self.grid[self.first + rows.start + r]))
            options.append((float(total[r, c]), f0, speed, distance, float(self.times[columns.start + c])))
        options.sort(key=lambda item: -item[0])
        return [np.array(item[1:]) for item in options[:count]]


def _curve(times, p, reverse=False):
    f0, speed, distance, cpa = p
    if reverse:
        return ideal_curve(2 * cpa - np.asarray(times), f0, speed, distance, cpa)
    return ideal_curve(times, f0, speed, distance, cpa)


def _tau(spec, options):
    """Curve sample offsets from CPA; STFT frames overlap, so finer than time_step adds little."""
    dt = max(float(np.median(np.diff(spec.times))), options.time_step)
    return np.arange(-options.span / 2, options.span / 2 + 1e-9, dt)


def _evidence(spec, p, options, tau, reverse=False, scale=1.0):
    """Per-sample slope weights and contrast of one (possibly null) curve."""
    times = p[3] + tau
    frequency = scale * _curve(times, p, reverse)
    weights = slope_weights(times, frequency, scale * p[0], options.slope_scale)
    contrast = ridge_contrast(spec, times, frequency, options.harmonics, options.sidebands, options.clip)
    return weights, contrast


def _score(spec, p, options, tau, reverse=False, scale=1.0):
    weights, contrast = _evidence(spec, p, options, tau, reverse, scale)
    return float(np.mean(weights * contrast))


def _sweep(p, tau):
    curve = _curve(p[3] + tau, p)
    return float((curve[0] - curve[-1]) / p[0])


def _bounds(options, cpa):
    return [
        tuple(options.band),
        tuple(options.speed_bounds),
        tuple(options.distance_bounds),
        (cpa - options.span / 4, cpa + options.span / 4),
    ]


def _maximise(
    spec, start, options, tau, bounds, fixed=None, iterations=600, shrinks=(1.0, 0.5, 0.25, 0.5, 0.25)
):
    """Nelder-Mead on the swept-contrast score; ``fixed`` = (index, value) for profiles."""
    free = [i for i in range(4) if fixed is None or i != fixed[0]]
    start = np.asarray(start, dtype=float)

    def full(x):
        p = start.copy()
        p[free] = x
        if fixed is not None:
            p[fixed[0]] = fixed[1]
        return p

    def objective(x):
        p = full(x)
        if any(not lo <= v <= hi for v, (lo, hi) in zip(p, bounds)) or _sweep(p, tau) < options.min_sweep:
            return 1e3
        return -_score(spec, p, options, tau)

    x0 = np.clip(start[free], [bounds[i][0] for i in free], [bounds[i][1] for i in free])
    relative = np.array([0.01, 0.15, 0.2, 0.0])
    absolute = np.array([0.0, 0.0, 0.0, 2.0])
    best = (objective(x0), x0)
    failures = 0
    # Restarts with shrinking simplices; a narrow ridge stalls a single simplex early.
    for shrink in shrinks:
        x = best[1]
        steps = shrink * (relative * np.abs(full(x)) + absolute)[free]
        simplex = np.vstack([x, x + np.diag(steps)])
        result = minimize(
            objective,
            x,
            method="Nelder-Mead",
            options={"initial_simplex": simplex, "maxiter": iterations, "xatol": 1e-3, "fatol": 1e-6},
        )
        failures = 0 if result.fun < best[0] - 1e-6 else failures + 1
        if result.fun < best[0]:
            best = (float(result.fun), result.x)
        if failures == 2:
            break
    return full(best[1]), -best[0]


def _null_scores(spec, p, options, tau):
    """Time-reversed copies (+/-20 % in f0) and forward copies shifted 8-30 % in f0."""
    reversed_scales = 1 + 0.02 * np.arange(-10, 11)
    shifts = np.arange(0.08, 0.305, 0.02)
    scores = [_score(spec, p, options, tau, reverse=True, scale=s) for s in reversed_scales]
    scores += [_score(spec, p, options, tau, scale=s) for s in np.r_[1 - shifts, 1 + shifts]]
    return np.asarray(scores)


def _calibrate(spec, candidate, options, tau):
    nulls = _null_scores(spec, candidate.params, options, tau)
    center = float(np.median(nulls))
    spread = max(1.4826 * float(np.median(np.abs(nulls - center))), 1e-3)
    weights, contrast = _evidence(spec, candidate.params, options, tau)
    evidence = weights * np.maximum(contrast, 0)
    length = max(1, len(evidence) // 5)
    total = float(evidence.sum())
    best_fifth = float(np.convolve(evidence, np.ones(length), mode="valid").max()) if total > 0 else 0.0
    candidate.extras.update(
        z=(candidate.score - center) / spread,
        null_median=center,
        null_sigma=spread,
        fragment=best_fifth / total if total > 0 else 1.0,
    )
    candidate.evidence = evidence


def _ratios(options):
    return sorted({j / k for j in range(1, options.harmonics + 1) for k in range(1, options.harmonics + 1)})


def _near(times, a, b, options, tolerance=0.03):
    """Samples where curve b lies within ``tolerance`` (relative) of a harmonic of a, inside a's span."""
    fa, fb = _curve(times, a.params), _curve(times, b.params)
    close = np.zeros(len(times), dtype=bool)
    for ratio in _ratios(options):
        close |= np.abs(np.log(fb / (ratio * fa))) < tolerance
    return close & (np.abs(times - a.cpa) <= options.span / 2)


def _curve_overlap(a, b, options):
    """Fraction of b's slope-weighted curve that follows a (or a harmonic alias of a)."""
    times = np.arange(b.cpa - options.span / 2, b.cpa + options.span / 2 + 1e-9, options.time_step)
    weights = slope_weights(times, _curve(times, b.params), b.f0, options.slope_scale)
    total = weights.sum()
    return float((weights * _near(times, a, b, options)).sum() / total) if total > 0 else 1.0


def _shared_evidence(a, b, options, tau):
    """Fraction of b's positive evidence collected where b runs along a's ridge."""
    total = float(b.evidence.sum())
    return float(b.evidence[_near(b.cpa + tau, a, b, options)].sum() / total) if total > 0 else 1.0


def _suppress(candidates, overlap, key, threshold):
    kept = []
    for candidate in sorted(candidates, key=key, reverse=True):
        if all(overlap(other, candidate) < threshold for other in kept):
            kept.append(candidate)
    return kept


def _coarse_candidates(spec, options, limits):
    """Local maxima of the bank response, deduplicated, each with several refinement starts."""
    bank_map = _BankMap.build(spec, options)
    best, shape = bank_map.best()
    times = bank_map.times
    allowed = (times >= limits[0]) & (times <= limits[1])
    neighbourhood = (
        max(3, int(round(0.05 / options.log_step))) | 1,
        max(3, int(round(15 / options.time_step))) | 1,
    )
    peaks = (maximum_filter(best, size=neighbourhood, mode="nearest") == best) & (best > 0) & allowed[None, :]
    rows, columns = np.nonzero(peaks)
    order = np.lexsort((columns, rows, -best[rows, columns]))[: 8 * options.candidates]
    found = []
    for row, column in zip(rows[order], columns[order]):
        speed, distance = bank_map.bank[shape[row, column]][:2]
        f0 = float(np.exp(bank_map.grid[bank_map.first + row]))
        found.append(_Pass(f0, speed, distance, float(times[column]), float(best[row, column])))
        found[-1].extras["cell"] = (int(row), int(column))
    kept = _suppress(
        found,
        lambda a, b: max(_curve_overlap(a, b, options), _curve_overlap(b, a, options)),
        key=lambda c: c.score,
        threshold=options.overlap,
    )[: options.candidates]
    row_radius = max(1, int(round(0.04 / options.log_step)))
    column_radius = max(1, int(round(8 / options.time_step)))
    for candidate in kept:
        row, column = candidate.extras.pop("cell")
        local = _local_bank(candidate.speed, candidate.distance, options)
        starts = bank_map.starts(row, column, local, options.starts, row_radius, column_radius)
        candidate.extras["starts"] = [candidate.params, *starts]
    return kept


def _profile_range(spec, candidate, options, tau, bounds, index, values):
    """Contiguous range of parameter ``index`` whose profile stays within range_sigma null deviations.

    The profile re-optimises the other parameters at each grid value, walking
    outward from the estimate on each side; the edge is interpolated linearly
    between the last value above the floor and the first below it.
    """
    floor = candidate.score - options.range_sigma * candidate.extras["null_sigma"]
    best = float(candidate.params[index])
    values = np.asarray(values, dtype=float)
    limits = []
    for side in (values[values < best][::-1], values[values > best]):
        edge, edge_score = best, candidate.score
        for value in side:
            if not bounds[index][0] <= value <= bounds[index][1]:
                break
            _, score = _maximise(
                spec,
                candidate.params,
                options,
                tau,
                bounds,
                (index, value),
                iterations=150,
                shrinks=(0.5, 0.25),
            )
            if score < floor:
                edge += (value - edge) * (edge_score - floor) / max(edge_score - score, 1e-12)
                break
            edge, edge_score = value, score
        limits.append(float(edge))
    return min(limits), max(limits)


def _attach_ranges(spec, candidate, options, tau):
    bounds = _bounds(options, candidate.cpa)
    p = candidate.params
    factors = np.geomspace(0.5, 2.0, 11)
    speed = _profile_range(spec, candidate, options, tau, bounds, 1, p[1] * factors)
    distance = _profile_range(spec, candidate, options, tau, bounds, 2, p[2] * factors)
    f0 = _profile_range(spec, candidate, options, tau, bounds, 0, p[0] * (1 + np.linspace(-0.03, 0.03, 7)))
    cpa = _profile_range(spec, candidate, options, tau, bounds, 3, p[3] + np.linspace(-6, 6, 7))
    candidate.extras.update(
        speed_range=(speed[0] / (1 + options.speed_margin), speed[1] * (1 + options.speed_margin)),
        distance_range=(
            distance[0] / (1 + options.distance_margin),
            distance[1] * (1 + options.distance_margin),
        ),
        f0_range=f0,
        cpa_range=cpa,
    )


def _to_event(spec, candidate, options):
    half = options.span / 2
    window = (
        float(max(spec.times[0], candidate.cpa - half)),
        float(min(spec.times[-1], candidate.cpa + half)),
    )
    extras = candidate.extras
    return Event(
        window,
        float(candidate.cpa),
        float(candidate.f0),
        float(candidate.speed),
        float(candidate.distance),
        options.harmonics,
        float(candidate.score),
        extras.get("speed_range"),
        extras.get("distance_range"),
        float(extras["z"]),
        extras.get("cpa_range"),
        extras.get("f0_range"),
    )


def _search(prepared, options, limits, tau, progress=None):
    """Coarse bank, multi-start refinement, null calibration, shared-evidence suppression."""
    coarse = _coarse_candidates(prepared, options, limits)
    refined = []
    failures = 0
    for i, candidate in enumerate(coarse):
        bounds = _bounds(options, candidate.cpa)
        bounds[3] = (max(bounds[3][0], limits[0]), min(bounds[3][1], limits[1]))
        if bounds[3][0] > bounds[3][1]:
            bounds[3] = (candidate.cpa, candidate.cpa)
        # Short fits screen the starts; only the best is polished to convergence.
        screened = [
            _maximise(prepared, start, options, tau, bounds, iterations=150, shrinks=(1.0,))
            for start in candidate.extras["starts"]
        ]
        p, score = _maximise(prepared, max(screened, key=lambda fit: fit[1])[0], options, tau, bounds)
        refined.append(_Pass(*map(float, p), score=score, extras={"coarse_score": candidate.score}))
        _calibrate(prepared, refined[-1], options, tau)
        failures = 0 if _passes(refined[-1], options) else failures + 1
        if progress:
            progress(i + 1, len(coarse))
        if options.patience and failures >= options.patience:
            break
    return _suppress(
        refined,
        lambda a, b: _shared_evidence(a, b, options, tau),
        key=lambda c: c.extras["z"],
        threshold=options.shared,
    )


def _passes(candidate, options):
    return candidate.extras["z"] >= options.min_z and candidate.extras["fragment"] <= options.max_fragment


def _summary(candidate, accepted=False):
    keys = ("z", "null_median", "null_sigma", "fragment", "coarse_score")
    return {
        "cpa": candidate.cpa,
        "f0": candidate.f0,
        "speed": candidate.speed,
        "distance": candidate.distance,
        "score": candidate.score,
        **{key: candidate.extras[key] for key in keys if key in candidate.extras},
        "accepted": accepted,
    }


def detect_flyovers(
    spec,
    options=FlyoverOptions(),
    top=10,
    limits=None,
    ranges=True,
    report=None,
    progress=None,
    null_search=False,
):
    """Calibrated, deterministic flyover detection over the whole recording.

    A log-frequency matched-filter bank of ideal passes proposes curves, overlap
    suppression removes duplicates (including harmonic aliases), each survivor is
    refined by Nelder-Mead, and its z-score against null curves decides whether
    it is reported at all.  ``limits`` restricts closest approaches to a time
    interval.  If ``report`` is a dict it receives every refined candidate,
    including rejected ones, with its z and fragment fraction.  With
    ``null_search`` the same search also runs on the time-reversed recording,
    where only rising (non-Doppler) curves can be found, and ``report`` receives
    those null candidates: an empirical false-alarm reference for ``min_z``.
    """
    if top < 1:
        raise ValueError("top must be positive")
    limits = (spec.times[0], spec.times[-1]) if limits is None else tuple(map(float, limits))
    if not limits[0] <= limits[1]:
        raise ValueError("detection limits must be increasing")
    prepared = suppress_tones(spec, options.tone_seconds, options.time_step)
    tau = _tau(prepared, options)
    refined = _search(prepared, options, limits, tau, progress)
    accepted = [c for c in refined if _passes(c, options)][:top]
    if ranges:
        for candidate in accepted:
            _attach_ranges(prepared, candidate, options, tau)
    if report is not None:
        report["candidates"] = [_summary(c, c in accepted) for c in refined]
        if null_search:
            flipped = Spectrogram(prepared.times, prepared.frequencies, prepared.power[:, ::-1])
            mirrored = (
                prepared.times[0] + prepared.times[-1] - limits[1],
                prepared.times[0] + prepared.times[-1] - limits[0],
            )
            nulls = _search(flipped, options, mirrored, tau)
            report["null_candidates"] = [
                {
                    **_summary(c),
                    "cpa": float(prepared.times[0] + prepared.times[-1] - c.cpa),
                    "passes_filters": _passes(c, options),
                }
                for c in nulls
            ]
    return [_to_event(spec, c, options) for c in accepted]


def fit_flyover(spec, window, options=FlyoverOptions(), ranges=True):
    """Best calibrated curve with closest approach inside ``window``; never abstains."""
    if len(window) != 2 or not window[0] < window[1]:
        raise ValueError("fit window must be an increasing pair")
    relaxed = replace(options, min_z=-np.inf, max_fragment=1.0, candidates=min(options.candidates, 6))
    events = detect_flyovers(spec, relaxed, top=1, limits=window, ranges=ranges)
    if not events:
        raise ValueError("no ideal pass with positive evidence inside the fit window")
    return events[0]


def save_events(events, path):
    with open(path, "w") as stream:
        json.dump([asdict(e) for e in events], stream, indent=2, allow_nan=False)


def load_events(paths):
    events = []
    for path in paths:
        with open(path) as stream:
            data = json.load(stream)
        events.extend(Event(**item) for item in (data if isinstance(data, list) else [data]))
    if not events:
        raise ValueError("at least one event is required")
    return sorted(events, key=lambda e: e.cpa)
