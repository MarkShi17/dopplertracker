"""Ideal-pass fitting seeds coarse ranges; actual flight geometry determines ranking."""

import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .physics import ideal_curve
from .scoring import score_curve


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
