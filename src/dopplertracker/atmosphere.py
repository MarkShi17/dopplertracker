"""Horizontally stratified, time-invariant atmospheres and their acoustic travel times.

``LayeredAtmosphere`` holds piecewise-linear sound speed c(z) and horizontal wind
w(z) at ellipsoid heights.  For a stationary receiver the identity used by
``physics.predict``, f_received = f_source / (1 + grad_tau . v_source), stays
exact; only the travel time tau(x_source) and its gradient change.

Ray relations (moving stratified medium, dispersion c|s| + w.s = 1): the
horizontal slowness s_h is constant along a ray.  With sigma = (1 - w.s_h)/c and
q = sqrt(sigma^2 - |s_h|^2),

    tau            = integral sigma / (c q) dz
    D (src -> rcv) = integral (s_h + w sigma / c) / q dz

over the vertical span of the ray.  Solving D(s_h) = receiver - source for s_h
gives the direct ray, and grad_tau at the source = (-s_h, +q(z_source)) for a
source above the receiver (minus the slowness of the wave leaving the source).
"""

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path

import numpy as np

PRESSURE_LEVELS_HPA = (1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300)
OPEN_METEO_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_LEVEL_FIELDS = ("temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height")
_SURFACE_FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
)

# Gauss-Legendre nodes on [0, 1] mapped by z = a + (b - a) sin^2(pi t / 2).  The map
# clusters nodes at both ends of each layer, which keeps the 1/q integrable endpoint
# singularity of a near-grazing ray resolvable while staying spectrally accurate
# for the smooth integrands of ordinary rays.
_GAUSS = 8
_x, _w = np.polynomial.legendre.leggauss(_GAUSS)
_t = (_x + 1) / 2
_MAP = np.sin(np.pi * _t / 2) ** 2
_MAP_WEIGHT = _w / 2 * (np.pi / 2) * np.sin(np.pi * _t)
del _x, _w, _t


def saturation_vapour_pressure(temperature_c):
    """Magnus formula over water, hPa."""
    t = np.asarray(temperature_c, dtype=float)
    return 6.1094 * np.exp(17.625 * t / (t + 243.04))


def sound_speed(temperature_c, relative_humidity=0.0, pressure_hpa=1013.25):
    """c = 20.0468 sqrt(T_v), with virtual temperature T_v = T (1 + 0.608 q)."""
    t = np.asarray(temperature_c, dtype=float)
    vapour = np.asarray(relative_humidity, dtype=float) / 100.0 * saturation_vapour_pressure(t)
    specific = 0.622 * vapour / (np.asarray(pressure_hpa, dtype=float) - 0.378 * vapour)
    return 20.0468 * np.sqrt((t + 273.15) * (1 + 0.608 * specific))


def wind_components(speed, direction_from_deg):
    """Meteorological (speed, direction wind blows FROM) to (east, north) components."""
    angle = np.radians(np.asarray(direction_from_deg, dtype=float))
    speed = np.asarray(speed, dtype=float)
    return -speed * np.sin(angle), -speed * np.cos(angle)


@dataclass(frozen=True)
class LayeredAtmosphere:
    """Piecewise-linear c(z) and horizontal wind(z); z in ellipsoid metres.

    Values are held constant below the first and above the last level.  The
    offsets are constant shifts of the whole profile used by nuisance fits.
    ``c`` and ``wind`` are the (offset) values at 0 m so code written for a
    homogeneous ``Atmosphere`` can record a representative value.
    """

    heights: tuple
    sound_speed: tuple
    wind_east: tuple
    wind_north: tuple
    sound_speed_offset: float = 0.0
    wind_offset: tuple = (0.0, 0.0)
    description: str = field(default="", compare=False)

    def __post_init__(self):
        for name in ("heights", "sound_speed", "wind_east", "wind_north"):
            object.__setattr__(self, name, tuple(float(v) for v in np.ravel(getattr(self, name))))
        object.__setattr__(self, "sound_speed_offset", float(self.sound_speed_offset))
        object.__setattr__(self, "wind_offset", tuple(float(v) for v in self.wind_offset))
        h, c, we, wn = (
            np.array(getattr(self, n)) for n in ("heights", "sound_speed", "wind_east", "wind_north")
        )
        if (
            len(h) < 1
            or not (len(h) == len(c) == len(we) == len(wn))
            or len(self.wind_offset) != 2
            or not np.isfinite(np.r_[h, c, we, wn, self.sound_speed_offset, self.wind_offset]).all()
            or np.any(np.diff(h) <= 0)
        ):
            raise ValueError("layered atmosphere needs finite levels with strictly increasing heights")
        speed = c + self.sound_speed_offset
        wind = np.hypot(we + self.wind_offset[0], wn + self.wind_offset[1])
        if np.any(speed <= 0) or np.any(wind >= speed):
            raise ValueError("sound speed must be positive and greater than wind speed at every level")

    # ----- profile -------------------------------------------------------------------------------

    @cached_property
    def _levels(self):
        return (
            np.array(self.heights),
            np.array(self.sound_speed) + self.sound_speed_offset,
            np.array(self.wind_east) + self.wind_offset[0],
            np.array(self.wind_north) + self.wind_offset[1],
        )

    def profile(self, z):
        """Sound speed and wind (east, north) at heights z, offsets included."""
        h, c, we, wn = self._levels
        z = np.asarray(z, dtype=float)
        return np.interp(z, h, c), np.interp(z, h, we), np.interp(z, h, wn)

    @property
    def c(self):
        return float(self.profile(0.0)[0])

    @property
    def wind(self):
        _, we, wn = self.profile(0.0)
        return (float(we), float(wn), 0.0)

    @property
    def min_sound_speed(self):
        return float(np.min(self._levels[1]))

    @property
    def max_wind_speed(self):
        return float(np.max(np.hypot(self._levels[2], self._levels[3])))

    def offset(self, sound_speed=0.0, wind=(0.0, 0.0)):
        """Return the profile shifted by constant sound-speed and horizontal-wind offsets."""
        return replace(
            self,
            sound_speed_offset=self.sound_speed_offset + float(sound_speed),
            wind_offset=(self.wind_offset[0] + float(wind[0]), self.wind_offset[1] + float(wind[1])),
        )

    def harmonic_mean_sound_speed(self, z_low, z_high):
        z, weight = _nodes(np.atleast_1d(float(z_low)), np.atleast_1d(float(z_high)), self.heights)
        c, _, _ = self.profile(z)
        return float((z_high - z_low) / np.sum(weight / c))

    # ----- constructors / records ----------------------------------------------------------------

    @classmethod
    def from_lapse_rate(
        cls,
        surface_temperature_c=15.0,
        lapse_rate_k_per_km=6.5,
        surface_height_m=0.0,
        top_height_m=10000.0,
        wind_surface=(0.0, 0.0),
        wind_top=(0.0, 0.0),
        wind_top_height_m=None,
        relative_humidity=0.0,
        surface_pressure_hpa=1013.25,
        step_m=250.0,
    ):
        """Constant lapse rate, winds linear in height between the surface and wind_top_height_m."""
        if top_height_m <= surface_height_m or step_m <= 0:
            raise ValueError("lapse-rate profile needs top above surface and a positive step")
        z = np.unique(np.r_[np.arange(surface_height_m, top_height_m, step_m), top_height_m])
        temperature = surface_temperature_c - lapse_rate_k_per_km * (z - surface_height_m) / 1000.0
        # Hypsometric pressure for the humidity term only; dry air ignores it.
        pressure = surface_pressure_hpa * np.exp(-(z - surface_height_m) / 8400.0)
        c = sound_speed(temperature, relative_humidity, pressure)
        top = top_height_m if wind_top_height_m is None else wind_top_height_m
        fraction = np.clip((z - surface_height_m) / max(top - surface_height_m, 1e-9), 0.0, 1.0)
        wind_surface, wind_top = np.asarray(wind_surface, float), np.asarray(wind_top, float)
        wind = wind_surface + fraction[:, None] * (wind_top - wind_surface)
        return cls(
            z,
            c,
            wind[:, 0],
            wind[:, 1],
            description=f"lapse {lapse_rate_k_per_km} K/km from {surface_temperature_c} C",
        )

    def to_record(self):
        return {
            "type": "layered",
            "heights_m": list(self.heights),
            "sound_speed_m_s": list(self.sound_speed),
            "wind_east_m_s": list(self.wind_east),
            "wind_north_m_s": list(self.wind_north),
            "sound_speed_offset_m_s": self.sound_speed_offset,
            "wind_offset_en_m_s": list(self.wind_offset),
            "height_reference": "ellipsoid",
            "description": self.description,
        }

    @classmethod
    def from_record(cls, record):
        if record.get("type") != "layered":
            raise ValueError("not a layered atmosphere record")
        return cls(
            record["heights_m"],
            record["sound_speed_m_s"],
            record["wind_east_m_s"],
            record["wind_north_m_s"],
            record.get("sound_speed_offset_m_s", 0.0),
            tuple(record.get("wind_offset_en_m_s", (0.0, 0.0))),
            record.get("description", ""),
        )

    # ----- travel times --------------------------------------------------------------------------

    def travel_time(self, source_minus_observer, observer_height):
        """Direct-ray travel time and its gradient with respect to source position.

        Rays without a direct solution (acoustic shadow, nearly level paths) use
        the straight-path approximation; ``ray_travel_time`` exposes the flag.
        """
        tau, gradient, _ = self.ray_travel_time(source_minus_observer, observer_height)
        return tau, gradient

    def ray_travel_time(self, source_minus_observer, observer_height):
        """Exact stratified-medium direct ray: (tau, grad_tau, solved mask)."""
        r = np.asarray(source_minus_observer, dtype=float)
        shape = r.shape[:-1]
        r = r.reshape(-1, 3)
        tau, gradient, solved = _solve_rays(self, r, float(observer_height))
        if not solved.all():
            missing = ~solved
            tau[missing], gradient[missing] = _straight(self, r[missing], float(observer_height))
        return tau.reshape(shape), gradient.reshape(shape + (3,)), solved.reshape(shape)

    def straight_travel_time(self, source_minus_observer, observer_height):
        """Fermat approximation: integral of ds / (c + w.u) along the straight segment."""
        r = np.asarray(source_minus_observer, dtype=float)
        shape = r.shape[:-1]
        tau, gradient = _straight(self, r.reshape(-1, 3), float(observer_height))
        return tau.reshape(shape), gradient.reshape(shape + (3,))


def _nodes(lo, hi, heights):
    """Quadrature nodes and weights in z over [lo, hi], split at profile levels."""
    edges = np.r_[-np.inf, heights, np.inf]
    finite = np.isfinite(lo) & np.isfinite(hi)  # NaN rows (positions off the track) stay NaN
    if not finite.any():
        return np.full((len(lo), 1), np.nan), np.zeros((len(lo), 1))
    keep = (edges[:-1] < np.max(hi[finite])) & (edges[1:] > np.min(lo[finite]))
    low, high = edges[:-1][keep], edges[1:][keep]
    a = np.clip(lo[:, None], low, high)
    b = np.clip(hi[:, None], low, high)
    width = (b - a)[..., None]
    z = a[..., None] + width * _MAP
    weight = width * _MAP_WEIGHT
    return z.reshape(len(lo), -1), weight.reshape(len(lo), -1)


def _solve_rays(atmosphere, r, z_r, tolerance=1e-9, iterations=40):
    """Newton solve for the horizontal slowness of each direct ray, vectorised over rows of r."""
    n = len(r)
    tau = np.full(n, np.nan)
    gradient = np.full((n, 3), np.nan)
    solved = np.zeros(n, dtype=bool)
    horizontal_all = np.hypot(r[:, 0], r[:, 1])
    # Nearly level paths (elevation below ~0.06 degrees) have no well-conditioned vertical integral.
    index = np.flatnonzero(np.abs(r[:, 2]) > 1e-3 * np.maximum(horizontal_all, 1.0))
    if not len(index):
        return tau, gradient, solved
    z_s = z_r + r[index, 2]
    target = -r[index, :2]
    horizontal, vertical = horizontal_all[index], np.abs(r[index, 2])
    lo, hi = np.minimum(z_r, z_s), np.maximum(z_r, z_s)
    z, weight = _nodes(lo, hi, atmosphere.heights)
    c, we, wn = atmosphere.profile(z)
    # Endpoints carry no quadrature weight but must also propagate (q^2 > 0).
    ce, wee, wne = atmosphere.profile(np.stack((z_s, np.full_like(z_s, z_r)), axis=1))
    c_all, we_all, wn_all = (np.concatenate(v, axis=1) for v in ((c, ce), (we, wee), (wn, wne)))

    # Straight-line launch guess: |s| = 1 / (c + w . n) for wave normal n along the chord.
    mean_c, mean_we, mean_wn = (np.sum(weight * v, axis=1) / vertical for v in (c, we, wn))
    cos = horizontal / np.hypot(horizontal, vertical)
    direction = np.divide(
        target, horizontal[:, None], out=np.zeros_like(target), where=horizontal[:, None] > 0
    )
    along = (mean_we * direction[:, 0] + mean_wn * direction[:, 1]) * cos
    a = direction * (cos / (mean_c + along))[:, None]

    def feasible(rows, trial):
        sigma = (1 - we_all[rows] * trial[:, :1] - wn_all[rows] * trial[:, 1:]) / c_all[rows]
        return np.min(sigma * sigma - np.sum(trial * trial, axis=1)[:, None], axis=1) > 0

    # s_h = 0 always propagates, so shrinking the guess towards it restores feasibility.
    every = np.arange(len(index))
    ok = feasible(every, a)
    for _ in range(60):
        if ok.all():
            break
        a[~ok] *= 0.9
        ok[~ok] = feasible(every[~ok], a[~ok])
    active = np.ones(len(index), dtype=bool)
    done = np.zeros(len(index), dtype=bool)
    for _ in range(iterations):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        ar, cr, wer, wnr, wt = a[rows], c[rows], we[rows], wn[rows], weight[rows]
        sigma = (1 - wer * ar[:, :1] - wnr * ar[:, 1:]) / cr
        q = np.sqrt(sigma * sigma - np.sum(ar * ar, axis=1)[:, None])
        fe = (ar[:, :1] + wer * sigma / cr) / q
        fn = (ar[:, 1:] + wnr * sigma / cr) / q
        residual = np.stack((np.sum(wt * fe, axis=1), np.sum(wt * fn, axis=1)), axis=1) - target[rows]
        converged = np.hypot(residual[:, 0], residual[:, 1]) < tolerance
        done[rows[converged]] = True
        active[rows[converged]] = False
        keep = ~converged
        if not keep.any():
            break
        rows, ar, residual = rows[keep], ar[keep], residual[keep]
        cr, wer, wnr, wt, sigma, q, fe, fn = (v[keep] for v in (cr, wer, wnr, wt, sigma, q, fe, fn))
        # dq/da = (-sigma w / c - a) / q ;  dF/da = (I - w w^T / c^2) / q - F (dq/da)^T / q
        dqe = (-sigma * wer / cr - ar[:, :1]) / q
        dqn = (-sigma * wnr / cr - ar[:, 1:]) / q
        inv = wt / q
        inv_c2 = 1 / (cr * cr)
        jee = np.sum(inv * (1 - wer * wer * inv_c2 - fe * dqe), axis=1)
        jen = np.sum(inv * (-wer * wnr * inv_c2 - fe * dqn), axis=1)
        jne = np.sum(inv * (-wnr * wer * inv_c2 - fn * dqe), axis=1)
        jnn = np.sum(inv * (1 - wnr * wnr * inv_c2 - fn * dqn), axis=1)
        determinant = jee * jnn - jen * jne
        good = np.isfinite(determinant) & (np.abs(determinant) > 0) & np.isfinite(residual).all(axis=1)
        safe = np.where(good, determinant, 1.0)
        step = np.stack(
            (
                (jnn * residual[:, 0] - jen * residual[:, 1]) / safe,
                (-jne * residual[:, 0] + jee * residual[:, 1]) / safe,
            ),
            axis=1,
        )
        active[rows[~good]] = False
        rows, ar, step, residual = rows[good], ar[good], step[good], residual[good]
        # Near grazing, dX/ds_h is so large that one ulp of s_h exceeds the absolute
        # tolerance: accept a converged step instead.
        resolved = (np.hypot(step[:, 0], step[:, 1]) < 1e-13 * np.hypot(ar[:, 0], ar[:, 1])) & (
            np.hypot(residual[:, 0], residual[:, 1]) < 1e-5
        )
        done[rows[resolved]] = True
        active[rows[resolved]] = False
        rows, ar, step = rows[~resolved], ar[~resolved], step[~resolved]
        # Backtrack so every trial ray keeps propagating over its whole vertical span.
        scale = np.ones(len(rows))
        trial = ar - step
        ok = feasible(rows, trial)
        for _ in range(40):
            if ok.all():
                break
            retry = ~ok
            scale[retry] *= 0.5
            trial[retry] = ar[retry] - scale[retry, None] * step[retry]
            ok[retry] = feasible(rows[retry], trial[retry])
        # A shadow-zone target drives the ray to grazing; stop once steps collapse.
        active[rows[~ok | (scale < 1e-6)]] = False
        a[rows[ok]] = trial[ok]
    rows = np.flatnonzero(done)
    if len(rows):
        ar = a[rows]
        sigma = (1 - we[rows] * ar[:, :1] - wn[rows] * ar[:, 1:]) / c[rows]
        q = np.sqrt(sigma * sigma - np.sum(ar * ar, axis=1)[:, None])
        sigma_s = (1 - wee[rows, 0] * ar[:, 0] - wne[rows, 0] * ar[:, 1]) / ce[rows, 0]
        q_s = np.sqrt(np.maximum(sigma_s * sigma_s - np.sum(ar * ar, axis=1), 0.0))
        out = index[rows]
        tau[out] = np.sum(weight[rows] * sigma / (c[rows] * q), axis=1)
        gradient[out, :2] = -ar
        gradient[out, 2] = np.sign(z_s[rows] - z_r) * q_s
        solved[out] = np.isfinite(tau[out])
    return tau, gradient, solved


def _slopes(heights, values, z):
    """Piecewise-constant d(value)/dz at interior nodes; zero outside the levels."""
    h = np.asarray(heights)
    slope = np.r_[0.0, np.diff(values) / np.diff(h), 0.0] if len(h) > 1 else np.zeros(2)
    return slope[np.searchsorted(h, z, side="right")]


def _straight(atmosphere, r, z_r):
    length = np.linalg.norm(r, axis=1)
    rz = np.where(np.abs(r[:, 2]) < 1e-6, np.where(r[:, 2] < 0, -1e-6, 1e-6), r[:, 2])
    z_s = z_r + rz
    z, weight = _nodes(np.minimum(z_r, z_s), np.maximum(z_r, z_s), atmosphere.heights)
    t = (z - z_r) / rz[:, None]
    weight = weight / np.abs(rz)[:, None]
    c, we, wn = atmosphere.profile(z)
    h, levels_c, levels_we, levels_wn = atmosphere._levels
    dc, dwe, dwn = (_slopes(h, v, z) for v in (levels_c, levels_we, levels_wn))
    unit = r / np.maximum(length, 1e-12)[:, None]
    ue, un = -unit[:, :1], -unit[:, 1:2]  # propagation direction, source -> receiver
    denominator = c + we * ue + wn * un
    inverse2 = weight / (denominator * denominator)
    i0 = np.sum(weight / denominator, axis=1)
    iz = np.sum(inverse2 * t * (dc + dwe * ue + dwn * un), axis=1)
    iw = np.stack((np.sum(inverse2 * we, axis=1), np.sum(inverse2 * wn, axis=1), np.zeros(len(r))), axis=1)
    # d/dr of L * integral dt / (c(z_r + t r_z) + w . u(r)), u = -r / L.
    gradient = unit * i0[:, None] + iw - unit * np.sum(unit * iw, axis=1)[:, None]
    gradient[:, 2] -= length * iz
    gradient[length == 0] = 0.0
    return length * i0, gradient


@dataclass(frozen=True, eq=False)
class FlightAtmospheres:
    """One layered profile per flight, e.g. the hourly forecast interpolated to each event.

    ``physics.predict`` resolves the profile from the track id; ``default`` serves
    flights without their own entry (for example candidate searches).
    """

    flights: dict
    default: LayeredAtmosphere = None

    def __post_init__(self):
        object.__setattr__(self, "flights", {str(k): v for k, v in self.flights.items()})
        if not self.flights and self.default is None:
            raise ValueError("per-flight atmosphere needs at least one profile")

    def for_flight(self, flight_id):
        profile = self.flights.get(str(flight_id), self.default)
        if profile is None:
            raise ValueError(f"no atmosphere profile for flight {flight_id}")
        return profile

    @property
    def _members(self):
        return [*self.flights.values(), *([self.default] if self.default is not None else [])]

    @property
    def nominal(self):
        return self.default if self.default is not None else next(iter(self.flights.values()))

    @property
    def c(self):
        return self.nominal.c

    @property
    def wind(self):
        return self.nominal.wind

    @property
    def min_sound_speed(self):
        return min(m.min_sound_speed for m in self._members)

    @property
    def max_wind_speed(self):
        return max(m.max_wind_speed for m in self._members)

    def offset(self, sound_speed=0.0, wind=(0.0, 0.0)):
        return FlightAtmospheres(
            {k: v.offset(sound_speed, wind) for k, v in self.flights.items()},
            None if self.default is None else self.default.offset(sound_speed, wind),
        )

    def to_record(self):
        return {
            "type": "per_flight",
            "flights": {k: v.to_record() for k, v in self.flights.items()},
            "default": None if self.default is None else self.default.to_record(),
        }

    @classmethod
    def from_record(cls, record):
        return cls(
            {k: LayeredAtmosphere.from_record(v) for k, v in record["flights"].items()},
            None if record.get("default") is None else LayeredAtmosphere.from_record(record["default"]),
        )


def atmosphere_from_record(record):
    if record.get("type") == "layered":
        return LayeredAtmosphere.from_record(record)
    if record.get("type") == "per_flight":
        return FlightAtmospheres.from_record(record)
    raise ValueError(f"unknown atmosphere record type {record.get('type')!r}")


def atmosphere_from_result(result, default=None):
    """The medium a saved fit used: its layered record, else its homogeneous fields."""
    from .physics import Atmosphere

    if result.get("atmosphere") is not None:
        return atmosphere_from_record(result["atmosphere"])
    if default is not None and not isinstance(default, Atmosphere):
        return default
    return Atmosphere(
        result.get("sound_speed_m_s", 343.0), tuple(result.get("wind_enu_m_s", (0.0, 0.0, 0.0)))
    )


# ----- Open-Meteo profiles ------------------------------------------------------------------------


def open_meteo_url(date, latitude, longitude, levels=PRESSURE_LEVELS_HPA):
    hourly = [f"{name}_{level}hPa" for level in levels for name in _LEVEL_FIELDS] + list(_SURFACE_FIELDS)
    query = dict(
        latitude=f"{float(latitude):.4f}",
        longitude=f"{float(longitude):.4f}",
        start_date=date,
        end_date=date,
        hourly=",".join(hourly),
        wind_speed_unit="ms",
        timezone="UTC",
    )
    return OPEN_METEO_URL + "?" + urllib.parse.urlencode(query, safe=",")


def fetch_open_meteo(date, latitude, longitude, timeout=60.0):
    """Download one UTC day of hourly pressure-level profiles (free, keyless API)."""
    url = open_meteo_url(date, latitude, longitude)
    request = urllib.request.Request(url, headers={"User-Agent": "dopplertracker"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    if payload.get("error"):
        raise ValueError(f"Open-Meteo error: {payload.get('reason')}")
    return url, payload


@dataclass(frozen=True, eq=False)
class WeatherSeries:
    """Hourly layered profiles with identical levels, interpolated linearly in time."""

    times: tuple
    profiles: tuple
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "times", tuple(float(t) for t in self.times))
        object.__setattr__(self, "profiles", tuple(self.profiles))
        if (
            not self.times
            or len(self.times) != len(self.profiles)
            or np.any(np.diff(self.times) <= 0)
            or len({len(p.heights) for p in self.profiles}) != 1
        ):
            raise ValueError("weather series needs increasing times and profiles with the same levels")

    def at(self, utc, tolerance=3600.0):
        """Level-by-level linear interpolation in time (pressure levels move slightly)."""
        utc = float(utc)
        times = np.array(self.times)
        if utc < times[0] - tolerance or utc > times[-1] + tolerance:
            raise ValueError("requested time is outside the weather series")
        if len(times) == 1:
            return self.profiles[0]
        k = int(np.clip(np.searchsorted(times, utc) - 1, 0, len(times) - 2))
        f = float(np.clip((utc - times[k]) / (times[k + 1] - times[k]), 0.0, 1.0))
        a, b = self.profiles[k], self.profiles[k + 1]

        def blend(name):
            return (1 - f) * np.array(getattr(a, name)) + f * np.array(getattr(b, name))

        stamp = datetime.fromtimestamp(utc, timezone.utc).isoformat()
        return LayeredAtmosphere(
            blend("heights"),
            blend("sound_speed"),
            blend("wind_east"),
            blend("wind_north"),
            description=f"{self.metadata.get('source', 'weather')} interpolated to {stamp}",
        )

    def for_events(self, events, tracks, clock):
        """Per-flight profiles at each event's heard CPA (recording clock + event.cpa)."""
        moments = [float(clock) + event.cpa for event in events]
        return FlightAtmospheres(
            {str(track.id): self.at(moment) for track, moment in zip(tracks, moments, strict=True)},
            self.at(float(np.mean(moments))),
        )

    @classmethod
    def from_open_meteo(cls, response, geoid_offset=0.0, source="open-meteo historical forecast"):
        """Pressure-level and surface fields to ellipsoid-height profiles.

        Geopotential height is used as height above mean sea level; ellipsoid
        height = MSL + geoid undulation.  The surface level is the model ground
        (``elevation``) + 2 m, with the 10 m wind assigned to it.  Pressure levels
        not at least 10 m above that level at every hour are dropped.
        """
        hourly = response["hourly"]
        times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc).timestamp() for t in hourly["time"]]
        ground = float(response.get("elevation", 0.0)) + 2.0
        surface = {
            name: np.asarray(hourly.get(name, [np.nan] * len(times)), dtype=float) for name in _SURFACE_FIELDS
        }
        levels = []
        for level in PRESSURE_LEVELS_HPA:
            names = [f"{name}_{level}hPa" for name in _LEVEL_FIELDS]
            if not all(name in hourly for name in names):
                continue
            values = [np.asarray(hourly[name], dtype=float) for name in names]
            if np.isfinite(values).all() and np.all(values[4] > ground + 10.0):
                levels.append((level, values))
        if not levels:
            raise ValueError("weather response has no usable pressure levels")
        surface_ok = bool(np.isfinite(np.stack(list(surface.values()))).all())
        profiles = []
        for i in range(len(times)):
            rows = []  # height, pressure, temperature, humidity, speed, direction
            if surface_ok:
                rows.append(
                    [ground]
                    + [
                        surface[k][i]
                        for k in (
                            "surface_pressure",
                            "temperature_2m",
                            "relative_humidity_2m",
                            "wind_speed_10m",
                            "wind_direction_10m",
                        )
                    ]
                )
            for level, (t, rh, speed, direction, height) in levels:
                rows.append([height[i], float(level), t[i], rh[i], speed[i], direction[i]])
            rows = np.array(sorted(rows))
            east, north = wind_components(rows[:, 4], rows[:, 5])
            profiles.append(
                LayeredAtmosphere(
                    rows[:, 0] + geoid_offset,
                    sound_speed(rows[:, 2], rows[:, 3], rows[:, 1]),
                    east,
                    north,
                    description=f"{source} {hourly['time'][i]} UTC",
                )
            )
        metadata = dict(
            source=source,
            latitude=response.get("latitude"),
            longitude=response.get("longitude"),
            model_elevation_m=response.get("elevation"),
            geoid_offset_m=float(geoid_offset),
            height_reference="ellipsoid (geopotential height taken as MSL, plus geoid offset)",
            pressure_levels_hpa=[level for level, _ in levels],
            surface_level=surface_ok,
        )
        return cls(times, profiles, metadata)

    def to_record(self):
        return {
            "format": "dopplertracker-weather-1",
            "metadata": self.metadata,
            "times_utc": list(self.times),
            "times_iso": [datetime.fromtimestamp(t, timezone.utc).isoformat() for t in self.times],
            "profiles": [p.to_record() for p in self.profiles],
        }

    @classmethod
    def from_record(cls, record):
        if record.get("format") != "dopplertracker-weather-1":
            raise ValueError("not a dopplertracker weather file")
        return cls(
            record["times_utc"],
            [LayeredAtmosphere.from_record(p) for p in record["profiles"]],
            record.get("metadata", {}),
        )

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_record(), indent=2, allow_nan=False))

    @classmethod
    def load(cls, path):
        return cls.from_record(json.loads(Path(path).read_text()))
