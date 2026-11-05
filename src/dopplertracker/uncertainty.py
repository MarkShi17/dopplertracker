"""Linearised error regions for projected fits (``advanced.fit_projected``).

The fit returns a point.  This module turns it into a region with a cause:

* **Measurement.**  Every fitted parameter (east, north, the recording clock, the
  fitted nuisances and each event's source-frequency polynomial) enters one
  Jacobian, computed by central differences through the same prediction path the
  fit used (prepared track, terrain projection, atmosphere offsets, clock rate,
  track offsets).  The source polynomial, which the fit eliminates inside each
  residual evaluation, is explicit here, so its uncertainty is carried rather
  than conditioned on.  The covariance is the M-estimator sandwich
  A^-1 B A^-1 for the loss the fit used: per event, the bread weight is the mean
  psi'(r) and the meat is the residual influence psi(r) with an AR(1)
  correlation between frames (below), both taken from the actual whitened
  residuals at the solution.  The source and nuisance priors of the fit enter A
  and B as pseudo-observations.
* **Considered inputs.**  Inputs held fixed in the fit but uncertain in reality
  (sound-speed and wind offsets, per-flight ADS-B time offsets, per-flight
  altitude offsets, observer height, per-flight velocity bias) get a prior
  (``InputPriors``); the estimate's sensitivity to each, G = -A^-1 J^T Omega J_k,
  maps it to position.  Inputs the fit estimates are not considered again: their
  uncertainty is already in the measurement sandwich (their prior part is
  reported under the input's own cause).

Why AR(1) and not a scalar effective sample size.  Spectrogram frames come from
overlapping 4 s windows (and demodulation from overlapping 2 s windows), and
model misfit (ADS-B noise, an atmosphere held fixed) wanders slowly, so
neighbouring residuals are correlated over seconds to tens of seconds.  A scalar
n_eff = n (1 - rho) / (1 + rho) is exact only for regressors that are constant
over the correlation time; the Doppler Jacobian columns change over the CPA
width (10-20 s), comparable to the misfit correlation time.  The sandwich with
the full AR(1) correlation matrix R_jk = rho^(|t_j - t_k| / dt) uses the
regressor structure, needs one parameter per event (stable with ~100 frames,
where a block or kernel estimator of a 7-parameter meat would not be), and its
exponential tail over-, not under-states the finite Hann-window correlation.  The
lag-1 estimate is corrected for its small-sample bias (Kendall: +(1 + 3 rho)/n)
and floored at 0.  The reported ``effective_frames`` is n (1 - rho) / (1 + rho).

Everything is local: one linearisation at the solution, Gaussian priors, no
multi-modality (a wrong flight, a wrong ridge or a wrong local minimum is not
covered).  Not modelled as considered inputs: audio sample-rate error (fit it
with ``fit_clock_rate``) and per-flight horizontal ADS-B position offsets
(along-track ones are equivalent to the time offsets).  Truth never enters.

Use: ``fit_projected(..., uncertainty=True)`` (or ``InputPriors(...)``) adds
``result["uncertainty"]``; ``position_uncertainty(result, events, tracks, ...)``
does the same for a result fitted with ``keep_observations=True``;
``predicted_uncertainty`` gives the expected region of a design without a fit;
``evaluate`` checks a region against a known position (evaluation only).
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from .advanced import ModelOptions, _noise_scale, fuse_velocity, prepare_track, smooth_trajectory
from .atmosphere import atmosphere_from_result
from .physics import Atmosphere, Trajectory, closest_approaches, predict

# 2-D Gaussian: P(d^2 <= k) = 1 - exp(-k / 2).
CHI2_2 = {0.5: -2 * np.log(0.5), 0.9: -2 * np.log(0.1)}
GROUPS = ("measurement", "atmosphere", "adsb_timing", "altitude", "observer_height", "velocity")

# Priors of the fit itself (advanced.fit_projected's residual); must match it.
SOURCE_PRIOR = (5.0, 1.2, 0.8)  # Hz, Hz/60 s, Hz/(60 s)^2
FITTED_PRIOR = {"sound_speed": 6.0, "wind_e": 12.0, "wind_n": 12.0, "altitude": 150.0, "clock_rate": 1500.0}
TRACK_OFFSET_PRIOR = 0.5

# Finite-difference half-steps for fitted parameters.
_STEP = {
    "east": 1.0,
    "north": 1.0,
    "clock": 0.005,
    "sound_speed": 0.25,
    "wind_e": 0.25,
    "wind_n": 0.25,
    "altitude": 1.0,
    "clock_rate": 20.0,
    "track_offset": 0.01,
}


@dataclass(frozen=True)
class InputPriors:
    """1-sigma priors for inputs a fit holds fixed.  ``None`` selects the documented default.

    sound_speed_m_s: constant offset of the whole sound-speed profile.
        Default atmosphere (one 343 m/s value, no weather): 6 m/s.  c = 20.05
        sqrt(T) changes 0.6 m/s per K, and the mean temperature of the lowest
        1-3 km at mid-latitudes spans roughly -10 to +25 C over a year, about
        10 K (1 sigma) around the 20 C behind 343 m/s; the fit's own prior when
        it estimates c is the same 6 m/s.
        Measured profile (``LayeredAtmosphere`` / weather): 1.0 m/s.  Reanalysis
        and short-range forecast temperatures in the lower troposphere are good
        to about 1-1.5 K RMS against radiosondes (ERA5 / GFS verification).
    wind_m_s: per horizontal component, constant offset of the wind profile.
        Default (still air): 8 m/s.  Climatological wind in the lowest 1-3 km is
        about 5-12 m/s in magnitude, i.e. 7-8 m/s per component.
        Measured profile: 2.5 m/s.  Reanalysis vector-wind errors against
        radiosondes are 2-3.5 m/s RMS in the lower troposphere, per component
        about 2-2.5 m/s.
    track_time_s: per flight, ADS-B position time-label error.  0.5 s: up to
        0.6 s uncompensated latency is allowed for ADS-B Out position
        (14 CFR 91.227 / AC 20-165B), plus aggregator timestamping; the fit's own
        track-offset prior is the same 0.5 s.
    altitude_m: per flight, aircraft height offset.  Geometric (GNSS ellipsoid)
        height: 20 m, from typical GNSS vertical accuracy (VFOM about 15-30 m at
        95 %) and 25 ft reporting steps.  Corrected barometric approximation
        (``altitude_sources`` contains ``corrected_baro_approximation``): 60 m;
        a 10 K non-standard column biases pressure altitude by about 4 % of the
        height above the station (60 m at 1500 m) before any QNH error.
    observer_height_m: receiver height.  10 m with an explicit constant
        elevation or none (no terrain data); 5 m with a DEM or terrain tiles
        (1-arcsecond DEM vertical RMSE about 4-6 m).
    velocity_m_s: per flight and axis, a constant velocity bias over the event
        (applied as a position drift about the event's CPA, so the track stays
        self-consistent).  Reported or fused velocity with reports present:
        0.2 m/s horizontal (``advanced.HORIZONTAL_VELOCITY_SIGMA``: 1 kt
        quantisation plus GNSS velocity error), 0.25 m/s vertical (geometric
        rate) or 0.6 m/s (barometric rate).  Differenced positions: the slope
        error of a straight-line fit over the 30 s around CPA (where the Doppler
        slope is set), sigma_p sqrt(12 / (N T^2)) with sigma_p the track's own
        position-noise estimate (``advanced._noise_scale``) and N reports in T
        = 30 s, floored at 0.3 m/s for interpolation bias in manoeuvres.
    """

    sound_speed_m_s: float | None = None
    wind_m_s: float | None = None
    track_time_s: float | None = None
    altitude_m: float | None = None
    observer_height_m: float | None = None
    velocity_m_s: float | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None and not (np.isfinite(value) and value >= 0):
                raise ValueError(f"prior {name} must be finite and nonnegative")


# Velocity bias window and floor for differenced positions (see InputPriors).
_VELOCITY_WINDOW_S = 30.0
_DIFFERENCED_VELOCITY_FLOOR = 0.3


def _position_noise(track, start, end):
    """Per-axis position noise (m) of raw reports in [start, end] (``advanced._noise_scale``)."""
    inside = (track.times >= start) & (track.times <= end)
    if inside.sum() < 3:
        inside = np.ones(len(track.times), bool)
    return _noise_scale(track.positions[inside], track.times[inside] - track.times[inside][0])


def _velocity_sigma(raw, options, heard_cpa):
    """Per-axis velocity-bias prior (m/s) for one flight; ``heard_cpa`` is track UTC."""
    reported = raw.velocities is not None and np.isfinite(raw.velocities[:, :2]).any()
    if options.velocity in ("reported", "fused") and reported:
        runs = raw.metadata.get("vertical_rate_source_runs") or []
        baro = any(label == "baro" for label, _ in runs)
        return np.array([0.2, 0.2, 0.6 if baro else 0.25])
    half = _VELOCITY_WINDOW_S / 2
    window = (raw.times >= heard_cpa - half) & (raw.times <= heard_cpa + half)
    count = max(int(window.sum()), 2)
    sigma_p = _position_noise(raw, heard_cpa - 3 * half, heard_cpa + 3 * half)
    slope = sigma_p * np.sqrt(12.0 / (count * _VELOCITY_WINDOW_S**2))
    return np.maximum(slope, _DIFFERENCED_VELOCITY_FLOOR)


def _resolve_priors(priors, result, raw_tracks, options, elevation, heard_cpa):
    """name -> (sigma, cause) for every considered (held-fixed) input."""
    priors = priors or InputPriors()
    measured = result.get("atmosphere") is not None
    terrain = elevation is not None and type(elevation).__name__ in ("GridElevation", "TerrainTiles")
    out = {}
    c = priors.sound_speed_m_s if priors.sound_speed_m_s is not None else (1.0 if measured else 6.0)
    w = priors.wind_m_s if priors.wind_m_s is not None else (2.5 if measured else 8.0)
    if not options.fit_sound_speed:
        out["sound_speed"] = (c, "atmosphere")
    if not options.fit_wind:
        out["wind_e"] = (w, "atmosphere")
        out["wind_n"] = (w, "atmosphere")
    timing = priors.track_time_s if priors.track_time_s is not None else 0.5
    height = priors.observer_height_m
    if height is None:
        height = 5.0 if terrain else 10.0
    out["observer_height"] = (height, "observer_height")
    for i, track in enumerate(raw_tracks):
        # Flights 2..N are fitted with fit_track_offsets; flight 1's offset is the
        # part the shared clock absorbs, so its residual effect is still considered.
        if not (options.fit_track_offsets and i > 0):
            out[f"track_offset_{i}"] = (timing, "adsb_timing")
        if priors.altitude_m is not None:
            altitude = priors.altitude_m
        else:
            baro = "corrected_baro_approximation" in (track.metadata.get("altitude_sources") or [])
            altitude = 60.0 if baro else 20.0
        out[f"aircraft_altitude_{i}"] = (altitude, "altitude")
        velocity = (
            np.full(3, priors.velocity_m_s)
            if priors.velocity_m_s is not None
            else _velocity_sigma(track, options, heard_cpa[i])
        )
        for axis, label in enumerate("enu"):
            out[f"velocity_{i}_{label}"] = (float(velocity[axis]), "velocity")
    return {k: (float(s), g) for k, (s, g) in out.items() if s > 0}


def _cause(name):
    if name in ("sound_speed", "wind_e", "wind_n"):
        return "atmosphere"
    if name.startswith("track_offset"):
        return "adsb_timing"
    if name == "altitude":
        return "altitude"
    return "measurement"


class _Model:
    """Re-predicts a saved projected fit under small parameter and input perturbations."""

    def __init__(self, result, events, tracks, frame=None, elevation=None):
        if result.get("method") != "projected":
            raise ValueError("uncertainty needs a projected fit (advanced.fit_projected)")
        if len(events) != len(tracks) or len(tracks) != len(result["f0"]):
            raise ValueError("events, flights and fitted source frequencies must match")
        self.options = ModelOptions(**result["options"])
        self.events, self.raw = list(events), list(tracks)
        self.frame, self.elevation = frame, elevation
        self.origin = float(result["clock"])
        self.position = np.asarray(result["position"], dtype=float)
        self.altitude = float(result.get("aircraft_altitude_offset_m", 0.0))
        self.rate_ppm = float(result.get("clock_rate_ppm", 0.0))
        n = len(events)
        self.offsets = list(result.get("track_time_offsets_s") or [0.0] * n)
        references = result.get("source_reference_utc") or [self.origin + e.cpa for e in events]
        self.reference = [float(r) - self.origin for r in references]
        degree = self.options.degree
        drift = result.get("frequency_drift_hz_s", [0.0] * n)
        curvature = result.get("frequency_curvature_hz_s2", [0.0] * n)
        self.coefficients = [
            np.array([f, d * 60.0, c * 3600.0])[: degree + 1]
            for f, d, c in zip(result["f0"], drift, curvature, strict=True)
        ]
        self.atmosphere = atmosphere_from_result(result)
        self.layered = not isinstance(self.atmosphere, Atmosphere)
        self.tracks = [self._prepare(t).relative_to(self.origin) for t in tracks]
        names = ["east", "north", "clock"]
        for enabled, group in (
            (self.options.fit_sound_speed, ["sound_speed"]),
            (self.options.fit_wind, ["wind_e", "wind_n"]),
            (self.options.fit_altitude, ["altitude"]),
            (self.options.fit_clock_rate, ["clock_rate"]),
        ):
            if enabled:
                names.extend(group)
        if self.options.fit_track_offsets:
            names.extend(f"track_offset_{i}" for i in range(1, n))
        self.fitted = names
        self._cpa_emit = [None] * n

    def _prepare(self, track):
        # As advanced.predict_result: replay the fit's smoothing / velocity model.
        options = self.options
        if options.velocity == "fused":
            return track if track.metadata.get("velocity_fusion") == "kalman" else fuse_velocity(track)
        if options.smoothing != "none" and track.metadata.get("smoothing") != options.smoothing:
            track = smooth_trajectory(track, options.smoothing)
        return prepare_track(track, "none", options.velocity)

    def observer(self, east=0.0, north=0.0):
        observer = self.position + np.array([east, north, 0.0])
        if self.frame is not None and self.elevation is not None and (east or north):
            lat, lon, _ = self.frame.to_geodetic(observer)
            observer = self.frame.to_local([lat, lon, float(self.elevation(lat, lon))])
        return observer

    def medium(self, delta):
        dc, de, dn = (delta.get(k, 0.0) for k in ("sound_speed", "wind_e", "wind_n"))
        if not (dc or de or dn):
            return self.atmosphere
        if self.layered:
            return self.atmosphere.offset(dc, (de, dn))
        a = self.atmosphere
        return Atmosphere(a.c + dc, (a.wind[0] + de, a.wind[1] + dn, a.wind[2]))

    def cpa_emit(self, i):
        """Track-local emission time of the event's heard CPA (velocity-bias pivot)."""
        if self._cpa_emit[i] is None:
            receive = np.array([self.reference[i]]) * (1 + self.rate_ppm * 1e-6)
            _, emit = predict(self.tracks[i], self.observer() - [0, 0, self.altitude], receive, 1.0,
                              self.atmosphere)  # fmt: skip
            self._cpa_emit[i] = float(emit[0]) if np.isfinite(emit[0]) else float(self.reference[i])
        return self._cpa_emit[i]

    def track(self, i, delta):
        dz = delta.get(f"aircraft_altitude_{i}", 0.0)
        dv = np.array([delta.get(f"velocity_{i}_{a}", 0.0) for a in "enu"])
        track = self.tracks[i]
        if not dz and not dv.any():
            return track
        positions = (
            track.positions + np.array([0.0, 0.0, dz]) + dv * (track.times - self.cpa_emit(i))[:, None]
        )
        velocities = None if track.velocities is None else track.velocities + dv
        return Trajectory(
            track.id, track.times, positions, track.max_gap, track.metadata, velocities, track.use_velocities
        )

    def factor_elapsed(self, i, times, delta):
        """Doppler factor and emission time relative to the source reference for event i."""
        observer = self.observer(delta.get("east", 0.0), delta.get("north", 0.0))
        observer = observer + np.array([0.0, 0.0, delta.get("observer_height", 0.0)])
        effective = observer - np.array([0.0, 0.0, self.altitude + delta.get("altitude", 0.0)])
        rate = 1 + (self.rate_ppm + delta.get("clock_rate", 0.0)) * 1e-6
        receive = np.asarray(times, dtype=float) * rate + delta.get("clock", 0.0)
        offset = self.offsets[i] + delta.get(f"track_offset_{i}", 0.0)
        factor, emit = predict(
            self.track(i, delta), effective, receive + offset, atmosphere=self.medium(delta)
        )
        return factor, emit - offset - self.reference[i]

    def prediction(self, i, times, delta=None, coefficients=None):
        factor, elapsed = self.factor_elapsed(i, times, delta or {})
        coefs = self.coefficients[i] if coefficients is None else coefficients
        return factor * np.polynomial.polynomial.polyval(elapsed / 60, coefs)

    def affects(self, name, i):
        """Whether parameter ``name`` can change event i's prediction."""
        for prefix in ("track_offset_", "aircraft_altitude_", "velocity_"):
            if name.startswith(prefix):
                return int(name[len(prefix) :].split("_")[0]) == i
        return True


def _normalise_observations(observations, result):
    """(times, ridge, mask, weights) per event, from fit internals or ``result["observations"]``."""
    if observations is None:
        if "observations" not in result:
            raise ValueError("pass observations or fit with keep_observations=True")
        if result.get("validation_samples"):
            raise ValueError("uncertainty describes a full fit, not a validation-fold fit")
        observations = []
        for i, o in enumerate(result["observations"]):
            valid = np.asarray(o["valid"], bool)
            sigma = np.asarray(o["sigma_hz"], float)
            weights = np.where(valid, (0.3 / np.where(valid, sigma, 1.0)) ** 2, 0.0)
            mask = valid & (i != result.get("excluded_event"))
            observations.append(
                (np.asarray(o["times"], float), np.asarray(o["ridge_hz"], float), mask, weights)
            )
    return [tuple(np.asarray(x) for x in o) for o in observations]


def _psi(loss, r):
    """Influence psi(r) and its derivative for scipy's losses with f_scale = 1."""
    if loss == "linear":
        return r, np.ones_like(r)
    if loss == "soft_l1":
        s = 1 + r * r
        return r / np.sqrt(s), s**-1.5
    if loss == "huber":
        inside = np.abs(r) <= 1
        return np.where(inside, r, np.sign(r)), inside.astype(float)
    if loss == "cauchy":
        s = 1 + r * r
        return r / s, (1 - r * r) / (s * s)
    raise ValueError(f"unsupported loss {loss}")


def ar1_correlation(times, values, spacing=None):
    """Bias-corrected lag-1 correlation of ``values`` at the nominal frame ``spacing``.

    Only pairs of consecutive samples exactly one spacing apart count.  Kendall's
    small-sample correction +(1 + 3 rho) / n is applied; the result is clipped to
    [0, 0.98] (anticorrelation is not allowed to shrink the region).
    """
    times, values = np.asarray(times, float), np.asarray(values, float)
    if len(times) < 4:
        return 0.0
    spacing = float(np.median(np.diff(times))) if spacing is None else spacing
    adjacent = np.abs(np.diff(times) - spacing) < 0.01 * spacing
    denominator = float(np.sum(values * values))
    if adjacent.sum() < 3 or denominator <= 0:
        return 0.0
    rho = float(np.sum(values[:-1][adjacent] * values[1:][adjacent]) / denominator)
    rho += (1 + 3 * rho) / len(values)
    return float(np.clip(rho, 0.0, 0.98))


def _derivatives(model, observations, names, steps):
    """Central-difference d prediction / d name for every event (zero where it cannot act)."""
    out = []
    for i, (times, _, _, _) in enumerate(observations):
        columns = np.zeros((len(times), len(names)))
        for k, name in enumerate(names):
            if not model.affects(name, i):
                continue
            h = steps[name]
            plus = model.prediction(i, times, {name: h})
            minus = model.prediction(i, times, {name: -h})
            columns[:, k] = np.nan_to_num((plus - minus) / (2 * h), nan=0.0)
        out.append(columns)
    return out


def _step(name, considered):
    if name in considered:
        return 0.5 * considered[name][0]
    return _STEP["track_offset" if name.startswith("track_offset") else name]


def _linearise(model, observations, considered, moments=None):
    """Bread, meat parts and sensitivities of the linearised estimator.

    ``moments`` (per event: bread weight, meat scale, lag-1 correlation) replaces
    the residual-based estimates; the what-if helper uses it.
    """
    n_events = len(observations)
    degree = model.options.degree
    fitted = model.fitted
    k_names = list(considered)
    steps = {name: _step(name, considered) for name in fitted + k_names}
    geometric = _derivatives(model, observations, fitted, steps)
    held = _derivatives(model, observations, k_names, steps)
    n_source = (degree + 1) * n_events
    p = len(fitted) + n_source
    rows, events = [], []
    for i, (times, ridge, mask, weights) in enumerate(observations):
        factor, elapsed = model.factor_elapsed(i, times, {})
        prediction = factor * np.polynomial.polynomial.polyval(elapsed / 60, model.coefficients[i])
        use = mask & np.isfinite(prediction) & np.isfinite(ridge)
        scale = np.sqrt(weights[use]) / 0.3
        j = np.zeros((use.sum(), p))
        j[:, : len(fitted)] = geometric[i][use]
        vander = factor[use, None] * np.polynomial.polynomial.polyvander(elapsed[use] / 60, degree)
        start = len(fitted) + i * (degree + 1)
        j[:, start : start + degree + 1] = vander
        residual = (prediction - ridge)[use] * scale
        rows.append((j * scale[:, None], held[i][use] * scale[:, None], residual, times[use]))
    total = sum(len(r[2]) for r in rows)
    if total <= p:
        raise ValueError("too few measured frames for an error region")
    dof = total / (total - p)
    bread = np.zeros((p, p))
    data_meat = np.zeros((p, p))
    cross = np.zeros((p, len(k_names)))
    for i, (j, jk, residual, times) in enumerate(rows):
        if not len(residual):
            events.append({"frames": 0})
            continue
        spacing = float(np.median(np.diff(observations[i][0]))) if len(observations[i][0]) > 1 else 1.0
        if moments is None:
            psi, derivative = _psi(model.options.loss, residual)
            omega = max(float(np.mean(derivative)), 1e-3)
            meat = float(np.mean(psi * psi)) * dof
            rho = ar1_correlation(times, psi, spacing)
        else:
            omega, meat, rho = moments[i]
        lags = np.abs(times[:, None] - times[None, :]) / spacing
        correlation = rho**lags if rho > 0 else np.eye(len(times))
        bread += omega * j.T @ j
        data_meat += meat * j.T @ correlation @ j
        cross += omega * j.T @ jk
        events.append(
            {
                "frames": int(len(residual)),
                "residual_rms_sigma": float(np.sqrt(np.mean(residual * residual))),
                # Effective per-frame noise relative to the frames' stated sigma.
                "noise_scale": float(np.sqrt(meat) / omega),
                "lag1_correlation": float(rho),
                "effective_frames": float(len(residual) * (1 - rho) / (1 + rho)),
            }
        )
    prior = np.zeros(p)
    causes = [_cause(name) for name in fitted]
    for k, name in enumerate(fitted):
        if name in FITTED_PRIOR:
            prior[k] = FITTED_PRIOR[name] ** -2
        elif name.startswith("track_offset"):
            prior[k] = TRACK_OFFSET_PRIOR**-2
    for i in range(n_events):
        causes.extend(["measurement"] * (degree + 1))
        if i != getattr(
            model, "excluded", None
        ):  # an excluded event's polynomial is unconstrained and unused
            start = len(fitted) + i * (degree + 1)
            prior[start : start + degree + 1] = np.array(SOURCE_PRIOR[: degree + 1]) ** -2
    bread += np.diag(prior)
    return {
        "bread": bread,
        "data_meat": data_meat,
        "prior": prior,
        "prior_causes": causes,
        "cross": cross,
        "considered": k_names,
        "events": events,
        "parameters": p,
        "frames": total,
    }


def _assemble(model, linear, considered):
    """Position covariance by cause from the linearised estimator."""
    inverse = np.linalg.pinv(linear["bread"])
    groups = {g: np.zeros((2, 2)) for g in GROUPS}
    groups["measurement"] += (inverse @ linear["data_meat"] @ inverse)[:2, :2]
    for k, (precision, cause) in enumerate(zip(linear["prior"], linear["prior_causes"], strict=True)):
        if precision > 0:
            column = inverse[:2, k]
            groups[cause] += precision * np.outer(column, column)
    sensitivity = -(inverse @ linear["cross"])[:2]  # metres per unit of each considered input
    rows = {}
    for k, name in enumerate(linear["considered"]):
        sigma, cause = considered[name]
        shift = sensitivity[:, k] * sigma
        groups[cause] += np.outer(shift, shift)
        rows[name] = {"prior_sigma": sigma, "cause": cause, "position_shift_en_m": shift.tolist()}
    return groups, rows, inverse


def ellipse(covariance, probability):
    """Semi-axes (m) and major-axis bearing (degrees clockwise from north, 0-180)."""
    covariance = np.asarray(covariance, float)
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    values = np.maximum(values, 0.0)
    k = -2 * np.log(1 - probability)
    major = vectors[:, 1]
    bearing = float(np.degrees(np.arctan2(major[0], major[1])) % 180.0)
    return {
        "probability": probability,
        "semi_major_m": float(np.sqrt(k * values[1])),
        "semi_minor_m": float(np.sqrt(k * values[0])),
        "major_axis_bearing_deg": bearing,
    }


def mahalanobis(covariance, error_en):
    """Normalised horizontal error sqrt(e^T C^-1 e); Rayleigh(1) if the region is honest."""
    e = np.asarray(error_en, float)[:2]
    return float(np.sqrt(e @ np.linalg.pinv(np.asarray(covariance, float)) @ e))


def evaluate(uncertainty, error_en):
    """Truth-side check of one region: normalised error and 50 % / 90 % ellipse membership."""
    out = {}
    for key, matrix in (
        ("total", uncertainty["covariance_en_m2"]),
        ("measurement", uncertainty["breakdown"]["measurement"]["covariance_en_m2"]),
    ):
        d = mahalanobis(matrix, error_en)
        out[key] = {
            "normalised_error": d,
            "inside_50": bool(d * d <= CHI2_2[0.5]),
            "inside_90": bool(d * d <= CHI2_2[0.9]),
        }
    return out


def _report(groups, rows, events, model, considered, linear):
    total = sum(groups.values())
    values, vectors = np.linalg.eigh((total + total.T) / 2)
    major = vectors[:, 1]
    trace = float(np.trace(total))
    along = float(major @ total @ major)
    breakdown = {
        g: {
            "variance_m2": float(np.trace(c)),
            "fraction": float(np.trace(c) / trace) if trace > 0 else 0.0,
            "major_axis_fraction": float(major @ c @ major / along) if along > 0 else 0.0,
            "covariance_en_m2": c.tolist(),
        }
        for g, c in groups.items()
    }
    dominant = max(GROUPS, key=lambda g: breakdown[g]["fraction"])
    e90 = ellipse(total, 0.9)
    return {
        "covariance_en_m2": total.tolist(),
        "sd_east_north_m": np.sqrt(np.maximum(np.diag(total), 0)).tolist(),
        "ellipses": {"50": ellipse(total, 0.5), "90": e90},
        "measurement_ellipses": {
            "50": ellipse(groups["measurement"], 0.5),
            "90": ellipse(groups["measurement"], 0.9),
        },
        "breakdown": breakdown,
        "dominant_cause": dominant,
        "summary": (
            f"90 % region {e90['semi_major_m']:.0f} m x {e90['semi_minor_m']:.0f} m (semi-axes), "
            f"long axis bearing {e90['major_axis_bearing_deg']:.0f} deg; "
            f"largest cause {dominant} ({100 * breakdown[dominant]['fraction']:.0f} % of variance)"
        ),
        "fitted_parameters": model.fitted,
        "considered_inputs": rows,
        "events": events,
        "frames": linear["frames"],
        "parameters": linear["parameters"],
        "method": "linearised M-estimator sandwich, AR(1) frame correlation, considered-input priors",
        "note": "Local and Gaussian: a wrong flight, ridge or local minimum is not covered.",
    }


def position_uncertainty(result, events, tracks, frame=None, elevation=None, priors=None, observations=None):
    """Error region of a projected fit's horizontal position.

    ``tracks`` are the raw flights given to the fit; ``frame`` / ``elevation`` as
    given to it.  ``observations`` defaults to ``result["observations"]`` (fit with
    ``keep_observations=True``); ``fit_projected(..., uncertainty=True)`` passes
    its own.  ``priors`` (``InputPriors``) overrides the considered-input priors.
    """
    model = _Model(result, events, tracks, frame, elevation)
    model.excluded = result.get("excluded_event")
    observations = _normalise_observations(observations, result)
    heard = [model.origin + model.reference[i] for i in range(len(events))]
    considered = _resolve_priors(priors, result, model.raw, model.options, elevation, heard)
    linear = _linearise(model, observations, considered)
    groups, rows, _ = _assemble(model, linear, considered)
    return _report(groups, rows, linear["events"], model, considered, linear)


# ----- What-if: the expected region for a design, without a fit ---------------------------------


def hypothetical_event(track, position, clock, f0, atmosphere=Atmosphere(), half_window=45.0, harmonics=2):
    """An ``Event`` at a flight's heard closest approach to ``position`` (recording clock ``clock``)."""
    from .detection import Event

    hits = closest_approaches(track, np.asarray(position, float), atmosphere)
    if not hits:
        raise ValueError(f"flight {track.id} has no closest approach")
    heard, distance, speed = min(hits, key=lambda h: h[1])
    cpa = heard - clock
    return Event((cpa - half_window, cpa + half_window), cpa, f0, speed, distance, harmonics)


def window_correlation(spacing, window_seconds):
    """A-priori AR(1) lag-1 correlation of frames from overlapping Hann windows.

    Matches the long-run variance (1 + rho) / (1 - rho) to the frames per
    independent measurement, 2 T / (3 dt), of ``advanced._correlation_inflation``.
    """
    frames = max(1.0, 2 * window_seconds / 3 / spacing)
    return float((frames - 1) / (frames + 1))


def predicted_uncertainty(
    events,
    tracks,
    position,
    clock,
    f0,
    options=ModelOptions(degree=1),
    atmosphere=Atmosphere(),
    frame=None,
    elevation=None,
    priors=None,
    noise_hz=0.3,
    hop=0.5,
    window_seconds=4.0,
    lag1_correlation=None,
):
    """Expected error region at ``position`` for these events and flights, without measuring.

    Assumes every frame in each event window is measured (``hop`` spacing, thinned
    to ``options.max_frames``) with 1-sigma ``noise_hz`` (a number or one per
    event; for example a previous fit's ``noise_scale`` x its median frame
    sigma), AR(1) correlation ``lag1_correlation`` (a number or one per event;
    default: from ``window_seconds`` overlap), a linear loss and a constant
    source frequency ``f0`` per event.  Ask "how much would a third flyover
    help" by adding a ``hypothetical_event``.
    """
    n = len(events)
    if len(tracks) != n or len(f0) != n or not n:
        raise ValueError("events, flights and source frequencies must match")
    noise = np.broadcast_to(np.asarray(noise_hz, float), (n,))
    if np.any(noise <= 0):
        raise ValueError("noise must be positive")
    if lag1_correlation is not None:
        lag1_correlation = np.broadcast_to(np.asarray(lag1_correlation, float), (n,))
        if np.any((lag1_correlation < 0) | (lag1_correlation >= 1)):
            raise ValueError("lag-1 correlation must be in [0, 1)")
    layered = not isinstance(atmosphere, Atmosphere)
    result = {
        "method": "projected",
        "position": np.asarray(position, float).tolist(),
        "clock": float(clock),
        "f0": [float(f) for f in f0],
        "frequency_drift_hz_s": [0.0] * n,
        "frequency_curvature_hz_s2": [0.0] * n,
        "source_reference_utc": [float(clock) + e.cpa for e in events],
        "sound_speed_m_s": atmosphere.c,
        "wind_enu_m_s": list(atmosphere.wind),
        "options": asdict(replace(options, loss="linear")),
        **({"atmosphere": atmosphere.to_record()} if layered else {}),
        **({"track_time_offsets_s": [0.0] * n} if options.fit_track_offsets else {}),
    }
    model = _Model(result, events, tracks, frame, elevation)
    observations, moments = [], []
    for i, event in enumerate(events):
        times = np.arange(event.window[0], event.window[1] + hop / 2, hop)
        if options.max_frames is not None:
            times = times[:: max(1, int(np.ceil(len(times) / options.max_frames)))]
        model_hz = model.prediction(i, times)
        valid = np.isfinite(model_hz)
        weights = np.full(len(times), (0.3 / noise[i]) ** 2)
        observations.append((times, np.where(valid, model_hz, 0.0), valid, weights))
        spacing = float(np.median(np.diff(times)))
        rho = window_correlation(spacing, window_seconds) if lag1_correlation is None else lag1_correlation[i]
        moments.append((1.0, 1.0, float(rho)))
    heard = [model.origin + model.reference[i] for i in range(n)]
    considered = _resolve_priors(priors, result, model.raw, model.options, elevation, heard)
    linear = _linearise(model, observations, considered, moments)
    groups, rows, _ = _assemble(model, linear, considered)
    out = _report(groups, rows, linear["events"], model, considered, linear)
    out["method"] = "predicted (no measurement): " + out["method"]
    out["assumed_noise_hz"] = noise.tolist()
    return out


# ----- Cross-check: block bootstrap of the measurement part -------------------------------------


def _refit(model, observations, rows):
    """Refit east, north, clock, fitted nuisances and source polynomials on chosen frames.

    ``rows`` holds per event the frame indices to use (repeats allowed, so a
    resample is exact for robust losses).  Starts at, and returns the shift from,
    the saved solution.  Uses the fit's loss and prior widths; the priors are
    centred on the solution (a result does not store the start candidate they are
    anchored at), so they act only through their curvature.  Ridges are not
    re-measured and considered inputs stay fixed.
    """
    degree, fitted, n = model.options.degree, model.fitted, len(observations)
    width = [
        FITTED_PRIOR.get(name, TRACK_OFFSET_PRIOR if name.startswith("track_offset") else np.inf)
        for name in fitted
    ]
    width = np.r_[width, np.tile(SOURCE_PRIOR[: degree + 1], n)]

    def objective(x):
        delta = dict(zip(fitted, x[: len(fitted)], strict=True))
        out = []
        for i, (times, ridge, _, weights) in enumerate(observations):
            idx = rows[i]
            if not len(idx):
                continue
            coefs = model.coefficients[i] + x[len(fitted) + i * (degree + 1) :][: degree + 1]
            prediction = model.prediction(i, times[idx], delta, coefs)
            out.append(np.nan_to_num((prediction - ridge[idx]) * np.sqrt(weights[idx]) / 0.3, nan=50.0))
        out.append(x / width)
        return np.concatenate(out)

    scale = np.r_[
        [_STEP["track_offset" if name.startswith("track_offset") else name] * 50 for name in fitted],
        np.tile(SOURCE_PRIOR[: degree + 1], n),
    ]
    x0 = np.zeros(len(fitted) + n * (degree + 1))
    fit = least_squares(
        objective, x0, loss=model.options.loss, f_scale=1.0, x_scale=scale, max_nfev=80,
        ftol=1e-8, xtol=1e-8, gtol=1e-8,
    )  # fmt: skip
    return fit.x


def _used_rows(model, observations):
    out = []
    for i, (times, ridge, mask, _) in enumerate(observations):
        prediction = model.prediction(i, times)
        out.append(np.flatnonzero(mask & np.isfinite(prediction) & np.isfinite(ridge)))
    return out


def block_bootstrap(
    result,
    events,
    tracks,
    frame=None,
    elevation=None,
    observations=None,
    block_seconds=10.0,
    resamples=100,
    seed=0,
):
    """Moving-block bootstrap of the measurement part of the position covariance.

    Each resample draws, per event, blocks of ``block_seconds`` of consecutive
    measured frames with replacement until the event's frame count is reached,
    and refits (``_refit``).  The spread compares with the ``measurement`` part
    of ``position_uncertainty``; it is a cross-check of the linearisation and of
    the AR(1) correlation allowance, not of the considered inputs.
    """
    model = _Model(result, events, tracks, frame, elevation)
    observations = _normalise_observations(observations, result)
    used = _used_rows(model, observations)
    rng = np.random.default_rng(seed)
    shifts = []
    for _ in range(resamples):
        rows = []
        for i, idx in enumerate(used):
            if len(idx) < 2:
                rows.append(idx)
                continue
            t = observations[i][0][idx]
            starts = np.flatnonzero(t <= t[-1] - block_seconds + 1e-9)
            starts = starts if len(starts) else np.array([0])
            picked = []
            while sum(len(p) for p in picked) < len(idx):
                begin = int(rng.choice(starts))
                picked.append(idx[(t >= t[begin]) & (t < t[begin] + block_seconds)])
            rows.append(np.concatenate(picked)[: len(idx)])
        shifts.append(_refit(model, observations, rows)[:2])
    shifts = np.array(shifts)
    covariance = np.cov(shifts.T)
    return {
        "covariance_en_m2": covariance.tolist(),
        "ellipses": {"50": ellipse(covariance, 0.5), "90": ellipse(covariance, 0.9)},
        "resamples": resamples,
        "block_seconds": block_seconds,
        "shifts_en_m": shifts.tolist(),
    }
