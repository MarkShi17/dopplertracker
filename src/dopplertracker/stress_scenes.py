"""Independent, deliberately imperfect scenes for acoustic geolocation stress tests.

This module is a data generator, not an alternate implementation of the scorer: it
uses analytic curved paths and forward retarded-time propagation only.  In
particular it intentionally does not call ``predict``, ``closest_approaches`` or
``ideal_curve``.
"""

import numpy as np

from .detection import Event
from .physics import Trajectory
from .signal import Spectrogram

STRESS_SCENARIOS = (
    "clean",
    "curved_accelerating",
    "rpm_drift",
    "nonlinear_rpm",
    "wind",
    "sound_speed",
    "adsb_noise",
    "altitude_bias",
    "clock_drift",
    "gaps",
    "low_snr",
    "broadband_interference",
    "crossing_ridge",
    "missing_harmonic",
    "corrupted_event",
    "overhead",
    "combined_errors",
    "noise_only",
    "parallel_tracks",
    "mismeasured_event",
)


def _parameters(name):
    """Explicit, controlled nuisance settings; zero values mean the clean control."""
    p = dict(
        curvature=0.0,
        acceleration=0.0,
        rpm_drift=0.0,
        rpm_nonlinear=0.0,
        wind=(0.0, 0.0, 0.0),
        sound_speed=343.0,
        adsb_position_noise_m=0.0,
        altitude_bias_m=0.0,
        sample_rate_error=0.0,
        gap_seconds=0.0,
        ridge_snr=8.0,
        broadband=False,
        crossing=False,
        missing_harmonic=False,
        corrupt_event=False,
        mismeasured_event=False,
        overhead=False,
        noise_only=False,
    )
    changes = {
        "curved_accelerating": dict(curvature=0.018, acceleration=0.18),
        "rpm_drift": dict(rpm_drift=0.035),
        "nonlinear_rpm": dict(rpm_nonlinear=0.65),
        "wind": dict(wind=(13.0, -8.0, 0.0)),
        "sound_speed": dict(sound_speed=331.0),
        "adsb_noise": dict(adsb_position_noise_m=45.0),
        "altitude_bias": dict(altitude_bias_m=180.0),
        "clock_drift": dict(sample_rate_error=0.0018),
        "gaps": dict(gap_seconds=42.0),
        "low_snr": dict(ridge_snr=1.6),
        "broadband_interference": dict(broadband=True, ridge_snr=4.0),
        "crossing_ridge": dict(crossing=True, ridge_snr=5.0),
        "missing_harmonic": dict(missing_harmonic=True),
        "corrupted_event": dict(corrupt_event=True),
        "overhead": dict(overhead=True),
        "combined_errors": dict(
            curvature=0.014,
            acceleration=0.12,
            rpm_drift=0.025,
            wind=(10.0, -7.0, 0.0),
            adsb_position_noise_m=28.0,
            altitude_bias_m=110.0,
            sample_rate_error=0.001,
            gap_seconds=28.0,
            ridge_snr=3.0,
        ),
        "noise_only": dict(noise_only=True, ridge_snr=0.0),
        "parallel_tracks": dict(parallel_tracks=True),
        "mismeasured_event": dict(mismeasured_event=True),
    }
    if name not in STRESS_SCENARIOS:
        raise ValueError(f"unknown stress scenario {name!r}")
    p.update(changes.get(name, {}))
    return p


def _wind_travel(source_minus_observer, c, wind):
    """Forward travel time in a uniform moving medium, solved as a quadratic."""
    r = np.asarray(source_minus_observer, float)
    w = np.asarray(wind, float)
    a = c * c - np.dot(w, w)
    rw = r @ w
    return (rw + np.sqrt(rw * rw + a * np.sum(r * r, axis=-1))) / a


def _path(s, observer, angle, distance, altitude, speed, climb, curvature, acceleration, overhead):
    """Curved, accelerating, climbing ENU path at emission seconds relative to CPA."""
    forward = np.array([np.cos(angle), np.sin(angle), 0.0])
    side = np.array([-np.sin(angle), np.cos(angle), 0.0])
    if overhead:
        distance = 18.0
    along = speed * s + 0.5 * acceleration * s * s
    cross = distance + curvature * s * s
    z = altitude + climb * s + 0.0015 * s * s
    return (
        observer
        + along[..., None] * forward
        + cross[..., None] * side
        + np.stack((np.zeros_like(s), np.zeros_like(s), z), axis=-1)
    )


def stress_scene(scenario="clean", seed=0):
    """Return one reproducible, independent synthetic stress-test experiment.

    The ``candidate`` is intentionally fixed away from the randomized truth.  The
    metadata documents generator-only nuisance truth and must not be passed to an
    inference method as candidate parameters.
    """
    p = _parameters(scenario)
    rng = np.random.default_rng(seed)
    clock = 12345.6
    truth = np.array([rng.uniform(-250, 250), rng.uniform(-250, 250), 0.0])
    # 1,300 receive frames and 1,301 frequency bins: useful but still small.
    # Spectrogram/event time is recording-relative audio time.  Tracks remain
    # UTC, and callers add ``clock`` when predicting against them.
    times = np.arange(1300) * 0.5
    actual_receive = clock + times * (1.0 + p["sample_rate_error"])
    frequencies = 45.0 + np.arange(1301) * 0.25
    power = rng.exponential(0.20, (len(frequencies), len(times))).astype(np.float32)
    centers = np.array([80.0, 250.0, 420.0, 590.0])
    tracks, events, flight_metadata = [], [], []
    angles = (
        np.array([0.15, 0.22, 0.10, 0.18]) if p.get("parallel_tracks") else np.array([0.15, 1.72, 3.30, 4.90])
    )
    f0s = np.array([108.0, 126.0, 94.0, 145.0])
    for i, center in enumerate(centers):
        local = np.arange(-105.0, 106.0, 0.10)
        distance = 900.0 + 230.0 * i
        altitude = 1150.0 + 260.0 * i
        if p["overhead"]:
            altitude = 4500.0 + 260.0 * i
        speed, climb = 105.0 + 9 * i, (-2.0 + i * 1.4)
        overhead = p["overhead"]
        pos = _path(
            local,
            truth,
            angles[i],
            distance,
            altitude,
            speed,
            climb,
            p["curvature"],
            p["acceleration"],
            overhead,
        )
        tau = _wind_travel(pos - truth, p["sound_speed"], p["wind"])
        arrival = clock + center + local + tau
        # Source phase differentiation makes the observed frequency depend on
        # both non-linear RPM and the genuinely non-linear arrival-time map.
        phase = f0s[i] * local + 0.5 * p["rpm_drift"] * local**2 + p["rpm_nonlinear"] * local**3 / 1800.0
        observed_hz = np.gradient(phase, arrival)
        cpa_index = int(np.argmin(np.linalg.norm(pos - truth, axis=1)))
        cpa_receive = float(arrival[cpa_index])
        cpa = (cpa_receive - clock) / (1.0 + p["sample_rate_error"])
        valid = (actual_receive >= arrival[0]) & (actual_receive <= arrival[-1])
        curve = np.interp(actual_receive, arrival, observed_hz, left=np.nan, right=np.nan)
        # Avoid NaNs outside the finite emission-time support before building the
        # full spectrogram grid (multiplication by False would not remove NaN).
        curve_for_grid = np.nan_to_num(curve, nan=-1.0e6)
        if not p["noise_only"]:
            render_curve = curve_for_grid
            # Keep the ADS-B flight and its event seed, but substitute an
            # unrelated observed ridge.  Robust algorithms should omit this.
            if p["corrupt_event"] and i == 2:
                render_curve = 72.0 + 0.42 * (times - cpa) + 5.0 * np.sin((times - cpa) / 9)
            for harmonic in (1, 2):
                if harmonic == 2 and p["missing_harmonic"] and i == 1:
                    continue
                width = 0.38 + 0.04 * harmonic
                power += (
                    p["ridge_snr"]
                    / harmonic
                    * np.exp(-0.5 * ((frequencies[:, None] - harmonic * render_curve[None, :]) / width) ** 2)
                    * valid
                ).astype(np.float32)
        # Sparse ADS-B-like observations are deliberately not the source path.
        sample_rel = np.arange(-100.0, 101.0, 4.0)
        if p["gap_seconds"]:
            sample_rel = sample_rel[np.abs(sample_rel - 15.0) > p["gap_seconds"] / 2]
        observed = _path(
            sample_rel,
            truth,
            angles[i],
            distance,
            altitude,
            speed,
            climb,
            p["curvature"],
            p["acceleration"],
            overhead,
        )
        observed += rng.normal(0, p["adsb_position_noise_m"], observed.shape)
        observed[:, 2] += p["altitude_bias_m"]
        tracks.append(
            Trajectory(
                f"stress-{i}",
                clock + center + sample_rel,
                observed,
                max_gap=15.0,
                metadata={"synthetic": True},
            )
        )
        event_f0 = f0s[i] + rng.normal(0, 0.35)
        event_cpa = cpa + rng.normal(0, 0.35)
        if p["mismeasured_event"] and i == 2:
            event_cpa += 3.5
            event_f0 += 6.0
        event = Event(
            (cpa - 45.0, cpa + 45.0),
            event_cpa,
            event_f0,
            speed,
            float(np.linalg.norm(pos[cpa_index] - truth)),
            2,
        )
        events.append(event)
        flight_metadata.append(
            dict(
                id=f"stress-{i}",
                center_s=float(center),
                speed_mps=float(speed),
                climb_mps=float(climb),
                source_f0_hz=float(f0s[i]),
                true_cpa_audio_s=float(cpa),
            )
        )
    if p["broadband"]:
        busy = rng.random(len(times)) < 0.16
        power[:, busy] += rng.uniform(1.5, 5.5, busy.sum())
    if p["crossing"]:
        wrong = 175 + 0.12 * (times - 330) ** 2 / 20
        power += (5.0 * np.exp(-0.5 * ((frequencies[:, None] - wrong) / 0.65) ** 2)).astype(np.float32)
    candidate = dict(
        position=[0.0, 0.0, 0.0],
        clock=float(clock + rng.normal(0, 1.2)),
        f0=[float(e.f0 + rng.normal(0, 0.4)) for e in events],
        flights=[t.id for t in tracks],
    )
    metadata = dict(
        parameters=p,
        flights=flight_metadata,
        corrupted_event=(2 if p["corrupt_event"] else None),
        note="Generator-only nuisance truth; never supply it to candidate fitting.",
    )
    return dict(
        truth=truth,
        clock=clock,
        tracks=tracks,
        events=events,
        spectrogram=Spectrogram(times, frequencies, power),
        scenario=scenario,
        seed=seed,
        atmosphere_truth=dict(c=p["sound_speed"], wind=p["wind"]),
        candidate=candidate,
        metadata=metadata,
    )
