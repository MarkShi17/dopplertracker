"""Linearised error regions: geometry, replay of the fit, sensitivities, priors, what-if, CLI."""

import json
from dataclasses import replace

import numpy as np
import pytest

from dopplertracker.advanced import ModelOptions, fit_projected
from dopplertracker.physics import Trajectory
from dopplertracker.stress_scenes import stress_scene
from dopplertracker.uncertainty import (
    CHI2_2,
    GROUPS,
    InputPriors,
    _Model,
    _normalise_observations,
    _refit,
    _resolve_priors,
    _used_rows,
    ar1_correlation,
    block_bootstrap,
    ellipse,
    evaluate,
    hypothetical_event,
    mahalanobis,
    position_uncertainty,
    predicted_uncertainty,
)

SPLINE = ModelOptions(degree=1, smoothing="spline")


@pytest.fixture(scope="module")
def clean():
    return stress_scene("clean", 0)


@pytest.fixture(scope="module")
def clean_fit(clean):
    s = clean
    return fit_projected(
        s["spectrogram"], s["events"], s["tracks"], s["candidate"], SPLINE, keep_observations=True
    )


def test_ellipse_axes_bearing_and_mahalanobis():
    east_major = ellipse([[16.0, 0.0], [0.0, 1.0]], 0.9)
    assert east_major["semi_major_m"] == pytest.approx(4 * np.sqrt(CHI2_2[0.9]))
    assert east_major["semi_minor_m"] == pytest.approx(np.sqrt(CHI2_2[0.9]))
    assert east_major["major_axis_bearing_deg"] == pytest.approx(90.0)
    angle = np.radians(30.0)  # 30 degrees east of north
    u = np.array([np.sin(angle), np.cos(angle)])
    rotated = 25 * np.outer(u, u) + np.outer([u[1], -u[0]], [u[1], -u[0]])
    assert ellipse(rotated, 0.5)["major_axis_bearing_deg"] == pytest.approx(30.0)
    assert ellipse(rotated, 0.5)["semi_major_m"] == pytest.approx(5 * np.sqrt(CHI2_2[0.5]))
    assert mahalanobis(rotated, 5 * u) == pytest.approx(1.0)
    assert mahalanobis(rotated, [u[1], -u[0]]) == pytest.approx(1.0)


def test_ar1_correlation_recovers_known_process():
    rng = np.random.default_rng(3)
    x = np.zeros(4000)
    noise = rng.normal(size=4000)
    for k in range(1, 4000):
        x[k] = 0.7 * x[k - 1] + noise[k]
    times = np.arange(4000.0)
    assert ar1_correlation(times, x) == pytest.approx(0.7, abs=0.03)
    assert ar1_correlation(times, noise) < 0.05
    assert ar1_correlation(times[:50], (-1.0) ** np.arange(50)) == 0.0  # anticorrelation never shrinks


def test_model_replays_the_fit_prediction(clean, clean_fit):
    model = _Model(clean_fit, clean["events"], clean["tracks"])
    assert model.fitted == ["east", "north", "clock"]
    for i, o in enumerate(clean_fit["observations"]):
        valid = np.asarray(o["valid"])
        replay = model.prediction(i, np.asarray(o["times"]))
        np.testing.assert_allclose(replay[valid], np.asarray(o["model_hz"])[valid], atol=1e-6)


def test_model_replays_fused_velocity_with_nuisances():
    s = stress_scene("irregular_cadence", 0)
    options = ModelOptions(degree=1, velocity="fused", fit_track_offsets=True, fit_clock_rate=True)
    result = fit_projected(
        s["spectrogram"], s["events"], s["tracks"], s["candidate"], options, keep_observations=True
    )
    model = _Model(result, s["events"], s["tracks"])
    assert model.fitted[-4:] == ["clock_rate", "track_offset_1", "track_offset_2", "track_offset_3"]
    for i, o in enumerate(result["observations"]):
        valid = np.asarray(o["valid"])
        replay = model.prediction(i, np.asarray(o["times"]))
        np.testing.assert_allclose(replay[valid], np.asarray(o["model_hz"])[valid], atol=1e-5)


def test_hook_adds_region_without_changing_the_fit(clean, clean_fit):
    s = clean
    with_region = fit_projected(
        s["spectrogram"], s["events"], s["tracks"], s["candidate"], SPLINE, keep_observations=True,
        uncertainty=True,
    )  # fmt: skip
    region = with_region.pop("uncertainty")
    assert json.dumps(with_region, sort_keys=True) == json.dumps(clean_fit, sort_keys=True)
    assert "uncertainty" not in clean_fit
    assert set(region["breakdown"]) == set(GROUPS)
    total = np.asarray(region["covariance_en_m2"])
    parts = sum(np.asarray(region["breakdown"][g]["covariance_en_m2"]) for g in GROUPS)
    np.testing.assert_allclose(total, parts)
    assert sum(region["breakdown"][g]["fraction"] for g in GROUPS) == pytest.approx(1.0)
    assert region["ellipses"]["90"]["semi_major_m"] > region["ellipses"]["50"]["semi_major_m"] > 0
    # The saved observations give the same region as the fit's internals.
    again = position_uncertainty(clean_fit, s["events"], s["tracks"])
    np.testing.assert_allclose(again["covariance_en_m2"], region["covariance_en_m2"], rtol=1e-6)
    checked = evaluate(region, np.asarray(clean_fit["position"])[:2] - s["truth"][:2])
    assert checked["total"]["inside_90"]


def test_considered_sensitivity_predicts_a_refit(clean):
    """A 0.1 s ADS-B time-label error on flight 3 moves a (linear-loss) refit as predicted."""
    s = clean
    options = replace(SPLINE, loss="linear", robust_source=False)
    base = fit_projected(
        s["spectrogram"], s["events"], s["tracks"], s["candidate"], options, uncertainty=True
    )
    row = base["uncertainty"]["considered_inputs"]["track_offset_2"]
    tracks = list(s["tracks"])
    t = tracks[2]
    tracks[2] = Trajectory(t.id, t.times + 0.1, t.positions, t.max_gap, t.metadata)
    shifted = fit_projected(s["spectrogram"], s["events"], tracks, s["candidate"], options)
    actual = np.asarray(shifted["position"][:2]) - np.asarray(base["position"][:2])
    predicted = -np.asarray(row["position_shift_en_m"]) / row["prior_sigma"] * 0.1
    assert np.linalg.norm(predicted) > 1.0
    assert np.linalg.norm(actual - predicted) < 0.1 * np.linalg.norm(predicted)


def test_fitted_inputs_are_not_considered_twice(clean):
    s = clean
    options = replace(SPLINE, fit_sound_speed=True, fit_wind=True, fit_track_offsets=True)
    result = fit_projected(
        s["spectrogram"], s["events"], s["tracks"], s["candidate"], options, uncertainty=True
    )
    region = result["uncertainty"]
    considered = region["considered_inputs"]
    assert not {"sound_speed", "wind_e", "wind_n"} & set(considered)
    assert [k for k in considered if k.startswith("track_offset")] == ["track_offset_0"]
    assert {"sound_speed", "wind_e", "wind_n", "track_offset_1"} <= set(region["fitted_parameters"])
    # The fitted atmosphere's own uncertainty (data plus its prior) still shows under its cause.
    assert region["breakdown"]["atmosphere"]["variance_m2"] > 0


def test_prior_defaults_follow_the_inputs(clean):
    tracks = clean["tracks"]
    baro = Trajectory(
        "baro",
        tracks[0].times,
        tracks[0].positions,
        metadata={"altitude_sources": ["corrected_baro_approximation"]},
    )
    heard = [t.times[len(t.times) // 2] for t in (baro, *tracks[1:])]
    constant = _resolve_priors(None, {}, [baro, *tracks[1:]], SPLINE, None, heard)
    measured = _resolve_priors(None, {"atmosphere": {"type": "layered"}}, tracks, SPLINE, None, heard)
    assert constant["sound_speed"][0] == 6.0 and measured["sound_speed"][0] == 1.0
    assert constant["wind_e"][0] == 8.0 and measured["wind_n"][0] == 2.5
    assert constant["aircraft_altitude_0"][0] == 60.0 and constant["aircraft_altitude_1"][0] == 20.0
    assert constant["observer_height"][0] == 10.0 and constant["track_offset_3"][0] == 0.5
    overridden = _resolve_priors(
        InputPriors(velocity_m_s=1.5, observer_height_m=0.0), {}, tracks, SPLINE, None, heard
    )
    assert overridden["velocity_2_u"][0] == 1.5 and "observer_height" not in overridden
    # Reported velocity is a tighter velocity input than differenced positions.
    rng = np.random.default_rng(1)
    noisy = Trajectory("v", tracks[0].times, tracks[0].positions + rng.normal(0, 30.0, tracks[0].positions.shape),
                       velocities=np.gradient(tracks[0].positions, tracks[0].times, axis=0))  # fmt: skip
    fused = _resolve_priors(
        None, {}, [noisy], replace(SPLINE, smoothing="none", velocity="fused"), None, heard
    )
    differenced = _resolve_priors(None, {}, [noisy], SPLINE, None, heard)
    assert fused["velocity_0_e"][0] == 0.2 < differenced["velocity_0_e"][0]
    with pytest.raises(ValueError):
        InputPriors(wind_m_s=-1.0)


def test_what_if_a_third_flyover_shrinks_the_measurement_part(clean):
    s = clean
    position, clock = s["truth"], s["clock"]
    f0 = [e.f0 for e in s["events"]]
    two = predicted_uncertainty(s["events"][:2], s["tracks"][:2], position, clock, f0[:2])
    third = hypothetical_event(s["tracks"][2], position, clock, f0[2])
    assert abs(third.cpa - s["events"][2].cpa) < 2.0
    three = predicted_uncertainty([*s["events"][:2], third], s["tracks"][:3], position, clock, f0[:3])
    noisy = predicted_uncertainty(s["events"][:2], s["tracks"][:2], position, clock, f0[:2], noise_hz=0.6)

    def measured(region):
        return region["breakdown"]["measurement"]["variance_m2"]

    assert measured(three) < 0.5 * measured(two)
    assert 3.0 < measured(noisy) / measured(two) < 4.1  # measurement variance ~ noise^2
    assert three["ellipses"]["90"]["semi_major_m"] < two["ellipses"]["90"]["semi_major_m"]


def test_linearised_measurement_part_matches_monte_carlo(clean, clean_fit):
    """Correlated Gaussian ridge noise: refit scatter agrees with the AR(1) sandwich."""
    s = clean
    model = _Model(clean_fit, s["events"], s["tracks"])
    base = _normalise_observations(None, clean_fit)
    used = _used_rows(model, base)
    rng = np.random.default_rng(11)
    rho, sigma = 0.6, 0.4
    shifts, predicted = [], []
    for draw in range(20):
        observations = []
        for i, (times, _, mask, weights) in enumerate(base):
            e = np.zeros(len(times))
            e[0] = rng.normal()
            for k in range(1, len(times)):
                e[k] = rho * e[k - 1] + np.sqrt(1 - rho * rho) * rng.normal()
            observations.append((times, model.prediction(i, times) + sigma * e, mask, weights))
        shifts.append(_refit(model, observations, used)[:2])
        if draw < 5:
            region = position_uncertainty(
                clean_fit,
                s["events"],
                s["tracks"],
                priors=InputPriors(0, 0, 0, 0, 0, 0),
                observations=observations,
            )
            predicted.append(np.trace(region["breakdown"]["measurement"]["covariance_en_m2"]))
    empirical = np.trace(np.cov(np.array(shifts).T))
    assert 0.5 < np.mean(predicted) / empirical < 2.0


def test_block_bootstrap_runs_and_is_finite(clean, clean_fit):
    s = clean
    boot = block_bootstrap(clean_fit, s["events"], s["tracks"], resamples=8)
    assert np.isfinite(boot["covariance_en_m2"]).all() and len(boot["shifts_en_m"]) == 8


def test_excluded_event_contributes_no_information(clean):
    s = clean
    args = (s["spectrogram"], s["events"], s["tracks"], s["candidate"], SPLINE)
    full = fit_projected(*args, uncertainty=True)["uncertainty"]
    held = fit_projected(*args, exclude_event=1, uncertainty=True)["uncertainty"]
    assert held["events"][1]["frames"] == 0
    assert np.isfinite(held["covariance_en_m2"]).all()
    measured = [r["breakdown"]["measurement"]["variance_m2"] for r in (full, held)]
    assert measured[1] > measured[0]


def test_cli_refine_reports_uncertainty(clean, tmp_path, capsys):
    from dopplertracker import cli
    from dopplertracker.data import save_tracks
    from dopplertracker.detection import save_events
    from dopplertracker.geo import LocalFrame

    s = clean
    s["spectrogram"].save(tmp_path / "spec.npz")
    save_events(s["events"], tmp_path / "events.json")
    save_tracks(tmp_path / "traces.npz", s["tracks"], LocalFrame(40.85, -73.5))
    (tmp_path / "results.json").write_text(
        json.dumps({"frame": [40.85, -73.5], "candidates": [s["candidate"]]})
    )
    base = [
        "refine", str(tmp_path / "spec.npz"), str(tmp_path / "results.json"),
        "--events", str(tmp_path / "events.json"), "--traces", str(tmp_path / "traces.npz"),
        "--raw-score", "--elevation", "0", "--out", str(tmp_path / "refined.json"), "--uncertainty",
    ]  # fmt: skip
    assert cli.main([*base, "--method", "brightness"]) == 2
    assert "--uncertainty requires --method projected" in capsys.readouterr().err
    assert cli.main([*base, "--method", "projected", "--source-degree", "1", "--smooth", "spline"]) == 0
    assert "90 % region" in capsys.readouterr().err
    refined = json.loads((tmp_path / "refined.json").read_text())["candidates"][0]
    region = refined["uncertainty"]
    assert region["considered_inputs"]["observer_height"]["prior_sigma"] == 10.0
    assert (
        mahalanobis(region["covariance_en_m2"], np.asarray(refined["position"])[:2] - s["truth"][:2]) < 2.15
    )
