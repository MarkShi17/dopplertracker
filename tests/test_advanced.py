import numpy as np
import pytest

from dopplertracker.advanced import (
    ModelOptions,
    fit_projected,
    predict_result,
    project_source,
    select_model,
    smooth_trajectory,
)
from dopplertracker.benchmark import synthetic_scene
from dopplertracker.physics import Trajectory
from dopplertracker.stress_scenes import stress_scene


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_source_projection_recovers_polynomial_without_geometry_optimization(degree):
    t = np.linspace(-60, 60, 121)
    factor = np.linspace(1.15, 0.85, len(t))
    source = 110 + (0.012 * t if degree >= 1 else 0) + (0.0002 * t * t if degree >= 2 else 0)
    coefficients, residual = project_source(
        factor, t, factor * source, np.ones(len(t)), degree, 110, prior=False
    )
    reconstructed = factor * np.polynomial.polynomial.polyval(t / 60, coefficients)
    np.testing.assert_allclose(reconstructed, factor * source, atol=1e-6)
    assert np.max(np.abs(residual)) < 1e-6


def test_robust_source_projection_resists_outliers():
    t = np.linspace(-60, 60, 121)
    factor = np.linspace(1.1, 0.9, len(t))
    observed = factor * (110 + 0.01 * t)
    observed[::8] += 8
    coefficients, _ = project_source(factor, t, observed, np.ones(len(t)), 1, 110, robust=True)
    assert abs(coefficients[0] - 110) < 0.1
    assert abs(coefficients[1] / 60 - 0.01) < 0.001


@pytest.mark.parametrize("method", ["spline", "kalman"])
def test_smoothing_reduces_velocity_noise_without_using_truth(method):
    rng = np.random.default_rng(19)
    t = np.arange(0.0, 202.0, 2.0)
    clean = np.column_stack((80 * t, t * 0 + 1500, t * 0 + 1000))
    noisy = Trajectory("noisy", t, clean + rng.normal(0, 15, clean.shape))
    smoothed = smooth_trajectory(noisy, method)
    true_velocity = np.array([80, 0, 0])
    before = np.mean((noisy.at(t[5:-5])[1] - true_velocity) ** 2)
    after = np.mean((smoothed.at(t[5:-5])[1] - true_velocity) ** 2)
    assert after < before * 0.35
    np.testing.assert_array_equal(smoothed.times, noisy.times)


@pytest.mark.parametrize("degree", [0, 1])
def test_projected_fit_recovers_clean_location(degree):
    scene = synthetic_scene("clean", seed=5)
    candidate = {
        "position": [0.0, 0.0, 0.0],
        "clock": 0.3,
        "flights": [t.id for t in scene["tracks"]],
        "f0": [e.f0 for e in scene["events"]],
    }
    result = fit_projected(
        scene["spectrogram"], scene["events"], scene["tracks"], candidate, ModelOptions(degree=degree)
    )
    assert result["success"]
    assert np.linalg.norm(np.array(result["position"])[:2] - scene["truth"][:2]) < 20


def test_result_prediction_keeps_source_reference_when_clock_changes():
    scene = synthetic_scene("rpm_drift", seed=5)
    candidate = {
        "position": [0.0, 0.0, 0.0],
        "clock": 0.7,
        "flights": [t.id for t in scene["tracks"]],
        "f0": [e.f0 for e in scene["events"]],
    }
    result = fit_projected(
        scene["spectrogram"], scene["events"], scene["tracks"], candidate, ModelOptions(degree=1)
    )
    assert np.linalg.norm(np.array(result["position"])[:2] - scene["truth"][:2]) < 15
    times = scene["events"][0].sample_times(scene["spectrogram"])
    f = predict_result(scene["tracks"][0], scene["events"][0], times, result, 0)
    assert np.isfinite(f).all()
    assert result["source_reference_utc"][0] == pytest.approx(candidate["clock"] + scene["events"][0].cpa)


def test_smoothing_never_bridges_missing_track_span():
    t = np.r_[np.arange(0.0, 20.0, 2.0), np.arange(100.0, 120.0, 2.0)]
    track = Trajectory("gap", t, np.column_stack((80 * t, t * 0, t * 0 + 1000)), max_gap=10)
    for method in ["spline", "kalman"]:
        smoothed = smooth_trajectory(track, method)
        assert not smoothed.valid(np.array([50.0]))[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"degree": 3},
        {"ridge": "unknown"},
        {"loss": "unknown"},
        {"smoothing": "unknown"},
        {"radius": np.nan},
        {"max_nfev": 0},
    ],
)
def test_model_options_reject_invalid_parameterizations(kwargs):
    with pytest.raises(ValueError):
        ModelOptions(**kwargs)


def test_blocked_validation_selects_simple_model_on_clean_signal():
    scene = stress_scene("clean", 0)
    result = select_model(scene["spectrogram"], scene["events"], scene["tracks"], scene["candidate"])
    assert result["options"]["degree"] == 0
    assert len(result["selection_trials"]) == 3
    assert all(len(r["fold_losses"]) == 3 for r in result["selection_trials"])


def test_model_selection_simplicity_does_not_depend_on_option_order():
    scene = stress_scene("clean", 0)
    result = select_model(
        scene["spectrogram"],
        scene["events"],
        scene["tracks"],
        scene["candidate"],
        options=[ModelOptions(degree=2), ModelOptions(degree=0), ModelOptions(degree=1)],
    )
    assert result["options"]["degree"] == 0


def test_validation_observations_are_not_used_to_fit_source_coefficients():
    scene = stress_scene("rpm_drift", 0)
    options = ModelOptions(degree=1)
    a = fit_projected(
        scene["spectrogram"], scene["events"], scene["tracks"], scene["candidate"], options, validation_fold=0
    )
    # Only held-out frame spectra are shifted; independent per-frame peak extraction
    # means train data and fitted geometry must remain unchanged.
    changed = scene["spectrogram"].power.copy()
    times = scene["spectrogram"].times
    for event in scene["events"]:
        held = (
            (times >= event.window[0])
            & (times <= event.window[1])
            & (np.floor((times - event.window[0]) / 15).astype(int) % 3 == 0)
        )
        changed[:, held] = np.roll(changed[:, held], 4, axis=0)
    from dopplertracker.signal import Spectrogram

    altered = Spectrogram(times, scene["spectrogram"].frequencies, changed)
    b = fit_projected(
        altered, scene["events"], scene["tracks"], scene["candidate"], options, validation_fold=0
    )
    np.testing.assert_allclose(a["position"], b["position"], atol=1e-7)
    assert b["validation_loss"] > a["validation_loss"]


def test_smoothed_result_prediction_uses_the_same_trajectory_as_fitting():
    scene = stress_scene("adsb_noise", 1)
    result = fit_projected(
        scene["spectrogram"],
        scene["events"],
        scene["tracks"],
        scene["candidate"],
        ModelOptions(degree=1, smoothing="spline"),
    )
    event, track = scene["events"][0], scene["tracks"][0]
    times = event.sample_times(scene["spectrogram"])
    expected = predict_result(
        smooth_trajectory(track, "spline"), event, times, {**result, "options": {"smoothing": "none"}}, 0
    )
    actual = predict_result(track, event, times, result, 0)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("method", ["spline", "kalman"])
def test_smoothing_improves_location_when_adsb_positions_are_noisy(method):
    scene = stress_scene("adsb_noise", 1)
    raw = fit_projected(
        scene["spectrogram"], scene["events"], scene["tracks"], scene["candidate"], ModelOptions(degree=1)
    )
    corrected = fit_projected(
        scene["spectrogram"],
        scene["events"],
        scene["tracks"],
        scene["candidate"],
        ModelOptions(degree=1, smoothing=method),
    )

    def error(result):
        return np.linalg.norm(np.array(result["position"])[:2] - scene["truth"][:2])

    assert error(corrected) < 0.5 * error(raw)


@pytest.mark.parametrize(
    "scenario,options",
    [
        ("altitude_bias", ModelOptions(degree=1, fit_altitude=True)),
        ("sound_speed", ModelOptions(degree=1, fit_sound_speed=True)),
        ("clock_drift", ModelOptions(degree=1, fit_clock_rate=True)),
    ],
)
def test_controlled_nuisance_corrections_reduce_model_bias(scenario, options):
    scene = stress_scene(scenario, 1)
    baseline = fit_projected(
        scene["spectrogram"], scene["events"], scene["tracks"], scene["candidate"], ModelOptions(degree=1)
    )
    corrected = fit_projected(
        scene["spectrogram"], scene["events"], scene["tracks"], scene["candidate"], options
    )

    def error(result):
        return np.linalg.norm(np.array(result["position"])[:2] - scene["truth"][:2])

    assert error(corrected) < error(baseline)


@pytest.mark.parametrize("method", ["spline", "kalman"])
@pytest.mark.parametrize("seed", [902, 903, 904])
@pytest.mark.parametrize("noise_m", [0.5, 5.0])
def test_irregular_sampling_does_not_masquerade_as_position_noise(method, seed, noise_m):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(1, 15, 200))
    positions = np.column_stack((90 * times, -30 * times + 1000, 2000 + 2 * times))
    positions += rng.normal(0, noise_m, positions.shape)
    track = Trajectory("irregular", times, positions, max_gap=20)
    result = smooth_trajectory(track, method)
    estimated = np.asarray(result.metadata["estimated_position_noise_m"])
    assert np.all(estimated < 1.5 * noise_m), estimated
    assert np.all(estimated > 0.5 * noise_m), estimated
