import json

import numpy as np
import pytest

from dopplertracker.advanced import ModelOptions, fit_projected, predict_result
from dopplertracker.atmosphere import (
    FlightAtmospheres,
    LayeredAtmosphere,
    WeatherSeries,
    atmosphere_from_record,
    sound_speed,
    wind_components,
)
from dopplertracker.evidence import assess_candidate
from dopplertracker.physics import Atmosphere, Trajectory, closest_approaches, predict, travel_time
from dopplertracker.stress_scenes import (
    EXTENDED_SCENARIOS,
    STRESS_SCENARIOS,
    layered_profiles,
    stress_scene,
    trace_ray_times,
)

SHEARED = LayeredAtmosphere.from_lapse_rate(
    15.0, 6.5, 0.0, 10000.0, wind_surface=(2.0, 0.0), wind_top=(14.0, 14.3), wind_top_height_m=6000.0
)


def _geometry(n, low_elevation=15.0, seed=3):
    rng = np.random.default_rng(seed)
    height = rng.uniform(300, 6000, n)
    elevation = np.radians(rng.uniform(low_elevation, 85, n))
    azimuth = rng.uniform(0, 2 * np.pi, n)
    horizontal = height / np.tan(elevation)
    return np.stack((horizontal * np.cos(azimuth), horizontal * np.sin(azimuth), height), axis=1)


@pytest.fixture(scope="module")
def layered_scene():
    return stress_scene("layered_atmosphere", 0)


def test_sound_speed_from_temperature_and_humidity():
    assert sound_speed(20.0) == pytest.approx(343.2, abs=0.1)
    assert sound_speed(20.0, 80.0, 1013.25) > sound_speed(20.0) + 0.4
    east, north = wind_components([5.0, 5.0], [0.0, 270.0])  # from north, from west
    assert np.allclose(east, [0.0, 5.0], atol=1e-12) and np.allclose(north, [-5.0, 0.0], atol=1e-12)


def test_uniform_profile_reproduces_homogeneous_closed_form():
    layered = LayeredAtmosphere([0.0, 3000.0], [340.0, 340.0], [12.0, 12.0], [-7.0, -7.0])
    r = _geometry(60, low_elevation=3.0)
    tau, gradient, solved = layered.ray_travel_time(r, 0.0)
    expected_tau, expected_gradient = travel_time(r, Atmosphere(340.0, (12.0, -7.0, 0.0)))
    assert solved.all()
    assert np.max(np.abs(tau - expected_tau)) < 1e-9
    assert np.max(np.abs(gradient - expected_gradient)) < 1e-12
    shifted = LayeredAtmosphere([0.0, 3000.0], [338.0, 338.0], [10.0, 10.0], [-4.0, -4.0]).offset(
        2.0, (2.0, -3.0)
    )
    assert np.max(np.abs(shifted.travel_time(r, 0.0)[0] - expected_tau)) < 1e-9


def test_exact_solver_matches_independent_ray_tracer_and_own_gradient():
    r = _geometry(12)
    observer = np.array([0.0, 0.0, 0.0])
    truth = trace_ray_times(
        SHEARED.heights, SHEARED.sound_speed, SHEARED.wind_east, SHEARED.wind_north, r + observer, observer
    )
    tau, gradient, solved = SHEARED.ray_travel_time(r, 0.0)
    assert solved.all() and np.isfinite(truth).all()
    assert np.max(np.abs(tau - truth)) < 1e-5
    step = 0.05
    numeric = np.stack(
        [
            (SHEARED.travel_time(r + step * u, 0.0)[0] - SHEARED.travel_time(r - step * u, 0.0)[0])
            / (2 * step)
            for u in np.eye(3)
        ],
        axis=1,
    )
    # |error in grad . v| for |v| <= 150 m/s
    assert np.max(150 * np.linalg.norm(gradient - numeric, axis=1)) < 3e-5
    # The straight-path (Fermat) form is cheaper but far outside the 2 ms requirement here.
    straight, _ = SHEARED.straight_travel_time(r, 0.0)
    assert np.max(np.abs(straight - truth)) > 2e-3


def test_straight_path_gradient_is_consistent():
    r = _geometry(10)
    tau, gradient = SHEARED.straight_travel_time(r, 25.0)
    step = 0.05
    numeric = np.stack(
        [
            (
                SHEARED.straight_travel_time(r + step * u, 25.0)[0]
                - SHEARED.straight_travel_time(r - step * u, 25.0)[0]
            )
            / (2 * step)
            for u in np.eye(3)
        ],
        axis=1,
    )
    assert np.max(np.abs(gradient - numeric)) < 1e-9


def test_shadow_zone_falls_back_to_straight_path():
    # 300 m source 30 km away under a strong upward-refracting lapse: no direct ray.
    r = np.array([[30000.0, 0.0, 300.0]])
    tau, gradient, solved = SHEARED.ray_travel_time(r, 0.0)
    assert not solved[0] and np.isfinite(tau).all() and np.isfinite(gradient).all()
    assert tau[0] == pytest.approx(SHEARED.straight_travel_time(r, 0.0)[0][0])
    # Off-track (NaN) positions stay NaN without disturbing other rows.
    mixed, _, _ = SHEARED.ray_travel_time(np.r_[r, [[np.nan] * 3], [[500.0, 0.0, 2000.0]]], 0.0)
    assert mixed[0] == pytest.approx(tau[0]) and np.isnan(mixed[1]) and np.isfinite(mixed[2])


def test_predict_dispatches_layered_and_per_flight_atmospheres():
    times = np.arange(-60.0, 61.0, 2.0)
    track = Trajectory(
        "a", times, np.c_[110 * times, np.full_like(times, 900.0), np.full_like(times, 1500.0)]
    )
    observer = np.array([30.0, -20.0, 5.0])
    uniform = LayeredAtmosphere([0.0, 3000.0], [340.0, 340.0], [6.0, 6.0], [-3.0, -3.0])
    receive = np.linspace(-40, 50, 31)
    expected, _ = predict(track, observer, receive, 120.0, Atmosphere(340.0, (6.0, -3.0, 0.0)))
    actual, _ = predict(track, observer, receive, 120.0, uniform)
    assert np.nanmax(np.abs(actual - expected)) < 1e-8
    per_flight = FlightAtmospheres({"a": uniform}, SHEARED)
    assert np.array_equal(predict(track, observer, receive, 120.0, per_flight)[0], actual, equal_nan=True)
    assert closest_approaches(track, observer, per_flight)[0][0] == pytest.approx(
        closest_approaches(track, observer, Atmosphere(340.0, (6.0, -3.0, 0.0)))[0][0], abs=1e-6
    )
    with pytest.raises(ValueError):
        travel_time(np.array([1.0, 2.0, 300.0]), uniform)


def test_records_round_trip():
    per_flight = FlightAtmospheres({"x": SHEARED.offset(1.5, (2.0, -1.0))}, SHEARED)
    record = json.loads(json.dumps(per_flight.to_record()))
    restored = atmosphere_from_record(record)
    assert restored.for_flight("x") == per_flight.for_flight("x")
    assert restored.for_flight("other") == SHEARED
    assert restored.for_flight("x").c == pytest.approx(SHEARED.c + 1.5)


def _open_meteo_response():
    hours = ["2026-09-08T20:00", "2026-09-08T21:00"]
    hourly = {
        "time": hours,
        "temperature_2m": [22.0, 21.0],
        "relative_humidity_2m": [60.0, 62.0],
        "wind_speed_10m": [2.0, 2.0],
        "wind_direction_10m": [270.0, 270.0],
        "surface_pressure": [1010.0, 1010.0],
    }
    for level, height, temperature in ((1000, 60.0, 21.0), (925, 780.0, 16.0), (850, 1500.0, 12.0)):
        hourly.update(
            {
                f"temperature_{level}hPa": [temperature, temperature - 1.0],
                f"relative_humidity_{level}hPa": [50.0, 50.0],
                f"wind_speed_{level}hPa": [5.0, 7.0],
                f"wind_direction_{level}hPa": [0.0, 0.0],
                f"geopotential_height_{level}hPa": [height, height + 10.0],
            }
        )
    return {"latitude": 40.85, "longitude": -73.5, "elevation": 93.0, "hourly": hourly}


def test_open_meteo_profiles_in_ellipsoid_heights_and_time_interpolation(tmp_path):
    series = WeatherSeries.from_open_meteo(_open_meteo_response(), geoid_offset=-30.854)
    # 1000 hPa (60 m) lies below the 95 m surface level and is dropped.
    assert series.metadata["pressure_levels_hpa"] == [925, 850]
    first = series.profiles[0]
    assert first.heights[0] == pytest.approx(95.0 - 30.854)
    assert first.wind_east[0] == pytest.approx(2.0)  # from the west blows east
    assert first.wind_north[1] == pytest.approx(-5.0)  # from the north blows south
    midway = series.at(series.times[0] + 1800.0)
    assert midway.heights[1] == pytest.approx(780.0 + 5.0 - 30.854)
    assert midway.wind_north[1] == pytest.approx(-6.0)
    series.save(tmp_path / "w.json")
    assert (
        WeatherSeries.load(tmp_path / "w.json").at(series.times[1]).sound_speed
        == series.profiles[1].sound_speed
    )
    with pytest.raises(ValueError):
        series.at(series.times[0] - 7300.0)


def test_extended_scenarios_are_separate_and_use_independent_truth(layered_scene):
    assert not set(EXTENDED_SCENARIOS) & set(STRESS_SCENARIOS)
    truth, forecast = layered_profiles(0)
    assert layered_scene["atmosphere_profile"] == truth and layered_scene["atmosphere_forecast"] == forecast
    assert forecast.sound_speed[0] > truth.sound_speed[0] + 0.5
    assert layered_scene["metadata"]["atmosphere_profile"]["type"] == "layered"
    assert np.hypot(truth.wind_east[-1], truth.wind_north[-1]) == pytest.approx(20.0)


def test_layered_fit_records_atmosphere_and_evidence_repredicts(layered_scene):
    scene = layered_scene
    args = (scene["spectrogram"], scene["events"], scene["tracks"], scene["candidate"])
    options = ModelOptions(degree=1, smoothing="spline")
    homogeneous = fit_projected(*args, options)
    layered = fit_projected(*args, options, scene["atmosphere_profile"])
    error = [
        np.linalg.norm(np.asarray(r["position"])[:2] - scene["truth"][:2]) for r in (homogeneous, layered)
    ]
    assert error[1] < 5.0 and error[1] < error[0]
    assert "atmosphere" not in homogeneous and layered["atmosphere"]["type"] == "layered"
    assert layered["sound_speed_m_s"] == pytest.approx(scene["atmosphere_profile"].c)
    evidence = assess_candidate(scene["spectrogram"], scene["events"], scene["tracks"], layered)
    assert evidence["accepted"]
    # predict_result interprets the saved record the same way the fit did.
    times = scene["events"][0].sample_times(scene["spectrogram"])
    curve = predict_result(scene["tracks"][0], scene["events"][0], times, layered, 0)
    flat = predict_result(scene["tracks"][0], scene["events"][0], times, {**layered, "atmosphere": None}, 0)
    assert np.nanmax(np.abs(curve - flat)) > 0.05


def test_layered_nuisance_fit_is_an_offset(layered_scene):
    scene = layered_scene
    options = ModelOptions(degree=1, smoothing="spline", fit_sound_speed=True, fit_wind=True, max_nfev=40)
    result = fit_projected(
        scene["spectrogram"],
        scene["events"],
        scene["tracks"],
        scene["candidate"],
        options,
        scene["atmosphere_forecast"],
    )
    record = result["atmosphere"]
    assert record["sound_speed_m_s"] == list(scene["atmosphere_forecast"].sound_speed)
    assert abs(record["sound_speed_offset_m_s"]) <= 18 and np.all(np.abs(record["wind_offset_en_m_s"]) <= 30)
    restored = atmosphere_from_record(record)
    assert restored.c == pytest.approx(result["sound_speed_m_s"])


def test_cli_fetch_weather_refine_and_validate(layered_scene, tmp_path, monkeypatch, capsys):
    from dopplertracker import cli
    from dopplertracker.data import save_tracks
    from dopplertracker.detection import save_events
    from dopplertracker.geo import LocalFrame

    monkeypatch.setattr(
        cli, "fetch_open_meteo", lambda date, lat, lon: ("https://example", _open_meteo_response())
    )
    assert (
        cli.main(
            ["fetch-weather", "2026-09-08", str(tmp_path / "w.json"), "--lat", "40.85", "--lon", "-73.5"]
        )
        == 0
    )
    assert (tmp_path / "w.raw.json").exists() and WeatherSeries.load(tmp_path / "w.json").metadata["url"]

    scene = layered_scene
    profile = scene["atmosphere_profile"]
    WeatherSeries([0.0, 86400.0], [profile, profile], {"source": "test"}).save(
        tmp_path / "scene-weather.json"
    )
    scene["spectrogram"].save(tmp_path / "spec.npz")
    save_events(scene["events"], tmp_path / "events.json")
    save_tracks(tmp_path / "traces.npz", scene["tracks"], LocalFrame(40.85, -73.5))
    (tmp_path / "results.json").write_text(
        json.dumps({"frame": [40.85, -73.5], "candidates": [scene["candidate"]]})
    )
    common = [
        "--events",
        str(tmp_path / "events.json"),
        "--traces",
        str(tmp_path / "traces.npz"),
        "--raw-score",
    ]
    refine = [
        "refine",
        str(tmp_path / "spec.npz"),
        str(tmp_path / "results.json"),
        *common,
        "--elevation",
        "0",
    ]
    weather = ["--weather", str(tmp_path / "scene-weather.json")]
    out = str(tmp_path / "refined.json")
    assert cli.main([*refine, "--method", "projected", "--sound-speed", "340", *weather, "--out", out]) == 2
    assert "replaces" in capsys.readouterr().err
    smooth = ["--source-degree", "1", "--smooth", "spline"]
    assert cli.main([*refine, "--method", "projected", *smooth, *weather, "--out", out]) == 0
    refined = json.loads((tmp_path / "refined.json").read_text())["candidates"][0]
    assert refined["atmosphere"]["type"] == "per_flight"
    assert np.linalg.norm(np.asarray(refined["position"])[:2] - scene["truth"][:2]) < 10.0
    validate = ["validate", str(tmp_path / "spec.npz"), out, *common, "--out", str(tmp_path / "v.json")]
    assert cli.main(validate) == 0
    assert json.loads((tmp_path / "v.json").read_text())["candidates"][0]["evidence"]["accepted"]
