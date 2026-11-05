import numpy as np
import pytest

from dopplertracker.geo import LocalFrame
from dopplertracker.physics import Atmosphere, Trajectory, closest_approaches, predict, travel_time


def test_retarded_time_matches_independent_straight_pass_solution():
    t = np.arange(-200.0, 201.0, 2.0)
    track = Trajectory(
        "plane", t, np.column_stack((80 * t, np.full(t.size, 1500.0), np.full(t.size, 1000.0)))
    )
    receive = np.linspace(-70, 80, 100)
    c, v, d = 343.0, 80.0, np.hypot(1500, 1000)
    emit = (c * c * receive - np.sqrt(c * c * v * v * receive**2 + (c * c - v * v) * d * d)) / (c * c - v * v)
    expected = 120 / (1 + v * v * emit / (c * np.sqrt(d * d + v * v * emit**2)))
    actual, emission = predict(track, np.zeros(3), receive, 120.0)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_allclose(emission, emit, atol=1e-7)


def test_uniform_wind_travel_time_and_frequency_derivative():
    atmosphere = Atmosphere(338.0, (12.0, -5.0, 0.0))
    r = np.array([3000.0, 1000.0, 500.0])
    tau, gradient = travel_time(r, atmosphere)
    np.testing.assert_allclose(np.linalg.norm(r + np.array(atmosphere.wind) * tau), atmosphere.c * tau)
    for i in range(3):
        offset = np.eye(3)[i] * 0.01
        numeric = (travel_time(r + offset, atmosphere)[0] - travel_time(r - offset, atmosphere)[0]) / 0.02
        assert gradient[i] == pytest.approx(numeric, abs=1e-9)


def test_no_extrapolation_or_gap_interpolation():
    t = np.array([0.0, 2.0, 4.0, 100.0, 102.0, 104.0])
    track = Trajectory("gaps", t, np.column_stack((t * 60, t * 0, t * 0 + 1000)), max_gap=10)
    freq, _ = predict(track, np.zeros(3), np.array([-10.0, 50.0, 200.0]), 100)
    assert np.isnan(freq).all()


def test_geographic_round_trip():
    frame = LocalFrame(40.8, -73.5)
    points = np.array([[40.81, -73.49, 25.0], [40.82, -73.53, 3000.0]])
    np.testing.assert_allclose(frame.to_geodetic(frame.to_local(points)), points, atol=1e-7)


def test_invalid_atmosphere_and_track():
    with pytest.raises(ValueError):
        Atmosphere(100, (200, 0, 0))
    with pytest.raises(ValueError):
        Trajectory("bad", np.array([0.0, 0.0]), np.ones((2, 3)))


def test_acoustic_cpa_has_zero_frequency_shift_in_uniform_wind():
    t = np.arange(-200.0, 201.0, 2.0)
    track = Trajectory("wind", t, np.column_stack((80 * t, t * 0 + 1500, t * 0 + 1000)))
    atmosphere = Atmosphere(343.0, (100.0, 0.0, 0.0))
    heard, _, _ = closest_approaches(track, np.zeros(3), atmosphere)[0]
    frequency, _ = predict(track, np.zeros(3), np.array([heard]), 120.0, atmosphere)
    assert frequency[0] == pytest.approx(120.0, abs=1e-4)


def test_boundary_closest_approach_is_retained():
    t = np.arange(0.0, 101.0, 2.0)
    track = Trajectory("boundary", t, np.column_stack((80 * (t - 100), t * 0 + 1500, t * 0 + 1000)))
    hits = closest_approaches(track, np.zeros(3))
    assert len(hits) == 1
    assert hits[0][0] == pytest.approx(100 + np.hypot(1500, 1000) / 343, abs=0.001)


def test_interpolation_derivative_does_not_depend_on_other_side_of_gap():
    t = np.array([0.0, 2.0, 4.0, 100.0, 102.0, 104.0])
    positions = np.column_stack((np.array([0, 160, 320, 90000, 90160, 90320]), t * 0, t * 0 + 1000))
    track = Trajectory("gap", t, positions, max_gap=10)
    _, velocity = track.at(np.array([3.0, 4.0, 100.0, 101.0]))
    np.testing.assert_allclose(velocity[:, 0], 80, atol=1e-8)


def test_supersonic_motion_relative_to_medium_is_rejected():
    t = np.arange(0.0, 100.0, 2.0)
    track = Trajectory("too-fast", t, np.column_stack((t * 250, t * 0, t * 0 + 1000)))
    with pytest.raises(ValueError, match="subsonic"):
        predict(track, np.zeros(3), np.array([50.0]), 120.0, Atmosphere(343.0, (-100.0, 0.0, 0.0)))
