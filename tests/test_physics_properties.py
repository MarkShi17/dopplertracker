import numpy as np
import pytest

from dopplertracker.physics import Atmosphere, Trajectory, closest_approaches, predict, travel_time


@pytest.mark.parametrize("gap", [np.nan, np.inf, -np.inf])
def test_trajectory_rejects_nonfinite_gap(gap):
    with pytest.raises(ValueError):
        Trajectory("bad-gap", [0, 1], [[0, 0, 0], [1, 0, 0]], max_gap=gap)


def curved_track(time_offset=0.0, translation=(0, 0, 0)):
    t = np.linspace(-20, 30, 101) + time_offset
    u = t - time_offset
    p = np.column_stack((45 * u, 700 + 0.7 * u**2, 1200 + 20 * np.sin(u / 6)))
    return Trajectory("curve", t, p + np.asarray(translation))


@pytest.mark.parametrize("wind", [(0, 0, 0), (18, -7, 3)])
def test_curved_retarded_time_matches_independent_bracketed_solution(wind):
    from scipy.optimize import brentq

    track, observer = curved_track(), np.array([50.0, -100.0, 10.0])
    atmosphere = Atmosphere(343, wind)
    receive = np.linspace(-8, 34, 17)
    frequency, emission = predict(track, observer, receive, 105.0, atmosphere)
    for got_f, got_e, heard in zip(frequency, emission, receive):
        root = brentq(lambda e: e + travel_time(track.at(e)[0] - observer, atmosphere)[0] - heard, -20, 30)
        _, velocity = track.at(root)
        _, gradient = travel_time(track.at(root)[0] - observer, atmosphere)
        np.testing.assert_allclose(got_e, root, atol=2e-6)
        np.testing.assert_allclose(got_f, 105 / (1 + gradient @ velocity), rtol=2e-7)


def test_predict_is_translation_invariant():
    shift = np.array([2e5, -4e5, 7e4])
    receive = np.linspace(-5, 25, 31)
    base = predict(curved_track(), [20, 30, -40], receive, 100)
    moved = predict(curved_track(translation=shift), shift + [20, 30, -40], receive, 100)
    np.testing.assert_allclose(moved[0], base[0])
    np.testing.assert_allclose(moved[1], base[1])


def test_predict_is_rotation_invariant():
    angle = 0.73
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    base = curved_track()
    rotated = Trajectory("rotated", base.times, base.positions @ rotation.T)
    receive = np.linspace(-5, 25, 31)
    np.testing.assert_allclose(
        predict(rotated, np.array([20, 30, -40]) @ rotation.T, receive, 100)[0],
        predict(base, [20, 30, -40], receive, 100)[0],
    )


def test_time_origin_only_shifts_emission_clock():
    receive = np.linspace(-5, 25, 31)
    base = predict(curved_track(), [0, 0, 0], receive, 98, drift=0.02, reference=0)
    shifted = predict(curved_track(10000), [0, 0, 0], receive + 10000, 98, drift=0.02, reference=10000)
    np.testing.assert_allclose(shifted[0], base[0])
    np.testing.assert_allclose(shifted[1] - 10000, base[1])


@pytest.mark.parametrize("point", [[1000, 0, 0], [0, -500, 800], [-800, 500, 40]])
def test_wind_travel_time_satisfies_moving_medium_ray_equation(point):
    atmosphere = Atmosphere(340, (20, -10, 5))
    tau, _ = travel_time(point, atmosphere)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(point) + tau * np.asarray(atmosphere.wind)), atmosphere.c * tau
    )


@pytest.mark.parametrize("query", [0.0, np.array(0.0), np.array([0.0, 0.5])])
def test_trajectory_scalar_and_vector_queries_have_position_axis(query):
    track = Trajectory("line", [0, 1, 2], [[0, 0, 1], [1, 0, 1], [2, 0, 1]])
    pos, vel = track.at(query)
    assert pos.shape == np.asarray(query).shape + (3,)
    assert vel.shape == pos.shape


@pytest.mark.parametrize(
    "times, positions",
    [([0, 1], [[0, 0], [1, 1]]), ([0, 1], [[0, 0, np.nan], [1, 1, 1]]), ([[0], [1]], [[0, 0, 0], [1, 1, 1]])],
)
def test_trajectory_rejects_adversarial_position_and_time_shapes(times, positions):
    with pytest.raises(ValueError):
        Trajectory("bad", times, positions)


@pytest.mark.parametrize("wind", [(343, 0, 0), (0, 0), (np.nan, 0, 0)])
def test_atmosphere_rejects_invalid_wind(wind):
    with pytest.raises(ValueError):
        Atmosphere(343, wind)


def test_gap_is_never_interpolated_and_valid_marks_only_measured_spans():
    track = Trajectory("gap", [0, 1, 10, 11], [[0, 0, 1], [1, 0, 1], [10, 0, 1], [11, 0, 1]], max_gap=2)
    np.testing.assert_array_equal(
        track.valid(np.array([-1, 0, 0.5, 1, 5, 10, 11, 12])),
        [False, True, True, True, False, True, True, False],
    )
    position, velocity = track.at(np.array([5.0]))
    assert np.isnan(position).all() and np.isnan(velocity).all()


def test_closest_approach_of_stationary_track_is_finite_endpoint():
    track = Trajectory("still", [0, 10, 20], [[3, 4, 0]] * 3)
    heard, distance, speed = closest_approaches(track, [0, 0, 0])[0]
    assert heard == pytest.approx(5 / 343)
    assert distance == pytest.approx(5)
    assert speed == pytest.approx(0)
