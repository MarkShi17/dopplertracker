import numpy as np
import pytest

from dopplertracker.geo import LocalFrame, ground_grid, validate_bbox
from dopplertracker.terrain import ConstantElevation, GridElevation, TerrainTiles, decode_terrarium


@pytest.mark.parametrize("llh", [[0, 0, 0], [45, 120, 100], [-80, -179.9, 10000]])
def test_local_frame_round_trips_wide_geographic_points(llh):
    frame = LocalFrame(10, -20)
    np.testing.assert_allclose(frame.to_geodetic(frame.to_local(llh)), llh, atol=2e-6)


@pytest.mark.parametrize("bbox", [(np.nan, 1, 0, 1), (0, np.inf, 0, 1), (0, 1, 170, -170), (-86, 0, 0, 1)])
def test_bbox_rejects_nonfinite_crossing_and_mercator_invalid_ranges(bbox):
    with pytest.raises(ValueError):
        validate_bbox(bbox)


@pytest.mark.parametrize("spacing", [np.nan, np.inf, 0, -10])
def test_ground_grid_rejects_nonfinite_or_nonpositive_spacing(spacing):
    with pytest.raises(ValueError):
        ground_grid(LocalFrame(0, 0), (0, 0.01, 0, 0.01), spacing, lambda lat, lon: np.zeros_like(lat))


def test_ground_grid_calls_elevation_vectorized_and_respects_limit():
    calls = []

    def elevation(lat, lon):
        calls.append((lat.shape, lon.shape))
        return 3 * lat - lon

    points = ground_grid(LocalFrame(0, 0), (0, 0.001, 0, 0.001), 100, elevation)
    assert calls == [(points.shape[:1], points.shape[:1])]
    assert points.shape[1] == 3
    with pytest.raises(ValueError, match="max_points"):
        ground_grid(LocalFrame(0, 0), (0, 1, 0, 1), 1, elevation, max_points=10)


@pytest.mark.parametrize("lat, lon", [(np.array([0, 1]), np.array([0, 1])), (0.5, 0.5)])
def test_grid_elevation_bilinear_interpolation_and_broadcasting(tmp_path, lat, lon):
    path = tmp_path / "dem.npz"
    np.savez(
        path,
        lat=np.array([0.0, 1.0]),
        lon=np.array([0.0, 1.0]),
        elevation_m=np.array([[0.0, 10.0], [20.0, 30.0]]),
    )
    result = GridElevation(path, geoid_offset=2)(lat, lon)
    expected = 2 + 20 * np.asarray(lat) + 10 * np.asarray(lon)
    np.testing.assert_allclose(result, expected)


def test_grid_elevation_rejects_out_of_bounds_coordinates(tmp_path):
    path = tmp_path / "dem.npz"
    np.savez(path, lat=[0.0, 1.0], lon=[0.0, 1.0], elevation_m=[[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        GridElevation(path)(2, 0.5)


@pytest.mark.parametrize("shape", [(), (3,), (2, 2)])
def test_constant_elevation_preserves_broadcast_shape(shape):
    lat = np.zeros(shape)
    result = ConstantElevation(100, 4)(lat, np.ones(shape))
    assert result.shape == shape
    np.testing.assert_allclose(result, 104)


class OfflineTiles(TerrainTiles):
    def __init__(self):
        self.zoom, self.geoid_offset, self.memory = 1, 1.5, {}

    def tile(self, x, y):
        # Each pixel's value uniquely exposes which adjacent tile was sampled.
        yy, xx = np.mgrid[:256, :256]
        return x * 100000 + y * 1000 + xx + 10 * yy


def test_terrarium_bilinear_sampling_crosses_tile_boundary_offline():
    tiles = OfflineTiles()
    # At zoom 1, lon 0 is the boundary: pixel coordinate 255.5, half from x=0 and x=1.
    got = tiles(0, 0)
    expected = (2805 + 102550 + 1255 + 101000) / 4 + 1.5
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("latitude", [-85, 85, 90])
def test_terrarium_rejects_web_mercator_edges(latitude):
    with pytest.raises(ValueError):
        OfflineTiles()(latitude, 0)


def test_terrarium_decode_is_linear_per_channel():
    a = np.array([[[1, 2, 3]]], dtype=np.uint8)
    b = np.array([[[5, 10, 20]]], dtype=np.uint8)
    np.testing.assert_allclose(
        decode_terrarium((a + b) // 2), (decode_terrarium(a) + decode_terrarium(b)) / 2, atol=0.51
    )
