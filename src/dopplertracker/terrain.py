"""Explicit constant elevation, local DEM grids, or cached public Terrarium tiles."""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from .data import open_url


def decode_terrarium(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


class ConstantElevation:
    def __init__(self, meters, geoid_offset=0.0):
        if not np.isfinite([meters, geoid_offset]).all():
            raise ValueError("elevation and geoid offset must be finite")
        self.height = meters + geoid_offset

    def __call__(self, lat, lon):
        return np.full(np.broadcast_arrays(lat, lon)[0].shape, self.height)


class GridElevation:
    """NPZ fields lat (ascending), lon (ascending), elevation_m (MSL)."""

    def __init__(self, path, geoid_offset=0.0):
        with np.load(path, allow_pickle=False) as z:
            self.interpolator = RegularGridInterpolator(
                (z["lat"], z["lon"]), z["elevation_m"] + geoid_offset, bounds_error=True
            )

    def __call__(self, lat, lon):
        lat, lon = np.broadcast_arrays(lat, lon)
        return self.interpolator(np.stack((lat, lon), axis=-1))


class TerrainTiles:
    def __init__(self, cache=".cache/terrain", zoom=12, geoid_offset=0.0):
        self.cache, self.zoom, self.geoid_offset = Path(cache), zoom, geoid_offset
        self.cache.mkdir(parents=True, exist_ok=True)
        self.memory = {}

    def tile(self, x, y):
        key = (int(x), int(y))
        if key not in self.memory:
            path = self.cache / f"{self.zoom}-{key[0]}-{key[1]}.png"
            if not path.exists():
                url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{self.zoom}/{key[0]}/{key[1]}.png"
                with open_url(url) as response:
                    raw = response.read(4 * 1024 * 1024)
                with Image.open(io.BytesIO(raw)) as picture:
                    picture.verify()
                with tempfile.NamedTemporaryFile(dir=self.cache, delete=False) as temp:
                    temp.write(raw)
                    temp_path = temp.name
                os.replace(temp_path, path)
            with Image.open(path) as picture:
                self.memory[key] = decode_terrarium(picture.convert("RGB"))
        return self.memory[key]

    def __call__(self, lat, lon):
        lat, lon = np.broadcast_arrays(lat, lon)
        if np.any(np.abs(lat) >= 85):
            raise ValueError("terrain tiles require latitude between -85 and 85")
        n = 2**self.zoom
        # Pixel centers; bilinear sampling crosses tile boundaries correctly.
        px = (lon.ravel() + 180) / 360 * n * 256 - 0.5
        py = (1 - np.arcsinh(np.tan(np.deg2rad(lat.ravel()))) / np.pi) / 2 * n * 256 - 0.5
        ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
        dx, dy = px - ix, py - iy
        result = np.zeros(len(px))
        for ox, oy, weight in (
            (0, 0, (1 - dx) * (1 - dy)),
            (1, 0, dx * (1 - dy)),
            (0, 1, (1 - dx) * dy),
            (1, 1, dx * dy),
        ):
            gx, gy = (ix + ox) % (n * 256), np.clip(iy + oy, 0, n * 256 - 1)
            tx, ty = gx // 256, gy // 256
            for x, y in np.unique(np.column_stack((tx, ty)), axis=0):
                mask = (tx == x) & (ty == y)
                result[mask] += weight[mask] * self.tile(x, y)[gy[mask] % 256, gx[mask] % 256]
        return result.reshape(lat.shape) + self.geoid_offset
