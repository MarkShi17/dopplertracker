"""WGS84 ECEF and local ENU transforms; heights must be ellipsoidal meters."""

from dataclasses import dataclass

import numpy as np

A = 6378137.0
E2 = 6.6943799901413165e-3


def ecef(llh):
    llh = np.asarray(llh, dtype=float)
    lat, lon = np.deg2rad(llh[..., 0]), np.deg2rad(llh[..., 1])
    h = llh[..., 2]
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.stack(
        (
            (n + h) * np.cos(lat) * np.cos(lon),
            (n + h) * np.cos(lat) * np.sin(lon),
            (n * (1 - E2) + h) * np.sin(lat),
        ),
        axis=-1,
    )


@dataclass
class LocalFrame:
    lat: float
    lon: float

    def __post_init__(self):
        lat, lon = np.deg2rad([self.lat, self.lon])
        self.origin = ecef([self.lat, self.lon, 0])
        self.rotation = np.array(
            [
                [-np.sin(lon), np.cos(lon), 0],
                [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
            ]
        )

    def to_local(self, llh):
        return (ecef(llh) - self.origin) @ self.rotation.T

    def to_geodetic(self, xyz):
        p = np.asarray(xyz) @ self.rotation + self.origin
        x, y, z = np.moveaxis(p, -1, 0)
        rho = np.hypot(x, y)
        lat = np.arctan2(z, rho * (1 - E2))
        for _ in range(10):
            n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
            lat = np.arctan2(z + E2 * n * np.sin(lat), rho)
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        height = rho * np.cos(lat) + z * np.sin(lat) - n * (1 - E2 * np.sin(lat) ** 2)
        return np.stack((np.rad2deg(lat), np.rad2deg(np.arctan2(y, x)), height), axis=-1)


def validate_bbox(bbox):
    south, north, west, east = bbox
    if not (-85 < south < north < 85 and -180 <= west < east <= 180):
        raise ValueError("bbox must be SOUTH NORTH WEST EAST, without crossing the antimeridian")


def ground_grid(frame, bbox, spacing, elevation, max_points=2_000_000):
    validate_bbox(bbox)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    south, north, west, east = bbox
    lat = np.arange(south, north + 1e-10, spacing / 111320.0)
    lon = np.arange(west, east + 1e-10, spacing / (111320.0 * np.cos(np.deg2rad((south + north) / 2))))
    if len(lat) * len(lon) > max_points:
        raise ValueError("Grid exceeds max_points; split the region or use a coarser spacing")
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    h = elevation(la.ravel(), lo.ravel())
    return frame.to_local(np.column_stack((la.ravel(), lo.ravel(), h)))
