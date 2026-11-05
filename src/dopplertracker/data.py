"""Read documented readsb traces; stream archives without extracting paths."""

import gzip
import io
import json
import tarfile
import warnings
from datetime import date
from urllib.request import Request, urlopen

import numpy as np

from .geo import LocalFrame, validate_bbox
from .physics import Trajectory


def open_url(url):
    return urlopen(Request(url, headers={"User-Agent": "dopplertracker/0.1"}), timeout=90)


def parse_trace(
    document, frame, jet_types=frozenset(), allow_baro=False, baro_offset=0.0, geoid_offset=0.0, max_gap=30.0
):
    """Prefer geometric feet. Optional corrected pressure altitude is marked approximate."""
    if not isinstance(document, dict):
        raise TypeError("trace document must be an object")
    if not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("max_gap must be finite and positive")
    if document.get("t") in jet_types:
        return []
    segments, current, sources = [], [], set()

    def finish():
        if len(current) >= 3:
            a = np.array(current)
            segments.append(
                Trajectory(
                    f"{document.get('icao', 'unknown')}:{a[0, 0]:.3f}",
                    a[:, 0],
                    frame.to_local(a[:, 1:]),
                    max_gap=max_gap,
                    metadata={
                        "type": document.get("t"),
                        "registration": document.get("r"),
                        "altitude_sources": sorted(sources),
                        "baro_offset_m": baro_offset,
                        "geoid_offset_m": geoid_offset,
                    },
                )
            )
        current.clear()
        sources.clear()

    for row in document.get("trace", []):
        if len(row) < 8:
            finish()
            continue
        raw_flags = row[6]
        if raw_flags is None:
            flags = 0
        elif (
            isinstance(raw_flags, (int, float, np.number))
            and np.isfinite(raw_flags)
            and float(raw_flags).is_integer()
        ):
            flags = int(raw_flags)
        else:
            finish()
            continue
        if row[3] == "ground" or flags & 1:
            finish()
            continue
        if flags & 2:
            finish()
        extra = row[8] if len(row) > 8 and isinstance(row[8], dict) else {}
        geometric = row[10] if len(row) > 10 else None
        if geometric is None:
            geometric = extra.get("alt_geom")
        if geometric is None and flags & 8:
            geometric = row[3]
        if isinstance(geometric, (int, float)):
            height, source = geometric * 0.3048, "geometric_ellipsoid"
        elif allow_baro and isinstance(row[3], (int, float)):
            height, source = row[3] * 0.3048 + baro_offset + geoid_offset, "corrected_baro_approximation"
        else:
            finish()
            continue
        values = [document["timestamp"] + row[0], row[1], row[2], height]
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in values):
            finish()
            continue
        if not (-90 <= values[1] <= 90 and -180 <= values[2] <= 180):
            finish()
            continue
        if current:
            dt = values[0] - current[-1][0]
            if dt <= 0:
                continue
            if dt > max_gap:
                finish()
            elif np.linalg.norm(frame.to_local(values[1:]) - frame.to_local(current[-1][1:])) / dt > 300:
                finish()
        current.append(values)
        sources.add(source)
    finish()
    return segments


def extract_archive(stream, frame, bbox, padding_km=15.0, progress=None, **options):
    validate_bbox(bbox)
    if not np.isfinite(padding_km) or padding_km < 0:
        raise ValueError("padding_km must be finite and nonnegative")
    if not np.isfinite(options.get("max_gap", 30.0)) or options.get("max_gap", 30.0) <= 0:
        raise ValueError("max_gap must be finite and positive")
    south, north, west, east = bbox
    dlat = padding_km / 111.0
    dlon = dlat / max(0.05, np.cos(np.deg2rad((south + north) / 2)))
    kept, seen, malformed = {}, 0, 0
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if not member.isfile() or "trace_full_" not in member.name:
                continue
            if member.size > 64 * 1024 * 1024:
                raise ValueError(f"trace member too large: {member.name}")
            seen += 1
            file = archive.extractfile(member)
            raw = file.read()
            try:
                if raw.startswith(b"\x1f\x8b"):
                    # Limit decompressed input too, not just compressed member size.
                    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                        raw = gz.read(64 * 1024 * 1024 + 1)
                    if len(raw) > 64 * 1024 * 1024:
                        raise ValueError("decompressed trace exceeds 64 MiB")
                document = json.loads(raw)
                for track in parse_trace(document, frame, **options):
                    llh = frame.to_geodetic(track.positions)
                    # Bounding-box overlap also retains sparse tracks crossing the region.
                    if (
                        llh[:, 0].max() >= south - dlat
                        and llh[:, 0].min() <= north + dlat
                        and llh[:, 1].max() >= west - dlon
                        and llh[:, 1].min() <= east + dlon
                    ):
                        kept[track.id] = track
            except (ValueError, KeyError, TypeError, EOFError, OSError):
                malformed += 1
            if progress and seen % 1000 == 0:
                progress(seen, len(kept))
    if malformed:
        warnings.warn(f"{malformed} malformed traces skipped out of {seen}", stacklevel=2)
    return list(kept.values())


class JoinedDownloads(io.RawIOBase):
    """Concatenate split .tar.aa/.ab assets in order with bounded memory."""

    def __init__(self, urls):
        super().__init__()
        self.urls = iter(urls)
        self.response = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.response is None:
                url = next(self.urls, None)
                if url is None:
                    return 0
                self.response = open_url(url)
            data = self.response.read(len(buffer))
            if data:
                buffer[: len(data)] = data
                return len(data)
            self.response.close()
            self.response = None

    def close(self):
        if self.response is not None:
            self.response.close()
        super().close()


def fetch_day(day, frame, bbox, **options):
    day = date.fromisoformat(day)
    url = f"https://raw.githubusercontent.com/adsblol/globe_history_{day.year}/main/PREFERRED_RELEASES.txt"
    with open_url(url) as response:
        lines = response.read().decode().splitlines()
    marker = f"/v{day:%Y.%m.%d}-"
    matches = [line for line in lines if marker in line]
    if not matches:
        raise ValueError(f"No preferred archive published for {day}; try extract-region with a local archive")
    urls = matches[0].split(",")
    if any(
        not u.startswith(f"https://github.com/adsblol/globe_history_{day.year}/releases/download/")
        for u in urls
    ):
        raise ValueError("unexpected archive URL in manifest")
    with io.BufferedReader(JoinedDownloads(urls)) as stream:
        return extract_archive(stream, frame, bbox, **options)


def save_tracks(path, tracks, frame):
    if not tracks:
        raise ValueError("no valid flight segments retained; check altitude policy, region and date")
    arrays = {
        "frame": np.array([frame.lat, frame.lon]),
        "metadata": np.array(json.dumps([{"id": t.id, "max_gap": t.max_gap, **t.metadata} for t in tracks])),
    }
    for i, track in enumerate(tracks):
        arrays[f"t{i}"], arrays[f"p{i}"] = track.times, track.positions
    np.savez_compressed(path, **arrays)


def load_tracks(path):
    with np.load(path, allow_pickle=False) as z:
        frame = LocalFrame(*z["frame"])
        metadata = json.loads(str(z["metadata"]))
        tracks = [
            Trajectory(item["id"], z[f"t{i}"], z[f"p{i}"], item.get("max_gap", 30.0), item)
            for i, item in enumerate(metadata)
        ]
    if not tracks:
        raise ValueError("trace file contains no flight segments")
    return tracks, frame
