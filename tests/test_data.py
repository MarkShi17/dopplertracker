import gzip
import io
import json
import tarfile

import numpy as np

from dopplertracker.data import extract_archive, load_tracks, parse_trace, save_tracks
from dopplertracker.geo import LocalFrame
from dopplertracker.terrain import decode_terrarium


def trace_fixture():
    return {
        "icao": "abc123",
        "t": "PC12",
        "timestamp": 1000.0,
        "trace": [
            [0, 40.0, -73.0, 3000, 100, 90, 0, 0, None, "adsb_icao", 3200],
            [10, 40.0, -72.99, 3000, 100, 90, 0, 0, None, "adsb_icao", 3200],
            [20, 40.0, -72.98, 3000, 100, 90, 0, 0, None, "adsb_icao", 3200],
        ],
    }


def test_geometric_altitude_and_time_units():
    frame = LocalFrame(40, -73)
    tracks = parse_trace(trace_fixture(), frame)
    assert len(tracks) == 1
    np.testing.assert_allclose(tracks[0].times, [1000, 1010, 1020])
    np.testing.assert_allclose(frame.to_geodetic(tracks[0].positions)[:, 2], 3200 * 0.3048, atol=1e-6)


def test_pressure_altitude_requires_explicit_opt_in():
    trace = trace_fixture()
    for row in trace["trace"]:
        row[10] = None
    assert parse_trace(trace, LocalFrame(40, -73)) == []
    assert len(parse_trace(trace, LocalFrame(40, -73), allow_baro=True, baro_offset=40)) == 1


def test_ground_and_large_gap_split_tracks():
    trace = trace_fixture()
    trace["trace"].append([30, 40, -72.97, "ground", 0, 0, 0, 0, None, None, 10])
    trace["trace"].extend([list(row) for row in trace["trace"][:3]])
    for row in trace["trace"][-3:]:
        row[0] += 200
    assert len(parse_trace(trace, LocalFrame(40, -73))) == 2


def test_archive_stream_and_npz_round_trip(tmp_path):
    archive = io.BytesIO()
    raw = gzip.compress(json.dumps(trace_fixture()).encode())
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo("traces/23/trace_full_abc123.json")
        member.size = len(raw)
        tar.addfile(member, io.BytesIO(raw))
    archive.seek(0)
    frame = LocalFrame(40, -73)
    tracks = extract_archive(archive, frame, (39, 41, -74, -72))
    assert len(tracks) == 1
    path = tmp_path / "traces.npz"
    save_tracks(path, tracks, frame)
    recovered, loaded_frame = load_tracks(path)
    assert recovered[0].id == tracks[0].id
    np.testing.assert_array_equal(recovered[0].positions, tracks[0].positions)
    assert loaded_frame.lat == 40


def test_terrarium_decodes_fractional_and_negative_elevation():
    rgb = np.array([[[128, 10, 128], [127, 255, 0]]], dtype=np.uint8)
    np.testing.assert_allclose(decode_terrarium(rgb), [[10.5, -1.0]])


def test_known_jets_removed_unknown_types_kept():
    frame = LocalFrame(40, -73)
    trace = trace_fixture()
    assert len(parse_trace(trace, frame, jet_types={"A320"})) == 1
    assert parse_trace(trace, frame, jet_types={"PC12"}) == []
