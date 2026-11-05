import gzip
import io
import json
import tarfile

import numpy as np
import pytest

from dopplertracker.data import extract_archive, parse_trace
from dopplertracker.geo import LocalFrame


def trace(rows):
    return {"icao": "edge", "timestamp": 10.0, "trace": rows}


def row(t, altitude=1200, flags=0, geometric=1250):
    return [t, 40.0, -73.0 + t / 10000, altitude, 0, 0, flags, 0, {}, None, geometric]


@pytest.mark.parametrize("altitude", [None, "1200", np.nan, np.inf, -np.inf])
def test_invalid_geometric_altitudes_break_segments_without_crashing(altitude):
    rows = [row(0), row(10, geometric=altitude), row(20), row(30), row(40)]
    tracks = parse_trace(trace(rows), LocalFrame(40, -73))
    assert len(tracks) == 1
    assert tracks[0].times[0] == 30


@pytest.mark.parametrize("flag", [1, 2, 8])
def test_documented_flags_have_safe_segment_semantics(flag):
    rows = [row(0), row(10, flags=flag), row(20), row(30), row(40)]
    tracks = parse_trace(trace(rows), LocalFrame(40, -73))
    if flag == 1:
        assert len(tracks) == 1 and tracks[0].times[0] == 30
    elif flag == 2:
        assert len(tracks) == 1 and tracks[0].times[0] == 20
    else:
        assert len(tracks) == 1


@pytest.mark.parametrize("flag", ["not-a-number", [], object()])
def test_malformed_flag_breaks_only_its_segment(flag):
    rows = [row(0), row(10, flags=flag), row(20), row(30), row(40)]
    tracks = parse_trace(trace(rows), LocalFrame(40, -73))
    assert len(tracks) == 1
    assert tracks[0].times[0] == 30


@pytest.mark.parametrize("offsets", [(0, 0), (20.5, -31.25)])
def test_baro_datum_offsets_are_meter_additive(offsets):
    rows = [row(t, altitude=1000, geometric=None) for t in [0, 10, 20]]
    frame = LocalFrame(40, -73)
    tracks = parse_trace(trace(rows), frame, allow_baro=True, baro_offset=offsets[0], geoid_offset=offsets[1])
    height = frame.to_geodetic(tracks[0].positions)[0, 2]
    assert height == pytest.approx(1000 * 0.3048 + sum(offsets), abs=1e-6)
    assert tracks[0].metadata["altitude_sources"] == ["corrected_baro_approximation"]


@pytest.mark.parametrize("maximum", [np.nan, np.inf, 0, -1])
def test_parse_trace_rejects_invalid_max_gap(maximum):
    with pytest.raises(ValueError):
        parse_trace(trace([row(t) for t in [0, 10, 20]]), LocalFrame(40, -73), max_gap=maximum)


def archive_with(*members):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w") as archive:
        for name, contents in members:
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    result.seek(0)
    return result


@pytest.mark.parametrize("raw", [b"not json", gzip.compress(b"not json"), b"[]"])
def test_archive_skips_malformed_trace_documents_with_warning(raw):
    stream = archive_with(("trace_full_bad.json", raw))
    with pytest.warns(UserWarning, match="malformed"):
        assert extract_archive(stream, LocalFrame(40, -73), (39, 41, -74, -72)) == []


def test_archive_ignores_non_trace_and_directory_members():
    good = json.dumps(trace([row(t) for t in [0, 10, 20]])).encode()
    stream = archive_with(("README", b"x"), ("folder/", b""), ("trace_full_good.json", good))
    assert len(extract_archive(stream, LocalFrame(40, -73), (39, 41, -74, -72))) == 1


@pytest.mark.parametrize("padding", [np.nan, np.inf, -0.1])
def test_archive_rejects_nonfinite_or_negative_padding(padding):
    with pytest.raises(ValueError):
        extract_archive(io.BytesIO(), LocalFrame(40, -73), (39, 41, -74, -72), padding_km=padding)
