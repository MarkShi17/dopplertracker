import hashlib
import io

import pytest

from dopplertracker.download import DownloadProtocolError, download_file


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None, timeout_after=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}
        self.timeout_after = timeout_after

    def read(self, size=-1):
        if self.timeout_after is not None and self.tell() >= self.timeout_after:
            raise TimeoutError("connection stalled")
        if self.timeout_after is not None:
            size = min(size, self.timeout_after - self.tell())
        return super().read(size)


def test_timeout_resumes_without_losing_or_duplicating_bytes(tmp_path, monkeypatch):
    data = b"abcdefgh"
    requests = []

    def open_response(request, **kwargs):
        requests.append(request)
        if len(requests) == 1:
            return Response(data, headers={"Content-Length": "8"}, timeout_after=3)
        return Response(data[3:], 206, {"Content-Range": "bytes 3-7/8", "Content-Length": "5"})

    monkeypatch.setattr("dopplertracker.download.urlopen", open_response)
    path = tmp_path / "archive.aa"
    download_file(
        "https://example.com/archive", path, 8, sha256=hashlib.sha256(data).hexdigest(), retry_delay=0
    )
    assert path.read_bytes() == data
    assert requests[1].get_header("Range") == "bytes=3-"
    assert not path.with_name(path.name + ".part").exists()


def test_separate_invocation_resumes_persisted_partial_file(tmp_path, monkeypatch):
    path = tmp_path / "archive.aa"
    monkeypatch.setattr(
        "dopplertracker.download.urlopen", lambda *a, **k: Response(b"abcdefgh", timeout_after=3)
    )
    with pytest.raises(TimeoutError):
        download_file("https://example.com/archive", path, 8, max_attempts=1, retry_delay=0)
    assert path.with_name(path.name + ".part").read_bytes() == b"abc"
    assert not path.exists()
    monkeypatch.setattr(
        "dopplertracker.download.urlopen",
        lambda *a, **k: Response(b"defgh", 206, {"Content-Range": "bytes 3-7/8"}),
    )
    download_file("https://example.com/archive", path, 8)
    assert path.read_bytes() == b"abcdefgh"


@pytest.mark.parametrize(
    "status,headers",
    [(200, {}), (206, {"Content-Range": "bytes 0-4/8"}), (206, {"Content-Range": "bytes 3-7/9"})],
)
def test_bad_resume_response_never_corrupts_saved_bytes(tmp_path, monkeypatch, status, headers):
    path = tmp_path / "archive.aa"
    monkeypatch.setattr(
        "dopplertracker.download.urlopen", lambda *a, **k: Response(b"abcdefgh", timeout_after=3)
    )
    with pytest.raises(TimeoutError):
        download_file("https://example.com/archive", path, 8, max_attempts=1)
    monkeypatch.setattr(
        "dopplertracker.download.urlopen", lambda *a, **k: Response(b"XXXXX", status, headers)
    )
    with pytest.raises(DownloadProtocolError):
        download_file("https://example.com/archive", path, 8, retry_delay=0)
    assert path.with_name(path.name + ".part").read_bytes() == b"abc"
    assert not path.exists()


def test_short_response_retries_remaining_bytes(tmp_path, monkeypatch):
    responses = iter([Response(b"abc"), Response(b"defgh", 206, {"Content-Range": "bytes 3-7/8"})])
    monkeypatch.setattr("dopplertracker.download.urlopen", lambda *a, **k: next(responses))
    path = tmp_path / "archive"
    download_file("https://example.com/archive", path, 8, retry_delay=0)
    assert path.read_bytes() == b"abcdefgh"


def test_complete_verified_download_is_reused_without_network(tmp_path, monkeypatch):
    path = tmp_path / "archive"
    monkeypatch.setattr("dopplertracker.download.urlopen", lambda *a, **k: Response(b"abc"))
    digest = hashlib.sha256(b"abc").hexdigest()
    download_file("https://example.com/archive", path, 3, sha256=digest)

    def forbidden(*args, **kwargs):
        raise AssertionError("complete download must not be fetched again")

    monkeypatch.setattr("dopplertracker.download.urlopen", forbidden)
    download_file("https://example.com/archive", path, 3, sha256=digest)
    assert path.read_bytes() == b"abc"


def test_checksum_failure_is_not_published_as_complete(tmp_path, monkeypatch):
    path = tmp_path / "archive"
    monkeypatch.setattr("dopplertracker.download.urlopen", lambda *a, **k: Response(b"bad"))
    with pytest.raises(DownloadProtocolError, match="checksum"):
        download_file("https://example.com/archive", path, 3, sha256=hashlib.sha256(b"abc").hexdigest())
    assert not path.exists()


def test_split_archive_can_cross_a_tar_header_boundary(tmp_path):
    import tarfile

    from dopplertracker.download import JoinedFiles

    archive = io.BytesIO()
    payload = b"actual archive member"
    with tarfile.open(fileobj=archive, mode="w") as tar:
        member = tarfile.TarInfo("trace.json")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(archive.getvalue()[:101])
    second.write_bytes(archive.getvalue()[101:])
    with io.BufferedReader(JoinedFiles([first, second])) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            member = tar.next()
            assert tar.extractfile(member).read() == payload
