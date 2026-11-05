"""Persistent, byte-validated downloads for large public archive assets."""

import hashlib
import io
import json
import os
import re
import time
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class DownloadProtocolError(ValueError):
    """An unsafe resume response or corrupted asset; never append its bytes."""


def _verify(path, size, sha256):
    if path.stat().st_size != size:
        raise DownloadProtocolError(f"Unexpected file size: {path}")
    if sha256:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            raise DownloadProtocolError(f"Archive checksum mismatch: {path}")


def download_file(
    url, path, expected_size, *, sha256=None, max_attempts=10, timeout=45, retry_delay=2, progress=None
):
    """Resume a .part file across retries and process restarts; publish atomically.

    A complete asset must match its published size and optional SHA256. Resumed
    requests must return HTTP 206 with the exact requested Content-Range. A server
    ignoring Range is rejected rather than corrupting the file by appending it.
    Partial files remain after errors or interruption. Only one process may write
    a given path at a time.
    """
    if expected_size < 1 or max_attempts < 1 or timeout <= 0 or retry_delay < 0:
        raise ValueError("size, attempts and timeout must be positive; retry delay nonnegative")
    if sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("sha256 must be 64 lowercase hex characters")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    metadata = path.with_name(path.name + ".download.json")
    identity = {"url": url, "size": expected_size, "sha256": sha256}
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise DownloadProtocolError(f"Download identity differs from saved checkpoint: {path}")
    if path.exists():
        _verify(path, expected_size, sha256)
        return path
    if partial.exists() and not metadata.exists():
        raise DownloadProtocolError(f"Partial file has no source metadata: {partial}")
    metadata.write_text(json.dumps(identity, indent=2))
    last_progress = 0.0
    for attempt in range(max_attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise DownloadProtocolError(f"Partial file exceeds expected size: {partial}")
        if offset == expected_size:
            break
        headers = {"User-Agent": "dopplertracker/0.1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                status = response.status
                if offset or status == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                    )
                    if status != 206 or not match:
                        raise DownloadProtocolError("Server did not honor the resume range")
                    start, end, total = map(int, match.groups())
                    if start != offset or total != expected_size or not start <= end < total:
                        raise DownloadProtocolError("Server returned an inconsistent Content-Range")
                    response_end = end + 1
                elif status == 200:
                    response_end = expected_size
                else:
                    raise DownloadProtocolError(f"Unexpected HTTP status: {status}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != response_end - offset:
                    raise DownloadProtocolError("Content-Length differs from expected byte range")
                with partial.open("ab") as output:
                    try:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            if offset + len(block) > response_end:
                                raise DownloadProtocolError("Response exceeds advertised byte range")
                            output.write(block)
                            offset += len(block)
                            now = time.monotonic()
                            if progress and (now - last_progress >= 5 or offset == expected_size):
                                progress(offset, expected_size)
                                last_progress = now
                    finally:
                        output.flush()
                        os.fsync(output.fileno())
                if offset == expected_size:
                    break
                raise EOFError(f"Incomplete archive: {offset}/{expected_size} bytes retained")
        except (OSError, HTTPException, EOFError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt + 1 == max_attempts:
                raise
            if progress:
                progress(partial.stat().st_size if partial.exists() else 0, expected_size)
            time.sleep(min(5, retry_delay * (attempt + 1)))
    _verify(partial, expected_size, sha256)
    partial.replace(path)
    return path


class JoinedFiles(io.RawIOBase):
    """Read split TAR files as one stream, without writing a second 4 GB copy."""

    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = Path(path).open("rb")
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()
