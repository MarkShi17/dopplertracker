"""Re-extract the eagle cam region with reported ADS-B velocities into a new NPZ.

Reads the local raw archive sequentially with the frame and bbox of
data/eagle-cam/inputs.json (as used for traces.npz) and never touches the
existing traces.npz.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

from dopplertracker.data import extract_archive, save_tracks
from dopplertracker.geo import LocalFrame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/eagle-cam")
    parser.add_argument("--out", default="data/eagle-cam/traces-velocity.npz")
    args = parser.parse_args()
    directory, output = Path(args.data), Path(args.out)
    if output.exists():
        raise SystemExit(f"{output} exists; refusing to overwrite")
    inputs = json.loads((directory / "inputs.json").read_text())
    source = json.loads((directory / "traces-source.json").read_text())
    archive = directory / "archive" / f"{source['release']}.tar"
    if archive.stat().st_size != source["sizes"][0]:
        raise SystemExit(f"{archive} size differs from traces-source.json")
    frame = LocalFrame(*inputs["frame"])
    bbox = (*inputs["bbox"]["lat"], *inputs["bbox"]["lon"])
    hasher = hashlib.sha256()

    class Hashing(io.RawIOBase):
        """Read-through stream that hashes the archive while it is parsed (one pass)."""

        def __init__(self, raw):
            self.raw = raw

        def readable(self):
            return True

        def readinto(self, buffer):
            n = self.raw.readinto(buffer)
            if n:
                hasher.update(memoryview(buffer)[:n])
            return n

    with archive.open("rb") as raw, io.BufferedReader(Hashing(raw), 1 << 20) as stream:
        tracks = extract_archive(
            stream,
            frame,
            bbox,
            progress=lambda read, kept: print(f"{read} traces read; {kept} retained", flush=True),
        )
        while stream.read(1 << 20):  # hash any trailing padding too
            pass
    if hasher.hexdigest() != source["sha256"][0]:
        raise SystemExit("archive SHA-256 differs from traces-source.json")
    temporary = output.with_name(output.stem + ".partial.npz")
    save_tracks(temporary, tracks, frame)
    temporary.replace(output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "archive": archive.name,
                "sha256": hasher.hexdigest(),
                "frame": inputs["frame"],
                "bbox": inputs["bbox"],
                "segments": len(tracks),
                "with_velocity": sum(t.velocities is not None for t in tracks),
            },
            indent=2,
        )
    )
    print(f"wrote {output}: {len(tracks)} segments", flush=True)


if __name__ == "__main__":
    main()
