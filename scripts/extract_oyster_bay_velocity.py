"""Re-extract the Oyster Bay region with reported ADS-B velocities into a new NPZ.

Reads the local split archive sequentially (same frame and bbox as
scripts/fetch_video_example.sh) and never touches the existing traces.npz.
"""

import argparse
import io
import json
from pathlib import Path

from dopplertracker.data import extract_archive, save_tracks
from dopplertracker.download import JoinedFiles
from dopplertracker.geo import LocalFrame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/oyster-bay/archive")
    parser.add_argument("--out", default="data/oyster-bay/traces-velocity.npz")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"{output} exists; refusing to overwrite")
    parts = sorted(Path(args.archive).glob("*.tar.a?"))
    if [p.suffix for p in parts] != [".aa", ".ab"]:
        raise SystemExit(f"expected .tar.aa and .tar.ab parts, found {parts}")
    frame = LocalFrame(40.85, -73.5)
    with io.BufferedReader(JoinedFiles(parts)) as stream:
        tracks = extract_archive(
            stream,
            frame,
            (40.4, 41.3, -74.4, -72.6),
            progress=lambda read, kept: print(f"{read} traces read; {kept} retained", flush=True),
        )
    temporary = output.with_name(output.stem + ".partial.npz")
    save_tracks(temporary, tracks, frame)
    temporary.replace(output)
    output.with_suffix(".json").write_text(
        json.dumps({"archive_parts": [p.name for p in parts], "segments": len(tracks)}, indent=2)
    )
    print(f"wrote {output}: {len(tracks)} segments", flush=True)


if __name__ == "__main__":
    main()
