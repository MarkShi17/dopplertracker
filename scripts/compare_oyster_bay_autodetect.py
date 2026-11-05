"""Real-audio comparison with automatically detected events in place of the published ones.

1. The calibrated detector runs on the whole whitened hour with no hints and the
   default settings; every detection and refined candidate is saved.
2. For each published event the detection with the highest z whose closest
   approach lies within --match-seconds is chosen.  The published events serve
   only to say which automatic detection stands for which flyover; no published
   parameter enters detection or the automatic events.
3. scripts/compare_oyster_bay.py then runs unchanged on a mirror data directory
   whose events.json holds the automatic events (all other files are symlinks),
   so the protocol, flight identities and coarse start are identical.  As
   there, truth is read only after every fit has finished.
"""

import argparse
import json
import runpy
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dopplertracker.detection import FlyoverOptions, detect_flyovers, load_events, save_events
from dopplertracker.signal import Spectrogram

MIRRORED = ("spectrogram.npz", "traces.npz", "published")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="data/oyster-bay")
    parser.add_argument("--out", default="outputs/oyster-bay/autodetect")
    parser.add_argument("--match-seconds", type=float, default=20.0)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--heights", nargs="+", default=["-30.0", "0.0"])
    parser.add_argument("--sound-speeds", nargs="+", default=["343.0", "340.0"])
    parser.add_argument("--detect-only", action="store_true", help="Stop after detection and event selection")
    args = parser.parse_args()
    directory, output = Path(args.data).resolve(), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    spec = Spectrogram.load(directory / "spectrogram.npz").whiten()
    options = FlyoverOptions()
    report = {}
    started = time.perf_counter()
    detections = detect_flyovers(spec, options, top=args.top, report=report)
    runtime = time.perf_counter() - started
    save_events(detections, output / "detections.json")
    print(f"{len(detections)} detections in {runtime:.1f} s (z >= {options.min_z:g})", flush=True)
    for rank, event in enumerate(detections, 1):
        print(
            f"  #{rank:2d} CPA {event.cpa:7.1f}s f0 {event.f0:6.2f}Hz speed {event.speed:5.1f}m/s "
            f"slant {event.distance:6.0f}m z {event.z:5.1f}",
            flush=True,
        )

    published = load_events([str(directory / "events.json")])
    chosen, matches = [], []
    for index, reference in enumerate(published, 1):
        nearby = [
            (rank, event)
            for rank, event in enumerate(detections, 1)
            if abs(event.cpa - reference.cpa) <= args.match_seconds
        ]
        if not nearby:
            raise SystemExit(
                f"No automatic detection within {args.match_seconds} s of published event {index}"
            )
        rank, event = max(nearby, key=lambda item: item[1].z)
        chosen.append(event)
        matches.append(
            {
                "published_cpa": reference.cpa,
                "published_f0": reference.f0,
                "rank": rank,
                "event": asdict(event),
                "cpa_minus_published_s": event.cpa - reference.cpa,
            }
        )
        print(f"Published event {index} at {reference.cpa:.1f}s -> detection #{rank}", flush=True)
    (output / "detection.json").write_text(
        json.dumps(
            {
                "validation": "Real recording, whole hour, no hints; published events used only for matching.",
                "options": asdict(options),
                "runtime_s": runtime,
                "matches": matches,
                "detections": [asdict(e) for e in detections],
                "candidates": report["candidates"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    if args.detect_only:
        return

    mirror = output / "data"
    mirror.mkdir(exist_ok=True)
    for name in MIRRORED:
        link = mirror / name
        if not link.exists():
            link.symlink_to(directory / name)
    save_events(chosen, mirror / "events.json")
    script = Path(__file__).resolve().parent / "compare_oyster_bay.py"
    sys.argv = [
        str(script),
        "--data",
        str(mirror),
        "--out",
        str(output / "comparison"),
        "--heights",
        *args.heights,
        "--sound-speeds",
        *args.sound_speeds,
    ]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
