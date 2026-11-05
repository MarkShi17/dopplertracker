"""Hum lines in demodulation: masking frames versus removing the lines (synthetic, dev seeds).

Runs ``audio_benchmark.run_measurement_benchmark`` with the ``HUM_METHODS``
(demodulation, two re-passes, SNR noise; hum masked, subtracted or notched) on
the ``hum_crossing`` scene, where stationary lines cross every tone near closest
approach, and on ``fast_chirp`` (a 120 Hz line crosses the first pass).
Development seeds 0-9 only.  Frequency-track error uses the generator's
received frequency and never enters fitting.
"""

import argparse

import numpy as np

from dopplertracker.audio_benchmark import HUM_METHODS, run_measurement_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--scenarios", nargs="+", default=["hum_crossing", "fast_chirp"])
    parser.add_argument("--out", default="outputs/hum-benchmark")
    args = parser.parse_args()
    if any(not 0 <= s <= 9 for s in args.seeds):
        raise SystemExit("development seeds are 0-9")
    report = run_measurement_benchmark(
        args.seeds, tuple(args.scenarios), tuple(HUM_METHODS), args.out, progress=None
    )
    print(
        f"{'scene/method':32s} {'fail':>4s} {'track rms':>9s} {'cpa rms':>8s} {'median|e|':>9s} "
        f"{'loc m':>6s} {'valid':>7s} {'cpa valid':>12s}"
    )
    for scenario in args.scenarios:
        for method in HUM_METHODS:
            rows = [r for r in report["trials"] if r["scenario"] == scenario and r["method"] == method]
            done = [r for r in rows if r["error"] is None]

            def median(key, done=done):
                values = [r[key] for r in done if r.get(key) is not None]
                return float(np.median(values)) if values else float("nan")

            valid = sum(sum(r["valid_frames"]) for r in done)
            near = sum(sum(r["cpa_valid_frames"]) for r in done)
            frames = sum(sum(r["cpa_frames"]) for r in done)
            print(
                f"{scenario + '/' + method:32s} {len(rows) - len(done):4d} {median('track_rms_hz'):9.4f} "
                f"{median('track_cpa_rms_hz'):8.4f} {median('track_median_abs_hz'):9.4f} "
                f"{median('location_error_m'):6.2f} {valid:7d} {f'{near}/{frames}':>12s}"
            )


if __name__ == "__main__":
    main()
