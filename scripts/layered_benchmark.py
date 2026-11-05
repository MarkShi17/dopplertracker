"""Extended synthetic matrix: spline-linear under layered (stratified) atmospheres.

Development seeds only by default.  Not part of the default stress benchmark.
"""

import argparse
import json

from dopplertracker.stress_benchmark import LAYERED_METHODS, run_layered_benchmark
from dopplertracker.stress_scenes import EXTENDED_SCENARIOS, LAYERED_SCENARIOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/layered-benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--scenarios", nargs="+", choices=EXTENDED_SCENARIOS, default=list(LAYERED_SCENARIOS))
    parser.add_argument("--methods", nargs="+", choices=list(LAYERED_METHODS), default=list(LAYERED_METHODS))
    parser.add_argument("--split", choices=["development", "held-out"], default="development")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    report = run_layered_benchmark(
        args.out,
        args.seeds,
        args.scenarios,
        args.methods,
        args.split,
        args.workers,
        progress=lambda done, total, scenario, seed: print(f"{done}/{total} {scenario} {seed}", flush=True),
    )
    for scenario, summary in report["by_scenario"].items():
        for method, row in summary.items():
            print(
                f"{scenario:22s} {method:36s} median {row['median_error_m']:.1f} m  "
                f"p90 {row['p90_error_m']:.1f}  worst {row['worst_error_m']:.1f}  "
                f"accepted {1 - row['rejected_signal_fraction']:.0%}  exceptions {row['exceptions']}"
            )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
