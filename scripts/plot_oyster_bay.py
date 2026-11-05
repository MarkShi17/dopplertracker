"""Plot the real audio and frozen comparison results (requires matplotlib)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dopplertracker.advanced import predict_result
from dopplertracker.data import load_tracks
from dopplertracker.detection import load_events
from dopplertracker.signal import Spectrogram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default="outputs/oyster-bay/comparison-irregular-cadence")
    parser.add_argument("--out", default="docs/results/oyster-bay/real-audio-comparison.png")
    args = parser.parse_args()
    directory = Path(args.comparison)
    report = json.loads((directory / "results.json").read_text())
    tracks, frame = load_tracks(directory / "matched-tracks.npz")
    spec = Spectrogram.load("data/oyster-bay/spectrogram.npz").whiten()
    events = load_events(["data/oyster-bay/events.json"])
    rows = {
        row["method"]: row
        for row in report["trials"]
        if row["height_m"] == -30.0 and row["sound_speed_m_s"] == 343.0
    }
    colors = {"brightness": "#ff9f43", "spline-linear": "#00d1c1", "kalman-linear": "#c683ff"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for index, event in enumerate(events):
        ax = axes[index]
        selected = (spec.times >= event.window[0]) & (spec.times <= event.window[1])
        band = (spec.frequencies >= 80) & (spec.frequencies <= 190)
        times = spec.times[selected]
        ax.pcolormesh(
            times,
            spec.frequencies[band],
            spec.power[np.ix_(band, selected)],
            shading="auto",
            cmap="magma",
            vmin=0,
            vmax=5,
            rasterized=True,
        )
        for method in ("brightness", "spline-linear"):
            hz = predict_result(tracks[index], event, times, rows[method]["result"], index)
            ax.plot(times, hz, color=colors[method], lw=1.1, label=method)
        ax.set(title=f"Actual recording: flyover {index + 1}", xlabel="Recording time (seconds)", ylabel="Hz")
        ax.legend(fontsize=8)
    ax = axes[2]
    truth = frame.to_local([*report["evaluation_truth"], 0.0])
    ax.scatter(0, 0, marker="*", color="black", s=140, label="Known camera (evaluation only)")
    for method, color in colors.items():
        row = rows[method]
        offset = np.asarray(row["result"]["position"]) - truth
        ax.scatter(
            *offset[:2], color=color, edgecolors="black", s=70, label=f"{method}: {row['error_m']:.1f} m"
        )
    published = json.loads(Path("data/oyster-bay/published/verified_published_baseline.json").read_text())
    best = published["oyster_bay_refined_50m.csv"]["highest_scoring_row"]
    offset = frame.to_local([float(best["lat"]), float(best["lon"]), 0.0]) - truth
    ax.scatter(*offset[:2], color="gray", marker="x", s=70, label="Published result: 43.3 m")
    ax.set(
        title="Same known flight pair; local refinement",
        xlabel="East of camera (m)",
        ylabel="North of camera (m)",
        xlim=(-150, 150),
        ylim=(-150, 150),
        aspect="equal",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Oyster Bay • 343 m/s sound speed • −30 m ellipsoid height • unmeasured atmosphere/height")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
