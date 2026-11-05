# Dopplertracker

**Find where a recording was made from the sound of aircraft flying past.**

A propeller plane's pitch drops as it passes a microphone. The exact shape of that
drop depends on where the microphone is relative to the flight path. Dopplertracker
measures those curves in the audio, compares them with public ADS-B flight tracks,
and solves for the microphone's location.

Based on the method shown in [colsto's YouTube video](https://www.youtube.com/watch?v=UytYrhYLOKI).

![Fitted Doppler curves on a real recording, and the estimated locations](docs/results/oyster-bay/real-audio-comparison.png)

*Two real flyovers from a one-hour recording in Oyster Bay, New York. Left and
middle: fitted Doppler curves over the spectrogram. Right: estimated locations
relative to the true camera position.*

## Results

**Real recordings**

| Recording | What happened |
|---|---|
| Oyster Bay, NY (one hour; used for development) | From raw audio with no hints: three flyovers detected, a 100 × 150 km region searched in 5–20 minutes, the correct three aircraft ranked first, 16–22 m from the true spot after refinement. |
| Same recording, ten model variants | Error from 11 m to 220 m, median about 70 m. The tool's own 90% error region is 100–270 m. |
| SW Florida eagle cam (unseen until the final test) | 771–1,107 m off with every method. Both aircraft pass nearly overhead, and the tool's evidence check rejected all of its own fits. |

For comparison, the original [dopplertrack](https://github.com/mjshiggins/dopplertrack)
publishes 43 m for Oyster Bay and, for the eagle cam, 1.7 km with two flyovers and
592 m with five. Its code was not rerun here.

**Synthetic tests on seeds never used in development**

| Metric | Previous method | New option |
|---|---:|---:|
| Flyovers found automatically | 66% | 100% |
| Detections that are real | 50% | 100% |
| Median error, scenes with realistic flight tracks | 19.3 m | 6.3 m |
| Median error, scenes with a layered atmosphere | 16.2 m | 6.7 m |
| Estimates within 100 m, original 20 scenarios | 95% | 96% |
| Regional search, full region | about 32 h | 5–20 min |

How to read these numbers:

- **Real accuracy is tens of metres to about a kilometre, depending on geometry.**
  Aircraft passing to the side at different bearings locate the microphone well;
  aircraft passing overhead do not.
- **Any single error figure is one draw.** On Oyster Bay the answer moves by more than
  100 m between reasonable model choices that fit the audio equally well.
- **The accuracy options are unproven on real audio.** Reported aircraft velocity and
  a measured atmosphere help clearly on synthetic data. On two real recordings neither
  showed a reliable gain.
- **Synthetic scores are optimistic.** The scenes have simpler physics than real air
  and real flight data.

Details: [what was tried and what each change did](docs/campaign-results.md) ·
[real-recording study](docs/oyster-bay-results.md) ·
[synthetic study](docs/accuracy-expansion.md)

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). `ffmpeg` is only needed
for real audio or video.

```bash
git clone https://github.com/MarkShi17/dopplertracker.git
cd dopplertracker
uv sync
```

Run the whole pipeline on generated data. No downloads needed.

```bash
# 1. Generate a synthetic recording with three flyovers
uv run dopplertrack demo --scenario rpm_drift --out outputs/demo

# 2. Search a region for the microphone
uv run dopplertrack search outputs/demo/spectrogram.npz \
  --events outputs/demo/events.json --traces outputs/demo/traces.npz \
  --bbox 39.995 40.005 -73.006 -72.994 --spacing 200 \
  --elevation 0 --raw-score --fast --out outputs/demo/search.json

# 3. Refine the best candidates and report an error region
uv run dopplertrack refine outputs/demo/spectrogram.npz outputs/demo/search.json \
  --events outputs/demo/events.json --traces outputs/demo/traces.npz \
  --elevation 0 --raw-score --method projected --smooth spline --source-degree 1 \
  --uncertainty --out outputs/demo/refined.json
```

The refined position lands about 1 m from the true one in `outputs/demo/truth.json`.
That file is written for scoring only; search and refine never read it.

## How it works

1. **Spectrogram.** Decode the audio into a high-resolution low-frequency
   spectrogram (40–450 Hz, 0.25 Hz bins).
2. **Find flyovers.** Detect the S-shaped Doppler curves and fit each one's speed,
   distance and time of closest approach.
3. **Get flight tracks.** Download public ADS-B history for that day and region.
4. **Search.** Score candidate locations and flights against the audio, with every
   flyover locked to one shared recording clock.
5. **Refine and validate.** Fit the location continuously, estimate an error region,
   and check that the match is supported by the audio rather than by noise.

Sound takes time to arrive, so the model solves *reception time = emission time +
travel time* and uses the aircraft's velocity at the moment the sound was emitted.

## Commands

| Command | What it does |
|---|---|
| `spectrogram` | Decode audio or video into a spectrogram |
| `detect`, `fit` | Find flyovers automatically, or fit one known window |
| `fetch-day`, `extract-region` | Download or extract regional ADS-B flight tracks |
| `fetch-weather` | Download the day's temperature and wind profile |
| `search`, `search-single` | Score candidate locations against real flight tracks |
| `refine` | Continuously refine the best candidates |
| `discover` | Look for further flyovers that a fitted location predicts |
| `validate` | Check a saved candidate against frequency-shifted null curves |
| `demo` | Write synthetic inputs for the quick start |
| `benchmark`, `stress-benchmark`, `controls-benchmark` | Reproduce the method comparisons |

Run `uv run dopplertrack <command> --help` for options.

## Use it on a real recording

```bash
uv run dopplertrack spectrogram recording.mp4 data/recording.npz
uv run dopplertrack detect data/recording.npz --detector calibrated --out data/events.json
uv run dopplertrack fetch-day 2026-09-08 data/traces.npz --bbox 40.4 41.3 -74.4 -72.6
uv run dopplertrack search data/recording.npz --events data/events.json \
  --traces data/traces.npz --bbox 40.4 41.3 -74.4 -72.6 --spacing 200 \
  --terrain --geoid-offset -30 --fast --out outputs/search.json
uv run dopplertrack refine data/recording.npz outputs/search.json \
  --events data/events.json --traces data/traces.npz \
  --terrain --geoid-offset -30 --method projected --smooth spline --source-degree 1 \
  --uncertainty --out outputs/refined.json
```

A daily flight archive is several GB. Check the detected flyovers before a large
search, aim for three or more of them, and replace `-30` with your local geoid offset.
The [usage guide](docs/usage.md) covers each step, the optional accuracy settings and
the file formats.

## Reproduce the benchmarks

```bash
# Original synthetic study: 27 methods x 20 conditions x 5 seeds (about 7 minutes)
uv run dopplertrack stress-benchmark --split held-out --workers 4 \
  --seeds 100 101 102 103 104 --out outputs/stress

# Final evaluation of the newer methods on unseen seeds (about 40 minutes)
uv run python scripts/final_heldout_evaluation.py

# Tests
uv run pytest
```

The real-recording studies need the audio and multi-GB flight archives. See
[how to reproduce them](docs/campaign-results.md#reproduce).

## Limitations

- **Two real recordings.** One was used for development; the other gave errors near
  1 km. Field accuracy is not established.
- **Geometry decides accuracy.** Overhead passes and single flyovers constrain the
  location poorly. The error region reports this, but it is too confident when the
  tone fits badly.
- **Propeller aircraft only.** Jets have no steady tone to track, and twin-engine
  aircraft may produce two close tones that the tracker does not model.
- **Stationary microphone.** Moving recorders are not supported.
- **No guarantee of a correct match.** The evidence check and detection scores are
  heuristics, not calibrated probabilities.

## Documentation

- [Usage guide](docs/usage.md): full workflow, options and file formats
- [Improvement campaign](docs/campaign-results.md): nine changes, what each did, and what failed
- [Real-recording study](docs/oyster-bay-results.md): the first Oyster Bay results and failed experiments
- [Synthetic accuracy study](docs/accuracy-expansion.md): 27 methods, 20 conditions
- [Original experiments](docs/experiments.md): the first seven-method comparison
- [Design notes](docs/design.md): physics and modelling choices

## Credits

This is an independent reimplementation of the method shown in
[colsto's video](https://www.youtube.com/watch?v=UytYrhYLOKI) and the original
[dopplertrack](https://github.com/mjshiggins/dopplertrack) project. It is not a
fork, and its file formats are not compatible with that project. The 43 m and
592 m field results quoted there belong to that project.

- Flight data: [adsb.lol historical archive](https://github.com/adsblol/globe_history_2024),
  under the ODbL. Follow its terms when redistributing data.
- Weather profiles: [Open-Meteo](https://open-meteo.com/). Geoid heights: NOAA GEOID18.
- Aircraft type list: the OpenSky mirror of ICAO Doc 8643.
- [readsb trace format](https://github.com/wiedehopf/readsb/blob/dev/README-json.md#trace-jsons)
- [Terrarium elevation tiles](https://github.com/tilezen/joerd/blob/master/docs/formats.md)
- [SciPy robust least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
