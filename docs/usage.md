# Usage guide

Full command walkthrough, option notes and file formats. For a short overview, see
the [README](../README.md).

## Setup

```bash
uv sync
```

Python 3.12 is selected by `.python-version`; Python 3.11+ is supported. `ffmpeg`
must be on PATH for audio/video input. The benchmarks and the synthetic demo need
no network or credentials.

## Synthetic search

1. Create inputs with source-frequency drift:

   ```bash
   uv run dopplertrack demo --scenario rpm_drift --out outputs/demo
   ```

2. Search a region without supplying its true location or recording start time:

   ```bash
   uv run dopplertrack search outputs/demo/spectrogram.npz \
     --events outputs/demo/events.json --traces outputs/demo/traces.npz \
     --bbox 39.995 40.005 -73.006 -72.994 --spacing 200 \
     --elevation 0 --raw-score --out outputs/demo/search.json
   ```

3. Refine the best competing spatial modes:

   ```bash
   uv run dopplertrack refine outputs/demo/spectrogram.npz outputs/demo/search.json \
     --events outputs/demo/events.json --traces outputs/demo/traces.npz \
     --elevation 0 --raw-score --fit-drift --top 3 \
     --out outputs/demo/refined.json
   ```

`truth.json` is written for evaluation only. Neither search nor refinement reads it.
The three input tracks in this demo are the true flights, so this is a pipeline
smoke test, not a test against thousands of distractor aircraft.

## Real-recording workflow

1. Decode the original recording:

   ```bash
   uv run dopplertrack spectrogram recording.mp4 data/recording.npz
   ```

2. Detect two candidate flyovers, or fit known windows:

   ```bash
   uv run dopplertrack detect data/recording.npz --band 60 300 --top 2 --out data/events.json
   # Or fit known windows directly (these are the two Oyster Bay flyovers):
   uv run dopplertrack fit data/recording.npz --window 2090 2200 --band 95 200 --harmonics 2 --out data/event1.json
   uv run dopplertrack fit data/recording.npz --window 3020 3130 --band 95 195 --harmonics 2 --out data/event2.json
   ```

   Detection scores are suggestions, not proof of aircraft. Check the event windows
   and sensible speed/distance ranges before a large search. Search consumes **all**
   events in the provided files; do not pass a top-40 list containing false positives.
   `load_events` and `search` order events chronologically even though detection
   prints its candidates in score order.

3. Obtain regional flight tracks for the UTC date:

   ```bash
   uv run dopplertrack fetch-day 2026-09-08 data/traces.npz \
     --bbox 40.4 41.3 -74.4 -72.6
   # Or use a previously downloaded archive:
   uv run dopplertrack extract-region day.tar data/traces.npz \
     --bbox 40.4 41.3 -74.4 -72.6
   ```

   Daily archives can transfer several GB. Only regional segments are saved, but
   the full archive must be read. The default 15 km padding must cover the maximum
   audible flight distance. `--jet-types jets.json` accepts a JSON list of known
   jet ICAO codes, such as `["A320", "B738"]`; without it, all aircraft types remain.
   Aircraft classification data is not bundled or automatically downloaded.

4. Search, using terrain and the local geoid correction:

   ```bash
   uv run dopplertrack search data/recording.npz \
     --events data/event1.json data/event2.json --traces data/traces.npz \
     --bbox 40.4 41.3 -74.4 -72.6 --spacing 200 \
     --terrain --geoid-offset -30 --out outputs/search.json
   ```

   `-30` is an **illustrative correction, not a measured value for this recording**.
   Replace it with the local geoid undulation, or provide an appropriately corrected
   local DEM. `--point LAT LON` scores a known location instead of searching a grid.
   Silence filtering defaults off because missing aircraft audio is uncertain;
   `--silence-radius METERS` enables a strict filter when justified by the recording.

5. Compare refinement methods:

   ```bash
   uv run dopplertrack refine data/recording.npz outputs/search.json \
     --events data/event1.json data/event2.json --traces data/traces.npz \
     --terrain --geoid-offset -30 --method brightness --out outputs/brightness.json

   uv run dopplertrack refine data/recording.npz outputs/search.json \
     --events data/event1.json data/event2.json --traces data/traces.npz \
     --terrain --geoid-offset -30 --fit-drift --fit-sound-speed --out outputs/joint.json
   ```

   `--method contrast` provides the local spectral-contrast alternative.
   `--extra event3.json event4.json` assigns additional events to flights at each
   candidate, then locks them to the same clock during refinement. These assignments
   are conditional hypotheses, not independent confirmation. Distinct candidate
   modes are preserved with `--top`, rather than refining only one grid winner.

For a single flyover with a known recording start:

```bash
uv run dopplertrack search-single data/recording.npz --events data/event1.json \
  --traces data/traces.npz --bbox 40.4 41.3 -74.4 -72.6 --terrain \
  --recording-start 2026-09-08T20:17:30Z --clock-tolerance 2 \
  --out outputs/single.json
```

`--recording-start` is the UTC time at **audio time zero**, not the flyover time.
Date boundaries need flight data from both UTC days. Combine the imported track
lists through the Python API if a recording spans midnight.

## Refinement methods

The projected fitter separates source-frequency regression from the nonlinear
location search. To try the spline-smoothed linear source model on an existing candidate:

```bash
uv run dopplertrack refine outputs/demo/spectrogram.npz outputs/demo/search.json \
  --events outputs/demo/events.json --traces outputs/demo/traces.npz \
  --elevation 0 --raw-score --method projected --smooth spline --source-degree 1 \
  --out outputs/demo/projected.json
```

`--method auto` chooses constant, linear or quadratic source frequency using blocked
validation and a preference for simpler supported models. `--method consensus`
compares all-events and leave-one-event-out fits (at least three events).
`--smooth kalman` provides an alternative trajectory smoother. Projected methods
also expose `--fit-wind`, `--fit-altitude`, `--fit-clock-rate`, and robust `--loss`
choices. These extra parameters can imitate location changes; enabling all of them
is not automatically more accurate. Existing defaults are preserved.

## Evidence checks

Every refinement includes a heuristic `evidence` decision. `validate` can reassess
saved candidates with the same spectrogram, event files and tracks. It compares
the fitted curves with nearby frequency-shifted curves, checks per-event support,
and accounts for unavailable harmonics and missing audio/track coverage. An accepted
match is **not** a calibrated probability or a guarantee of correct geolocation.

```bash
uv run dopplertrack validate outputs/demo/spectrogram.npz outputs/demo/projected.json \
  --events outputs/demo/events.json --traces outputs/demo/traces.npz --raw-score \
  --out outputs/demo/validated.json
```

## Newer optional settings

All of these are off by default, so older commands give the same results as before.
The [improvement campaign](campaign-results.md) reports what each one did on synthetic
and real data. In short: the detector and the fast search are recommended; the
accuracy settings help on synthetic data and are unproven on real recordings.

**Calibrated detector.** Finds flyovers over the whole recording, scores each against
null curves, and abstains below `--min-z` (default 12). Slower than the old detector.

```bash
uv run dopplertrack detect data/recording.npz --detector calibrated --out data/events.json
```

The score is not a probability; expect an occasional false detection. Three or more
flyovers identify the aircraft far more reliably than two.

**Fast regional search.** Same candidates and scores as `search`, computed for all grid
points at once. `--jet-types` takes a JSON list of ICAO type designators to exclude;
`scripts/experiments/make_jets.py` builds one from the OpenSky copy of ICAO Doc 8643.

```bash
uv run dopplertrack search data/recording.npz --events data/events.json \
  --traces data/traces.npz --bbox 40.4 41.3 -74.4 -72.6 --spacing 200 \
  --terrain --geoid-offset -30 --fast --jet-types data/aircraft-types/jets.json \
  --out outputs/search.json
```

**Measured atmosphere.** Downloads the day's temperature, humidity and wind by height
from Open-Meteo for the region centre, then propagates sound through those layers
instead of assuming one sound speed. `--weather` replaces `--sound-speed` and `--wind`.

```bash
uv run dopplertrack fetch-weather 2026-09-08 data/weather.json \
  --lat 40.85 --lon -73.5 --geoid-offset -30.854
uv run dopplertrack refine data/recording.npz outputs/search.json \
  --events data/events.json --traces data/traces.npz --terrain --geoid-offset -30 \
  --method projected --source-degree 1 --weather data/weather.json \
  --out outputs/layered.json
```

Layered fits are about ten times slower. `validate --weather` checks such candidates
with the same atmosphere.

**Reported aircraft velocity.** `fetch-day` and `extract-region` now keep each
aircraft's broadcast ground speed, track and vertical rate. `--velocity fused` combines
them with the positions in a Kalman smoother; `--velocity reported` uses them directly.
Trace files made by older versions carry no velocities and fall back to positions.

```bash
uv run dopplertrack refine data/recording.npz outputs/search.json \
  --events data/events.json --traces data/traces.npz --terrain --geoid-offset -30 \
  --method projected --source-degree 1 --velocity fused --out outputs/fused.json
```

**Error region.** `--uncertainty` (with `--method projected`) adds 50% and 90% ellipses
and a breakdown of what causes the uncertainty: measurement noise, atmosphere, ADS-B
timing, altitude, observer height and velocity.

```bash
uv run dopplertrack refine outputs/demo/spectrogram.npz outputs/demo/search.json \
  --events outputs/demo/events.json --traces outputs/demo/traces.npz \
  --elevation 0 --raw-score --method projected --smooth spline --source-degree 1 \
  --uncertainty --out outputs/demo/refined.json
```

The region is too cautious on clean synthetic scenes and too confident when the tone
fits poorly. Read it together with the evidence decision.

**Flyover discovery.** Given a fitted candidate, looks for every other aircraft the
candidate predicts should be audible, and reports which ones appear in the audio.

```bash
uv run dopplertrack discover data/recording.npz outputs/refined.json \
  --traces data/traces.npz --out outputs/found.json --confirmation outputs/confirmation.json
uv run dopplertrack refine data/recording.npz outputs/refined.json \
  --events data/events.json --extra outputs/found.json --traces data/traces.npz \
  --terrain --geoid-offset -30 --method projected --out outputs/joint.json
```

A confirmed flyover is evidence of the right time much more than of the right place,
and the default 6 km range can miss distant low aircraft (`--max-distance`,
`--ground-range`).

**Audio demodulation and hum removal.** Available from Python only:
`ModelOptions(ridge="demod", noise="snr", hum="subtract")` with raw audio passed to
`fit_projected(..., audio=...)`. `scripts/compare_oyster_bay_measurement.py` is a
working example. It measures the tone far more precisely on synthetic audio but made
real locations worse, so it is not exposed in the command line.

## Physics model

The propagator solves reception time = emission time + travel time. Source velocity
comes from the derivative of a continuous trajectory at emission time. It supports
a constant sound speed and uniform ENU wind, rejects missing track coverage, splits
interpolation at gaps, and retains boundary closest approaches. In wind, the
acoustic timing anchor minimizes propagation duration; in still air it coincides
with geometric closest approach.

Spectral scoring uses the entire event window. Uncovered times contribute zero,
preventing a short bright fragment from winning by dropping its missing portion.

## Accuracy choices and limitations

- **Source drift:** fitting a bounded linear source-frequency drift per flight can
  separate changing propeller RPM from Doppler motion. It can also absorb location
  information; leave it off when it is unsupported by the data.
- **Atmosphere:** `--sound-speed` and `--wind EAST NORTH UP` accept measured conditions.
  `--fit-sound-speed` adds a bounded shared estimate with a weak prior. This is a
  homogeneous-atmosphere approximation, without refraction, wind shear or multipath.
- **Vertical reference:** readsb geometric altitude is ellipsoidal; terrain tiles
  approximate mean sea level. `ellipsoid = MSL + geoid undulation`. Pressure altitude
  is rejected unless `--allow-baro` explicitly enables an approximate constant
  `--baro-offset` correction. A constant geoid/pressure correction is inadequate
  over large regions or strongly changing weather.
- **Model selection:** contrast scoring did not consistently beat brightness in the
  experiments. Ridge extraction is guided by each coarse candidate and can inherit
  its bias. Overhead passes remain weakly informative. Neither an optimizer's
  convergence flag nor a high score establishes a correct location.
- **Uncertainty and scale:** results expose residuals, coverage, optimizer bounds
  and local geometric conditioning, not calibrated confidence radii. Earth-scale
  searches, 12-hour detection speed, and live terrain downloads have not been
  validated here. Resumable archive download and regional extraction have been
  exercised on the actual September 8 archive. Corridor pruning is implemented, but
  this Python reference search is not performance-equivalent to the 19-minute run
  quoted by the original project.

Use independent recordings with known answers and held-out flights before claiming
field accuracy. Real ADS-B position/altitude errors can dominate the synthetic errors.
The small differences on clean scenes are not evidence of universal superiority.
The model assumes a stationary microphone; moving-recorder localization is not implemented.

## File formats

- Spectrogram NPZ: `times` (seconds from recording start), `frequencies` (Hz),
  `power` (frequency × time). Saved audio spectra contain raw power; downstream
  commands whiten it once unless `--raw-score` is supplied.
- Event JSON: a list with `window`, `cpa` (heard time in recording seconds), `f0`
  (source Hz), `speed` (m/s), `distance` (slant meters), `harmonics`, optional
  `speed_range` and `distance_range`. Fits use generous defaults for the ranges.
- Trace NPZ: frame origin, JSON metadata, per-segment `t0/p0`, `t1/p1`, ...;
  positions are ENU meters and times are Unix seconds. No pickle is used.
- Result JSON: frame plus ranked `candidates`, with positions, geographic
  coordinates, shared start clock, flight IDs, source frequencies and scores.
- DEM NPZ: ascending `lat`, ascending `lon`, and `elevation_m[lat,lon]` in MSL meters.
  DEM interpolation refuses to extrapolate beyond supplied coverage.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```
