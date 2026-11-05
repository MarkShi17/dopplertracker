# Expanded accuracy experiments

**Later real-audio correction:** the Oyster Bay recording exposed gain-dependent
whitening, which has now been fixed. The archived waveform and detection control
numbers below describe the earlier preprocessing version. They have not been
rerun or relabeled as results of the correction. The main stress matrix receives
generated spectra directly and does not call whitening. See the
[real-recording status](video-reference-status.md).

The expanded study compares 27 methods across 20 synthetic conditions. Development
used seeds 0–1 (1,080 fits). The final evaluation uses seeds 100–104 (2,700 fits).
Spline-smoothed linear source fitting was selected for emphasis from development
results, before examining final evaluation errors. All methods remain in the report.

Open the [interactive comparison](results/stress/report.html), [full JSON](results/stress/results.json),
or [per-trial CSV](results/stress/trials.csv). The original seven-method study is
retained in [experiments.md](experiments.md).

## Final evaluation results

Each method receives the same 95 signal scenes and five noise-only scenes. All
2,700 fits returned numeric estimates and converged, with no exceptions. The run
took 402.9 seconds with four workers. The table below reports location errors
before the separate evidence gate, preserving difficult and rejected cases.

| Method | Median error (m) | 90th percentile (m) | Within 100 m | Median fit time (s) |
|---|---:|---:|---:|---:|
| Continuous brightness | 1.6 | 232.7 | 81.1% | 2.13 |
| Continuous contrast | 2.4 | 240.0 | 81.1% | 2.91 |
| Existing ridge + drift | 9.1 | 117.9 | 89.5% | 0.35 |
| Projected linear source | 3.9 | 128.8 | 86.3% | 0.17 |
| **Spline + linear source** | **3.9** | **45.5** | **94.7%** | **0.17** |
| Kalman + linear source | 3.9 | 45.9 | 95.8% | 0.20 |
| Spline + blocked source selection | 2.3 | 39.2 | 96.8% | 1.36 |
| Spline + all physical corrections | 2.5 | 107.5 | 87.4% | 0.37 |

The development-selected spline/linear preset reduces the 90th-percentile error
by **80.5%** against continuous brightness. It is a substantial tail-error
improvement, not a universal win: brightness has the better overall median.
On noisy ADS-B scenes the median improves from 240.0 to 27.8 m; with linear RPM
drift it improves from 78.5 to 0.3 m. On combined errors it improves from 221.8 to
78.4 m. Conversely, corrupted-event median error grows from 0.5 to 9.7 m, and
mismeasured-event error from 0.5 to 25.3 m. Candidate-guided ridge extraction can
inherit bad event estimates even when direct spectral matching succeeds.

Spline/blocked selection has an encouraging 39.2 m tail and a 120.6 m worst case,
but costs about eight times the median fit time of the spline/linear preset.
These are descriptive comparisons on this evaluation set; selecting new settings
from this table would require new validation data. Enabling every correction
increases the worst error to 598.0 m. Event consensus reaches 794.0 m in its worst
case, and five-start fitting does not consistently improve the tail. These methods
remain experimental options, with existing CLI defaults unchanged.

The spline/linear evidence gate rejects 18 of 95 signal scenes. Of the remaining
77, 74 are within 100 m; three accepted estimates are 117.1, 146.3 and 244.9 m off.
This is why an accepted curve match must not be presented as a guaranteed location.
Every method rejects all five noise-only scenes. The 135 method/noise evaluations
reuse the same five scenes and must not be treated as 135 independent recordings.

The [development location summary](results/development-location-summary.json)
records the original 1,080 fits. Its obsolete evidence decisions are deliberately
excluded; the harmonic-coverage correction affects acceptance, not the fitted
locations. All final acceptance results above use the corrected check.

## What changed

| Family | Experiments and purpose |
|---|---|
| Signal matching | Continuous brightness and contrast baselines, existing ridge/drift fits, independent frame peaks versus Viterbi ridges, logarithmic power |
| Source model | Constant, linear and quadratic source frequency; eliminate their linear coefficients inside the nonlinear location fit; compare least squares, soft-L1, Huber and Cauchy losses |
| Flight measurements | Cubic smoothing splines and Kalman/RTS smoothing before differentiating flight positions; compare constant/linear/quadratic source fits |
| Physical corrections | Bounded sound speed, horizontal wind, common aircraft altitude and recording clock-rate corrections; also test a joint model |
| Search/model selection | Blocked source-model selection, leave-one-event-out consensus, five deterministic initial positions, and two/three/four-event ablations |

The first-principles motivation is that Doppler depends on **velocity toward the
microphone**, not merely aircraft position. Differentiating noisy ADS-B positions
amplifies their errors. Smoothing before differentiation can therefore improve
location estimates even when the spectral ridge is measured precisely. Separately,
propeller RPM changes affect source frequency and can masquerade as motion. For
fixed geometry, polynomial source-frequency coefficients enter the prediction
linearly; solving that small regression inside the location optimizer reduces the
nonlinear search dimension.

More parameters do not automatically improve accuracy. Wind, source drift, altitude,
clock error and position can compensate for each other. Corrections are bounded,
regularized and optional. Blocked validation chooses among source models using
recorded spectra rather than true locations, preferring the simpler supported model.
Its peak extraction is independent per frame to prevent Viterbi smoothing from
leaking validation information into training blocks.

## Evaluation design

The main generator independently constructs accelerating, turning and climbing
flight paths, solves acoustic travel time, and obtains received frequency from
emitted phase versus reception time. It does not call the production predictor.
The inference path receives only spectra, flight tracks, event estimates and a
fixed region-center initial candidate. True locations and scenario labels are
available only to evaluation.

The 20 conditions cover clean/curved paths, linear/nonlinear RPM changes, wind,
sound speed, noisy ADS-B positions, altitude bias, clock drift, track gaps, low SNR,
broadband interference, crossing tones, missing harmonics, a corrupted event,
overhead geometry, combined errors, noise only, parallel paths and mismeasured
event parameters. Evaluation seeds are new, but these scenario families were
already known during development.

**This is conditional localization:** flight identities and approximate event
windows/parameters are seeded. It is not a blind nationwide test. Synthetic
conditions do not include full weather profiles, acoustic multipath, all microphone
processing or realistic correlated ADS-B error. These synthetic results do not
reproduce the original `dopplertrack` README's 43 m / 592 m field results. The subsequent
[Oyster Bay study](oyster-bay-results.md) separately evaluates the actual video.

The listener is assumed stationary. Refinement also cannot recover a true flight
pair already discarded by coarse detection or search filters; the separate detector
and identification checks expose only part of that upstream risk.

Every attempted fit remains in the report. Numeric error quantiles include
nonconverged estimates; exceptions remain in attempt counts and the denominator
for the fraction within 100 m. Noise-only cases have no meaningful location error
and are evaluated separately. Timings measure these runs on this machine, with
concurrent workers, rather than portable speed claims.

## Separate pipeline checks

The [detector experiment](results/controls/detection.json) supplies no event windows
to detection. Proposals are matched one-to-one to the generator's unjittered CPA
times afterward, within 10 seconds. There are 24 actual events across the signal
scenes, plus two noise-only scenes per method:

| Detector | Recovered events | Recall | Proposal precision |
|---|---:|---:|---:|
| Raw, one harmonic | 10 / 24 | 41.7% | 31.3% |
| Whitened, one harmonic | 14 / 24 | 58.3% | 45.2% |
| Whitened, two harmonics | 19 / 24 | 79.2% | 59.4% |

There is no detector abstention threshold in this experiment: noise-only proposals
count as false proposals. Harmonics help here but do not make detection reliable
enough to accept every proposed event without validation.

The [flight-identification check](results/controls/identification.json) runs actual
coarse searches with six distractor tracks, a 5 × 5 regional grid, and no supplied
recording clock. Both brightness and contrast rank the correct pair first in all
nine signal trials; each rejects the three noise-only winners. These are small,
seeded-event searches, not evidence of national-scale identification performance.

The [waveform experiment](results/controls/audio.json) independently generates
phase-integrated audio with three flyovers, harmonics, hum and noise, then runs
STFT, whitening and location fitting. All 32 fits converged. Clean-audio median
error is 1.53 m; noisy-audio median is 3.59 m. Increasing the STFT window from 1 to
8 seconds improves combined median error from 3.42 to 2.15 m but raises maximum
error from 6.58 to 8.74 m and spectrum memory from 1.42 to 11.22 MiB. More frequency
resolution trades against temporal smearing and memory; it is not uniformly better.
Flights and windows are supplied for these waveform fits.

The [frozen evidence controls](results/controls/evidence.json), using additional
seeds 200–219, accept all 20 true fixtures and reject all 20 examples of each of
noise, broadband interference, wrong position and wrong clock. These fixtures
intentionally use the production predictor to isolate validation logic; independent
physics is exercised by the stress and waveform generators. The counts are not a
calibrated false-match probability, particularly after searching many candidates.

## Evidence checks and regression fixes

Refinement now returns a separate heuristic evidence decision. It compares the
frozen fitted curve against 22 frequency-shifted alternatives, checks narrow-band
support in every event, and requires sufficient audio and trajectory coverage.
Available harmonics are normalized per frame, with exactly the same harmonic mask
for the center and all null curves. Losing an out-of-band second harmonic no longer
rejects an otherwise well-observed fundamental. Missing audio at the ends or inside
a recording cannot count as support. Thresholds were not tuned on evaluation seeds.

The development run exposed the harmonic-coverage bug; the final evaluation was
rerun after correcting it. A later internal-audio-gap regression adds a coverage
factor that is exactly one for all regularly sampled benchmark spectra, and thus
does not change their results. Saved spectral refinements now also retain the
atmosphere used by their fit, so subsequent validation reconstructs the same model.
The superseded local report is not the published final report.

New scientific tests also enforce coordinate-rotation invariance of curved-flight
interpolation, finite parameter checks, correct gap splitting, malformed readsb
handling, altitude reference conversions and terrain tile boundaries. Componentwise
PCHIP interpolation violated rotation invariance; vector Hermite interpolation
with linear vector derivatives fixes that demonstrated inconsistency.

Final verification: **208 tests passed**, Ruff lint and formatting passed, and both
wheel and source distribution built successfully. CLI smoke checks exercised
projected refinement, blocked automatic selection and saved-candidate validation.

## Reproduce or use the new methods

```bash
.venv/bin/dopplertrack stress-benchmark --split held-out --workers 4 \
  --seeds 100 101 102 103 104 --out outputs/stress
.venv/bin/dopplertrack controls-benchmark --out outputs/controls
.venv/bin/python -m pytest -q
```

Use `uv run` instead of `.venv/bin/` after a normal `uv sync`. The interactive report
can filter by scenario and method name. The JSON retains source hashes at run start,
dependency versions, options, errors, convergence flags, evidence decisions and
timings. Formatting and CLI/documentation changes after run start can change file
hashes without changing the recorded inference configurations.

For an existing search result, try `refine --method projected --smooth spline
--source-degree 1`, or `--method auto --smooth spline` for blocked source-model
selection. Complete runnable examples are in the [usage guide](usage.md).

Subsequent work used the actual Oyster Bay recording and matching flight archive.
The [real-video study](oyster-bay-results.md) reports 39.6 m for corrected spline/
linear refinement versus 102.4 m for our brightness baseline, plus a 311-pair
shortlist comparison and sensitivity tests. It also documents the irregular-
sampling bug exposed by real data. These are development results on the known
example, not independent held-out field accuracy. The synthetic reports above
retain their original source hashes and verification counts.
