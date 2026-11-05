# Accuracy experiments

Run: `.venv/bin/dopplertrack benchmark --out outputs/benchmark`.

These are synthetic localization tests with known flight assignments, not validations of the public recordings or of a nationwide search. The generator supplies event windows and coarse curve estimates. Ground truth is never supplied to location optimization.

## Median horizontal error by scenario (meters)

| Scenario | Continuous brightness | Continuous contrast | Robust ridge | + RPM drift | + Drift and sound speed |
|---|---:|---:|---:|---:|---:|
| clean | 2.2 | 2.2 | 1.7 | 3.5 | 3.4 |
| interference | 2.1 | 1.5 | 1.7 | 2.0 | 1.7 |
| rpm_drift | 87.8 | 87.5 | 100.1 | 2.3 | 2.3 |
| overhead | 9.2 | 7.8 | 21.2 | 33.5 | 33.5 |
| sound_speed_error | 91.2 | 91.9 | 63.1 | 19.4 | 1.9 |

Three seeds per scenario; all seven methods also appear in the [complete HTML report](results/benchmark.html) and [raw JSON](results/benchmark.json). Both grid baselines use 200 m spacing. Continuous baselines fit position, one shared clock, and each source frequency, making them a stronger comparison than grid rounding alone.

## What the experiments support

1. Continuous brightness fitting already works well on clean, ideal signals. A more complex scorer is not automatically better.
2. Unmodeled RPM drift creates systematic position bias here. Fitting linear drift substantially reduces it in the constructed drift scenes.
3. Fitting sound speed helps when the generator uses 335 m/s and inference starts at 343 m/s. Independent weather observations are preferable when available.
4. Additional fitted parameters worsen the overhead cases. Poor flight geometry remains a limitation; the ridge solver is also conditioned on its coarse candidate.
5. Spectral contrast does not consistently improve on brightness. It remains an experimental option.

## Evidence and limits

The generator uses an analytic straight-flight emission-time solution, independently of the interpolated propagator. It adds random spectral noise and interference. This avoids the most direct inverse-crime test, but it still shares important ideal assumptions with inference: exact trajectories, constant wind-free atmosphere, known event windows, no multipath, and synthetic frequency ridges rather than full acoustic propagation.

The separate test suite decodes an actual generated WAV with ffmpeg and fits its Doppler tone. Another smoke run searches the synthetic geographic grid, then refines the same assigned flights without reading the truth file. These checks establish that the pipeline connects; they do not establish performance against thousands of real distractor flights.

Scores are arbitrary evidence units, not probabilities. The local geometry diagnostic is not a confidence radius. Report all candidate modes, residuals and boundary/coverage flags when interpreting real data.

No audio from the two public recordings was available locally. Package-index DNS was blocked in this environment; scientific dependencies came from the existing local cache. Live archive and terrain downloads were not exercised. Their decoders were tested with fixtures matching the published schemas.

## Environment

- Python 3.12.13
- numpy 2.5.3
- scipy 1.18.1
- pillow 12.3.0

Optimizer convergence: 105/105 method/trial combinations. Non-convergence is preserved in the raw report rather than silently discarded.
