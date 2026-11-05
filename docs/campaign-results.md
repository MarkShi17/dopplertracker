# Improvement campaign, November 2026

Nine changes were built and tested against the previous recommended method
(spline-smoothed trajectory with a linear source model, "spline-linear"). This page
records what each one did on synthetic data and on two real recordings, including
the changes that did not help.

**Short version.** Automatic detection and the regional search are clearly better.
The tool can now go from raw audio to the right aircraft on the first recording
without hints. No change produced a demonstrated accuracy gain on real recordings:
on the first recording the answer moves by more than any one change is worth, and
on the second, unseen recording the pre-chosen new configuration did slightly worse
than the old one.

All new behaviour is opt-in. Existing defaults, method names and synthetic scenes
give the same numbers as before; 341 tests pass.

## Verdicts

| Change | Synthetic, unseen seeds | Recording 1 (development) | Recording 2 (unseen) | Status |
|---|---|---|---|---|
| Calibrated detector | finds 100% of flyovers (was 66%); no false detections (was 50%) | finds both known flyovers and a third real one | finds both flyovers | recommended |
| Fast regional search | same winner as the old search on 30 of 30 scenes | full region in 5–20 min (was about 32 h) | not run | recommended |
| Reported ADS-B velocity, fused | median 19.3 → 6.3 m on scenes with realistic tracks | better in 4 of 4 paired fits | worse in 6 of 8 | optional, unproven on real data |
| Measured layered atmosphere | median 16.2 → 6.7 m on layered scenes | worse in 4 of 4 paired fits; smaller error region | better in 5 of 8 (typically by 12 m) | optional, unproven on real data |
| Audio demodulation | frequency error 10–100 times lower | worse in 4 of 4 | worse in 7 of 8 | Python API only; not advised on real audio yet |
| Hum-line removal | worst-case frequency error 0.119 → 0.006 Hz | keeps 16–24 more frames per flyover; mixed effect on location | used in the demodulation fits | optional |
| Flyover discovery | median 1.4 → 0.7 m, tail unchanged | found a real third aircraft, only with the range limit raised | confirmed one pass, probably a false one | experimental |
| Error regions | 96–100% of truths inside the 90% region (too cautious); fails on gross misfits | truth inside 90% in 9 of 10 fits | truth inside 90% in 4 of 16 | experimental |
| Per-flight time offsets | help only when a timing fault is present; hurt elsewhere | 35.4 m and 59.8 m against 39.6 m | not run | not recommended |

"Paired fits" compare two fits that differ in one choice only.

## Synthetic results on unseen seeds

Seeds 300–304 (300–307 for detection) were never used during development. Everything
ran once from a plan written before the run
([plan](results/campaign/final-heldout-synthetic/plan.json),
[summary](results/campaign/final-heldout-synthetic/summary.txt)).

**Original 20 scenarios** (95 scenes with a signal per method):

| Method | Median | 90th percentile | Worst | Within 100 m |
|---|---:|---:|---:|---:|
| Brightness | 1.8 m | 204.4 m | 326.0 m | 85% |
| Spline-linear | 2.1 m | 45.0 m | 305.9 m | 95% |
| Fused velocity | 2.3 m | 46.5 m | 305.8 m | 96% |

These scenes carry no velocity reports, so the fused method runs its position-only
fallback here. Spline-linear is still worse than brightness on track gaps (median
134.9 m against 0.8 m) and on mismeasured or corrupted events.

**Extended scenarios** (35 scenes; all fits within 100 m):

| Method | Realistic-track scenes (20) | Layered-atmosphere scenes (15) |
|---|---:|---:|
| Spline-linear, constant 343 m/s | 19.3 m | 16.2 m |
| Fused velocity | 6.3 m | 12.8 m |
| Forecast atmosphere profile | n/a | 6.7 m |
| Fused velocity + forecast profile | n/a | 6.7 m |
| That, plus fitted sound-speed and wind offsets | n/a | 1.2 m |

Values are median errors. The forecast profile is deliberately wrong by 1.5 K and
3 m/s of wind, to imitate weather-model error. On the timing-fault scene the fused
method is no better (20.9 m both ways).

**Detection** (32 scenes, 96 flyovers):

| Detector | Found | Detections that are real | Timing error | Source-frequency error |
|---|---:|---:|---:|---:|
| Old | 66% | 50% | 2.39 s | 7.78 Hz |
| Calibrated | 100% | 100% | 0.13 s | 0.13 Hz |

The calibrated detector reported nothing on all 8 noise-only scenes. It is about six
times slower. These scenes share one geometry and differ only in noise, so the
perfect score overstates real performance.

**Discovery** (55 scenes with silent decoy aircraft): it found 87% of hidden flyovers,
and 91% of what it accepted was real. Median error went from 1.4 m to 0.7 m; the 90th
percentile went from 197.7 m to 209.0 m. Candidates with the wrong clock were never
confirmed (0 of 55), but candidates 1–5 km off in position were confirmed about half
the time (52 of 110). A confirmation is evidence of the right time far more than of
the right place.

**Audio measurement** (median location error and frequency-track error):

| Scene | Spectrogram ridge | Demodulation |
|---|---:|---:|
| Fast close pass | 19.0 m, 0.476 Hz | 0.02 m, 0.004 Hz |
| Low signal-to-noise | 25.4 m, 2.42 Hz | 10.1 m, 0.375 Hz |
| Clean | 1.21 m, 0.174 Hz | 0.01 m, 0.001 Hz |

These scenes have exact physics and exact flight tracks, so the demodulation numbers
are optimistic.

**Error regions:** 96% (original scenarios) and 100% (extended) of true positions fell
inside the 90% region, with typical misses far smaller than the region implies. The
regions are too cautious on clean scenes and too confident where the fit lands on a
wrong solution (track gaps: 40% inside 90%).

## Recording 1: Oyster Bay (development data)

Everything below was developed on this recording, so none of it is independent
evidence.

**Two flyovers, ten model variants from one shared start** (truth read only after all
fits; [results](results/campaign/combined/oyster-combined-results.json),
[error regions](results/campaign/uncertainty/oyster-uncertainty-summary.txt)):

| Atmosphere | Velocity | Measurement | Error | 90% region, half-widths |
|---|---|---|---:|---:|
| Constant | Differenced | Ridge | 39.6 m | 266 × 129 m |
| Constant | Fused | Ridge | 30.3 m | 230 × 126 m |
| Constant | Differenced | Demodulation | 92.7 m | 456 × 129 m |
| Constant | Fused | Demodulation | 56.9 m | 321 × 121 m |
| Layered | Differenced | Ridge | 74.5 m | 141 × 111 m |
| Layered | Fused | Ridge | 64.4 m | 109 × 97 m |
| Layered | Differenced | Demodulation | 220.1 m | 183 × 107 m |
| Layered | Fused | Demodulation | 78.8 m | 133 × 105 m |
| Layered, offsets fitted | Fused | Demodulation | 72.7 m | 250 × 123 m |
| Layered | Fused | Demodulation, 4 harmonics, all frames | 11.2 m | 112 × 99 m |

The error ranges from 11 m to 220 m while the fit quality barely changes. Two flyovers
do not pin this location down to tens of metres. The earlier headline of 39.6 m, and
the 11.2 m here, are single draws from that spread. The fused velocity roughly halved
the frequency misfit on the second flyover (0.89 → 0.48 Hz); the layered atmosphere
did not reduce misfit consistently.

**A third flyover.** The calibrated detector found a strong unlisted flyover at 783 s.
Discovery independently matched it to a Cessna Caravan 8.5 km away and 2.9° above the
horizon (predicted 782.0 s). Adding it did not stabilise the position:

| Events | Lowest | Median | Highest |
|---|---:|---:|---:|
| Two flyovers, published events | 11.2 m | 68.6 m | 220.1 m |
| Three flyovers, published + discovered | 33.8 m | 71.7 m | 225.8 m |
| Two flyovers, automatic detector | 27.4 m | 74.7 m | 126.4 m |
| Three flyovers, automatic detector | 27.7 m | 73.6 m | 119.2 m |

**From raw audio to a location.** With jets removed, the fast search covers the same
region the original project searched (about 380,000 grid points at 200 m spacing):

| Events given to the search | Correct aircraft ranked | Margin over runner-up |
|---|---:|---:|
| Published two flyovers | 4th of 1,747 | 0.014 |
| Detector's two known flyovers | 4th of 209 | 0.003 |
| Detector's three flyovers, no hints | 1st of 51 | 0.26 |

With two flyovers a wrong pair 53 km away wins narrowly. With three, the correct
aircraft win clearly, at 570 m on the coarse grid, and refining that winner gives
21.6 m (16.3 m with fused velocity). Ranking the refined fits by their own scores
instead picked a wrong pairing (442 m), because those scores average over different
numbers of flyovers. The rule that worked ranks coarse candidates by a common score
over all detections and was fixed before the truth was read.

## Recording 2: SW Florida eagle cam (unseen)

The protocol was frozen and written to disk before any fit
([plan](results/campaign/eagle-cam-heldout/plan.json),
[results](results/campaign/eagle-cam-heldout/results.json)). The chosen configuration
was fused velocity + layered atmosphere + spectrogram ridge. Fits start from the
original project's coarse answer, which is 915 m from the truth.

| Events | Old baseline | Chosen new configuration |
|---|---:|---:|
| Published detections | 802 m | 891 m |
| Automatic detector | 771 m | 917 m |

- All 16 two-flyover fits were between 771 m and 1,107 m off.
- The tool's own evidence check rejected every fit. On recording 1 it accepted them all.
- The truth was inside the 90% error region in 4 of 16 fits.
- Both aircraft pass nearly overhead (85 m and 1.1 km to the side, at 2.5 km and
  5.7 km altitude), which constrains ground position poorly. The tone fit was also
  poor: 1.6–3 Hz misfit on the first aircraft, against 0.1–0.6 Hz on recording 1.
- Discovery confirmed one extra pass, a business jet whose window overlaps the second
  aircraft's. It is probably spurious.
- The original project reports 1.7 km for its two-flyover refinement and 592 m with
  five flyovers.

One caveat: the true coordinates appear in the input-preparation script, and the
engineer running the test saw them before fitting. The plan was written before any
fit and no choice was changed afterwards.

## What the evidence supports

- **Use the calibrated detector and the fast search.** Both are large, measured gains,
  and together they make an unaided run possible.
- **Use three or more flyovers to identify the aircraft.** The third flyover did not
  sharpen the position on recording 1, but it turned a narrow wrong win into a clear
  right one.
- **Treat any single location error as one draw.** On recording 1 the realistic figure
  is about 70 m with a 90% region of 100–270 m.
- **Trust the evidence check more than the score.** It separated the roughly 70 m
  recording from the roughly 900 m one.
- **Fused velocity and the layered atmosphere are physically motivated and help on
  synthetic data, but are unproven on real recordings.** Two recordings cannot settle it.

## Known problems

- Error regions are too confident when the tone fit is poor, and the measurement part
  is 1.4–2 times too small when misfit is strongly correlated in time.
- Demodulation follows a wrong physics model more faithfully than the spectrogram
  ridge does, so it made real locations worse.
- Discovery's 6 km default range excluded the one real extra aircraft on recording 1,
  and it accepted a jet on recording 2.
- The detector's score is not a probability: the first recording played backwards
  yields two false detections per hour.
- Twin-engine aircraft may produce two close tones; the single-tone tracker does not
  model that. This is a suspected cause of the poor fit on recording 2, not a tested one.
- Layered fits are about ten times slower.

## Reproduce

Run from the repository root, in this order. Each script lists its inputs and stages
under `--help`, writes its plan before fitting, and reads the truth only at the end.

```bash
# Synthetic, unseen seeds (about 40 minutes, one process)
uv run python scripts/final_heldout_evaluation.py

# Recording 1
bash scripts/fetch_video_example.sh
uv run python scripts/experiments/make_jets.py
uv run python scripts/extract_oyster_bay_velocity.py
uv run python scripts/compare_oyster_bay_atmosphere.py
uv run python scripts/compare_oyster_bay_combined.py
uv run python scripts/compare_oyster_bay_three.py
uv run python scripts/oyster_bay_uncertainty.py
uv run python scripts/search_oyster_bay.py detect   # then: fast, rank-auto, refine, evaluate

# Recording 2
bash scripts/fetch_eagle_cam_example.sh
uv run python scripts/extract_eagle_cam_velocity.py
uv run python scripts/eagle_cam_heldout.py prepare  # then: fit, evaluate
```

Saved result files for every experiment are indexed in
[results/campaign/INDEX.md](results/campaign/INDEX.md).
