# Dopplertracker implementation and experiment plan

Build a local Python CLI, matching the workflow in colsto's video. This is a new
implementation, not the original `dopplertrack` repository. Input formats and numerical
scores are not promised to be compatible with that repository.

1. Implement and test retarded-time propagation, continuous trajectories, geographic
   conversion, audio spectrograms, and archive/elevation import.
2. Implement automatic ideal-curve detection, coarse candidate filtering and
   joint, clock-locked scoring of actual trajectories.
3. Compare brightness, spectral contrast, and robust ridge fitting. Refine location,
   a common recording clock, per-event source frequency and optional frequency drift.
   Expose sound speed and uniform wind rather than treating 343 m/s as ground truth.
4. Run reproducible synthetic experiments with noise, outliers and model mismatch;
   report errors for every method, not just winning trials. Document real-data use.

Accuracy is an empirical claim: synthetic experiments establish numerical behavior,
not accuracy on the two recordings in the original `dopplertrack` README. Those files are not
provided. No precision claim from that README is inherited by this project.

Physics: all coordinates are meters in a local ENU frame derived from WGS84 ECEF.
For uniform wind w and sound speed c, travel time tau solves
|observer - source - w*tau| = c*tau. Evaluate source position and velocity at emission
time, not reception time. Frequency is f0/(1 + grad_source(tau) dot velocity).
Do not extrapolate flight tracks or bridge missing ADS-B intervals.

Use a supplied DEM or cached public Terrarium tiles; constant elevation must be
explicit. GNSS height is ellipsoidal, terrain is approximately mean sea level,
and pressure altitude is neither. Expose geoid and barometric corrections; retain
provenance. Unknown aircraft types remain candidates; drop only known jets.

Search uses coarse speed/slant-distance/time-offset filters followed by scoring.
Silence filtering is optional: an unrecorded aircraft is not proof it was absent.
All event times share one clock. Missing curve coverage contributes zero to the
full-window score. Keep multiple flight pairs and spatial modes for refinement.

Refinement's local Jacobian describes geometric conditioning only. It is not a
calibrated confidence interval: correlated STFT bins, altitude error and weather
can dominate. Report residuals, bounds, coverage and conditioning.

Sources checked during implementation:
- https://github.com/wiedehopf/readsb/blob/dev/README-json.md#trace-jsons
- https://github.com/adsblol/globe_history_2024
- https://github.com/tilezen/joerd/blob/master/docs/formats.md
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
