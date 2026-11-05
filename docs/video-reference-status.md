# Actual video reference: evaluation complete

The original audio and historical ADS-B inputs are now downloaded and processed.
See [the real-video results](oyster-bay-results.md) for all methods, sensitivity
tests, candidate rankings, caveats and reproduction commands.

| Check | Result |
|---|---|
| Real recording | Original one-hour `4CxVxuMVkMc` clip, 7,193 STFT frames |
| Historical tracks | Verified 3.9 GB split archive, 13,077 regional segments |
| Conditional brightness refinement | 102.4 m error |
| Corrected spline/linear refinement | 39.6 m primary; 39.6–86.0 m across assumptions |
| Published baseline | 43.3 m from saved results; original algorithm not rerun |
| Published shortlist refinement | Correct pair first of 311 usable pairs with spline method |

The shortlist contained 328 pairs; 17 lacked sufficient geometric-altitude data.
The experiment uses published event windows and candidate positions. It is not a
fresh blind regional search or automatic end-to-end detection benchmark.

The previous missing-track blocker is resolved. Downloads now persist `.part`
files, resume HTTP byte ranges, and verify release sizes and SHA256 hashes before
parsing. Rerunning `bash scripts/fetch_video_example.sh` reuses the cached audio and
verified archive parts. The pinned production receiver archive is recorded in
the [input manifest](results/oyster-bay/inputs.json); its preparation-time status
predates the completed evaluation.

Real inputs exposed gain-dependent whitening and an irregular-report timing bug
in the trajectory noise estimator. Both are corrected and covered by regression
tests. The failed pre-correction experiment is preserved in the results rather
than presented as successful validation.
