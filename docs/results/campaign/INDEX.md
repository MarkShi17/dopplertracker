# Campaign result files

Copied from finished worktrees (.claude/worktrees/agent-<ID>/outputs/); local paths scrubbed. Descriptions come from each file's own keys/first line.

| Destination file | Source worktree ID | Source path | Size (bytes) | Description |
|---|---|---|---|---|
| atmosphere/oyster-atmosphere-comparison-results.json | a6c5a2cdb7b469feb | outputs/oyster-bay/atmosphere-comparison/results.json | 233252 | JSON object; top-level keys: plan, evaluation_truth, trials |
| atmosphere/oyster-atmosphere-comparison-plan.json | a6c5a2cdb7b469feb | outputs/oyster-bay/atmosphere-comparison/plan.json | 11015 | JSON object; top-level keys: validation, conditions, methods, weather_file, weather_metadata, layered_profiles, initial_latlon, initial_clock, ... |
| atmosphere/layered-benchmark-dev-results.json | a6c5a2cdb7b469feb | outputs/layered-benchmark-dev/results.json | 49522 | JSON object; top-level keys: validation, split, seeds, scenarios, methods, method_options, method_atmosphere, seconds, ... |
| velocity/oyster-velocity-run1-unsliced-results.json | a8b646833fc7a0d36 | outputs/oyster-bay/velocity-comparison-run1-unsliced-runs/results.json | 62495 | JSON object; top-level keys: plan, evaluation_truth, trials |
| velocity/oyster-velocity-run1-unsliced-plan.json | a8b646833fc7a0d36 | outputs/oyster-bay/velocity-comparison-run1-unsliced-runs/plan.json | 5352 | JSON object; top-level keys: validation, protocol, traces, methods, heights_ellipsoid_m, sound_speeds_m_s, initial_latlon, initial_clock, ... |
| velocity/oyster-velocity-corrected-rerun-results.json | a8b646833fc7a0d36 | outputs/oyster-bay/velocity-comparison/results.json | 62463 | JSON object; top-level keys: plan, evaluation_truth, trials |
| velocity/oyster-velocity-corrected-rerun-plan.json | a8b646833fc7a0d36 | outputs/oyster-bay/velocity-comparison/plan.json | 5350 | JSON object; top-level keys: validation, protocol, traces, methods, heights_ellipsoid_m, sound_speeds_m_s, initial_latlon, initial_clock, ... |
| velocity/oyster-velocity-4cond-results.json | a8b646833fc7a0d36 | outputs/oyster-bay/velocity-comparison-4cond/results.json | 232935 | JSON object; top-level keys: plan, evaluation_truth, trials |
| velocity/oyster-velocity-4cond-plan.json | a8b646833fc7a0d36 | outputs/oyster-bay/velocity-comparison-4cond/plan.json | 5370 | JSON object; top-level keys: validation, protocol, traces, methods, heights_ellipsoid_m, sound_speeds_m_s, initial_latlon, initial_clock, ... |
| velocity/dev-bench-default-results.json | a8b646833fc7a0d36 | outputs/dev/bench-default/results.json | 1041187 | JSON object; top-level keys: validation, split, seeds, scenarios, methods, method_options, source_hashes, versions, ... |
| velocity/dev-bench-extended-results.json | a8b646833fc7a0d36 | outputs/dev/bench-extended/results.json | 217265 | JSON object; top-level keys: validation, split, seeds, scenarios, methods, method_options, source_hashes, versions, ... |
| velocity/dev-bench-extended2-results.json | a8b646833fc7a0d36 | outputs/dev/bench-extended2/results.json | 217128 | JSON object; top-level keys: validation, split, seeds, scenarios, methods, method_options, source_hashes, versions, ... |
| measurement/track-c-final-dev-results.json | a598905fc5ef4ddea | outputs/track-c/final-dev/results.json | 326184 | JSON object; top-level keys: validation, seeds, window_seconds, methods, summary_median_over_seeds, trials |
| measurement/oyster-measurement-results.json | a598905fc5ef4ddea | outputs/oyster-bay/measurement/results.json | 573141 | JSON object; top-level keys: plan, evaluation_truth, trials |
| discovery/dev-0-4-results.json | a545b5fcb9f227a45 | outputs/discovery/dev-0-4/results.json | 1473915 | JSON object; top-level keys: validation, seeds, scenarios, distractors, options, summary, by_scenario, false_pass_z, ... |
| discovery/oyster-discovery-results.json | a545b5fcb9f227a45 | outputs/oyster-bay/discovery/results.json | 132679 | JSON object; top-level keys: validation, options, model, initial_candidate, initial_error_m, first_discovery, first_confirmed, joint_result, ... |
| detection/bench_dev_5_12.json | a5ee96e7f92bbb2ad | outputs/autodetect/bench_dev_5_12.json | 123476 | JSON object; top-level keys: validation, generator, seeds, scenarios, detector, noise_only_policy, truth_cpa_source, summary, ... |
| detection/bench_dev_2_4.json | a5ee96e7f92bbb2ad | outputs/autodetect/bench_dev_2_4.json | 48019 | JSON object; top-level keys: validation, generator, seeds, scenarios, detector, noise_only_policy, truth_cpa_source, summary, ... |
| detection/real_candidates.json | a5ee96e7f92bbb2ad | outputs/autodetect/real_candidates.json | 11000 | JSON object; top-level keys: candidates, null_candidates |
| detection/oyster-autodetect-detection.json | a5ee96e7f92bbb2ad | outputs/oyster-bay/autodetect/detection.json | 7244 | JSON object; top-level keys: validation, options, runtime_s, matches, detections, candidates |
| detection/oyster-autodetect-detections.json | a5ee96e7f92bbb2ad | outputs/oyster-bay/autodetect/detections.json | 1792 | JSON list of 3 items |
| search-timing/scale_6_nojet.json | a1c61e797ed4dce5d | outputs/scale_6_nojet.json | 19735 | JSON object; top-level keys: summary, top |
| search-timing/scale_12_nojet.json | a1c61e797ed4dce5d | outputs/scale_12_nojet.json | 19768 | JSON object; top-level keys: summary, top |
| search-timing/profile_6_all.log | a1c61e797ed4dce5d | outputs/profile_6_all.log | 1857 | text; first line: 697254430 function calls (697254420 primitive calls) in 300.010 seconds |
| combined/oyster-combined-results.json | a1dfa7db2d5239f6a | outputs/oyster-bay/combined/results.json | 446191 | JSON object; top-level keys: plan, evaluation_truth, trials |
| combined/oyster-combined-plan.json | a1dfa7db2d5239f6a | outputs/oyster-bay/combined/plan.json | 12570 | JSON object; top-level keys: validation, protocol, traces, weather_file, atmospheres, layered_profiles, cells, initial_latlon, ... |
| combined/realistic-combined-results.json | a1dfa7db2d5239f6a | outputs/realistic-combined/results.json | 24340 | JSON object; top-level keys: validation, split, seeds, scenarios, methods, method_options, method_atmosphere, seconds, ... |
| uncertainty/calibration-summary.json | a0f80cbf86d462445 | outputs/uncertainty/calibration/summary.json | 54610 | JSON object; top-level keys: rayleigh_reference, methods, by_scenario |
| uncertainty/calibration-summary.txt | a0f80cbf86d462445 | outputs/uncertainty/calibration/summary.txt | 9699 | text; first line: Rayleigh(1) reference: d50 1.18, d90 2.15; in50 0.50, in90 0.90 |
| uncertainty/bootstrap.json | a0f80cbf86d462445 | outputs/uncertainty/bootstrap/bootstrap.json | 1274 | JSON list of 3 items |
| uncertainty/oyster-uncertainty-results.json | a0f80cbf86d462445 | outputs/oyster-bay/uncertainty/results.json | 543075 | JSON object; top-level keys: plan, evaluation_truth, trials, consistency |
| uncertainty/oyster-uncertainty-summary.txt | a0f80cbf86d462445 | outputs/oyster-bay/uncertainty/summary.txt | 1795 | text; first line: cell                                  err m  a90 m  b90 m  brg  in50  in90     d d_meas  v |
| three-flyovers/three-summary.txt | a9a575f65d0f28fd3 | outputs/oyster-bay/three-summary.txt | 27469 | text; first line: ERRORS (m) per cell |
| three-flyovers/three-events.json | a9a575f65d0f28fd3 | outputs/oyster-bay/three/events.json | 12565 | JSON object; top-level keys: detector_options, detector_seconds, detections, detector_candidates, detector_matches, discovery_options, discovery_candidate, discovery_passes, ... |
| three-flyovers/hum-benchmark-results.json | a9a575f65d0f28fd3 | outputs/hum-benchmark/results.json | 51543 | JSON object; top-level keys: validation, seeds, window_seconds, methods, summary_median_over_seeds, trials |
| three-flyovers/hum-benchmark-summary.txt | a9a575f65d0f28fd3 | outputs/hum-benchmark/summary.txt | 665 | text; first line: scene/method                     fail track rms  cpa rms median/e/  loc m   valid    cpa v |
| fast-search/evaluation.json | a4e89c5ce82ae7686 | outputs/oyster-bay/search/evaluation.json | 8425 | JSON object; top-level keys: truth, correct_flights, report |
| fast-search/compare.json | a4e89c5ce82ae7686 | outputs/oyster-bay/search/compare.json | 12365 | JSON object; top-level keys: 6km-capped, 6km-exhaustive, 12km-capped, 12km-exhaustive |
| fast-search/synthetic.json | a4e89c5ce82ae7686 | outputs/oyster-bay/search/synthetic.json | 35278 | JSON object; top-level keys: rows, summary, decoys |
| fast-search/rank-auto-full.json | a4e89c5ce82ae7686 | outputs/oyster-bay/search/rank-auto-full.json | 33323 | JSON object; top-level keys: detections, rule, rows |
| fast-search/refine-full.json | a4e89c5ce82ae7686 | outputs/oyster-bay/search/refine-full.json | 12233 | JSON object; top-level keys: fits, ranked, peak_rss_gb |
| fast-search/detections.json | a4e89c5ce82ae7686 | outputs/oyster-bay/search/detections.json | 1792 | JSON list of 3 items |
| fast-search/detect-run.json | a4e89c5ce82ae7686 | outputs/oyster-bay/search/detect-run.json | 76 | JSON object; top-level keys: seconds, peak_rss_gb, count |
| eagle-cam-heldout/plan.json | ac61430d850dcd370 | outputs/eagle-cam/plan.json | 16170 | JSON object; top-level keys: protocol, cells, discovery_options, detector_options, observer_height_m, deviations, notes, source_hashes, ... |
| eagle-cam-heldout/preparation.json | ac61430d850dcd370 | outputs/eagle-cam/preparation.json | 20403 | JSON object; top-level keys: start, published_event_pairs, published_event_indices, published_event_records, published_events_section, published_windows_inside_section, recording_epoch_implied_utc, initial_clock_published, ... |
| eagle-cam-heldout/results.json | ac61430d850dcd370 | outputs/eagle-cam/results.json | 28181 | JSON object; top-level keys: evaluation_truth, start_candidate_error_m, table, contrast_F1_F2, factorial, published |
| eagle-cam-heldout/fits.json | ac61430d850dcd370 | outputs/eagle-cam/fits.json | 779551 | JSON object; top-level keys: plan_written_utc, status, trials, discoveries, completed_utc |
| final-heldout-synthetic/plan.json | aecb4c558fb79ad31 | outputs/final-heldout/plan.json | 16126 | JSON object; top-level keys: purpose, git_commit, git_dirty, seeds, detection_seeds, workers, A, B, ... |
| final-heldout-synthetic/summary.txt | aecb4c558fb79ad31 | outputs/final-heldout/summary.txt | 14086 | text; first line: part A: 217 s, peak RSS so far 154 MB |
| final-heldout-synthetic/summary.json | aecb4c558fb79ad31 | outputs/final-heldout/summary.json | 122219 | JSON object; top-level keys: _meta, A, B, C, F, D, E |

Not copied: a9a575f65d0f28fd3 outputs/oyster-bay/three/results.json (1526264 bytes, over 1.5 MB limit).
