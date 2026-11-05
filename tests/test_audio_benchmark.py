import json

import numpy as np

from dopplertracker.audio_benchmark import SAMPLE_RATE, audio_scene, run_audio_benchmark
from dopplertracker.signal import from_samples


def test_waveform_scene_is_a_physical_chirp_with_harmonics():
    scene = audio_scene(7, "clean_audio", duration=120.0)
    frequency = scene["observed_frequency_hz"][0]
    assert len(scene["waveform"]) == int(120 * SAMPLE_RATE)
    assert np.ptp(frequency) > 10.0
    spectrum = from_samples(scene["waveform"], SAMPLE_RATE, window_seconds=2.0, band=(40, 350))
    assert spectrum.power.max() > 0.01


def test_stft_bin_width_tracks_window_length():
    scene = audio_scene(3, duration=120.0)
    short = from_samples(scene["waveform"], SAMPLE_RATE, window_seconds=1.0, band=(40, 350))
    long = from_samples(scene["waveform"], SAMPLE_RATE, window_seconds=4.0, band=(40, 350))
    assert np.diff(short.frequencies)[0] == 1.0
    assert np.diff(long.frequencies)[0] == 0.25


def test_audio_benchmark_smoke_keeps_all_attempted_trials(tmp_path):
    report = run_audio_benchmark(seeds=(100,), windows=(2.0,), output_dir=tmp_path)
    # two scenarios x constant/linear projected fits; exceptions still count.
    assert report["attempted_trials"] == 4
    assert len(report["trials"]) == report["attempted_trials"]
    assert report["failed_trials"] == sum(not row["success"] for row in report["trials"])
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["validation"] == "synthetic waveform; not real recording validation"
