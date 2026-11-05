import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile

from dopplertracker.benchmark import analytic_pass


def test_cli_help_lists_pipeline():
    result = subprocess.run(
        [sys.executable, "-m", "dopplertracker", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    for command in (
        "spectrogram",
        "detect",
        "fit",
        "fetch-day",
        "extract-region",
        "search",
        "search-single",
        "refine",
        "benchmark",
        "stress-benchmark",
        "controls-benchmark",
        "validate",
    ):
        assert command in result.stdout


@pytest.mark.parametrize(
    "flag", [["--smooth", "spline"], ["--fit-wind"], ["--source-degree", "2"], ["--top", "0"]]
)
def test_refinement_rejects_ignored_or_invalid_options_before_loading_data(flag, capsys, tmp_path):
    from dopplertracker.cli import main

    code = main(
        [
            "refine",
            "missing.npz",
            "missing.json",
            "--events",
            "missing-events.json",
            "--traces",
            "missing-traces.npz",
            "--elevation",
            "0",
            "--method",
            "brightness",
            "--out",
            str(tmp_path / "result.json"),
            *flag,
        ]
    )
    assert code == 2
    error = capsys.readouterr().err
    assert "No such file" not in error
    assert "require" in error or "positive" in error


def test_cli_demo_creates_reusable_inputs(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "dopplertracker", "demo", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "spectrogram.npz").exists()
    assert (tmp_path / "traces.npz").exists()
    assert (tmp_path / "events.json").exists()


def test_bad_command_fails_with_helpful_error():
    result = subprocess.run(
        [sys.executable, "-m", "dopplertracker", "search-single"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "required" in result.stderr


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_audio_decode_to_fitted_flyover(tmp_path):
    sample_rate = 2048
    times = np.arange(140 * sample_rate) / sample_rate
    frequency, _, cpa, _ = analytic_pass(times, [80, 0, 0], [0, 1500, 1000], 60, np.zeros(3), 110, 343)
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    rng = np.random.default_rng(9)
    audio = 0.3 * np.sin(phase) + 0.1 * np.sin(2 * phase) + rng.normal(0, 0.02, len(times))
    wavfile.write(tmp_path / "recording.wav", sample_rate, audio.astype(np.float32))
    decode = subprocess.run(
        [
            sys.executable,
            "-m",
            "dopplertracker",
            "spectrogram",
            str(tmp_path / "recording.wav"),
            str(tmp_path / "audio.npz"),
        ],
        capture_output=True,
        text=True,
    )
    assert decode.returncode == 0, decode.stderr
    fit = subprocess.run(
        [
            sys.executable,
            "-m",
            "dopplertracker",
            "fit",
            str(tmp_path / "audio.npz"),
            "--window",
            "10",
            "125",
            "--band",
            "95",
            "125",
            "--harmonics",
            "2",
            "--out",
            str(tmp_path / "event.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert fit.returncode == 0, fit.stderr
    event = json.loads((tmp_path / "event.json").read_text())[0]
    assert abs(event["cpa"] - cpa) < 3
    assert abs(event["f0"] - 110) < 3
