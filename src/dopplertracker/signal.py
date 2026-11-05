"""Memory-bounded audio decoding and low-frequency spectral analysis."""

import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import map_coordinates


@dataclass
class Spectrogram:
    times: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray

    def __post_init__(self):
        self.times, self.frequencies, self.power = map(np.asarray, (self.times, self.frequencies, self.power))
        if (
            len(self.times) < 2
            or len(self.frequencies) < 2
            or self.power.shape != (len(self.frequencies), len(self.times))
            or not np.isfinite(self.power).all()
            or not np.isfinite(self.times).all()
            or not np.isfinite(self.frequencies).all()
            or np.any(np.diff(self.times) <= 0)
            or np.any(np.diff(self.frequencies) <= 0)
        ):
            raise ValueError("spectrogram needs finite power and increasing time/frequency axes")

    def whiten(self):
        # Robust per-frequency whitening suppresses stationary hum; no hard gate on faint ridges.
        # Audio amplitudes are arbitrary. log1p(raw power) plus an absolute floor
        # suppressed quiet recordings by many orders of magnitude. A relative
        # log-power floor makes gain changes cancel when subtracting the center.
        power = np.maximum(np.asarray(self.power, dtype=np.float64), 0)
        positive = power[power > 0]
        if not len(positive):
            return Spectrogram(self.times, self.frequencies, np.zeros_like(self.power, dtype=np.float32))
        minimum = max(float(np.median(positive)) * 1e-12, np.finfo(np.float64).tiny)
        logged = np.log(np.maximum(power, minimum))
        center = np.median(logged, axis=1, keepdims=True)
        spread = 1.4826 * np.median(np.abs(logged - center), axis=1, keepdims=True)
        floor = max(float(np.median(spread)) * 0.2, 1e-4)
        whitened = np.clip((logged - center) / np.maximum(spread, floor), 0, 20)
        return Spectrogram(self.times, self.frequencies, whitened.astype(np.float32))

    def sample(self, times, frequencies):
        times, frequencies = np.broadcast_arrays(times, frequencies)
        valid = (
            np.isfinite(frequencies)
            & (frequencies >= self.frequencies[0])
            & (frequencies <= self.frequencies[-1])
            & np.isfinite(times)
            & (times >= self.times[0])
            & (times <= self.times[-1])
        )
        ti = np.interp(np.nan_to_num(times), self.times, np.arange(len(self.times)))
        fi = np.interp(np.nan_to_num(frequencies), self.frequencies, np.arange(len(self.frequencies)))
        values = map_coordinates(self.power, np.array([fi.ravel(), ti.ravel()]), order=1, prefilter=False)
        return np.where(valid, values.reshape(times.shape), 0.0)

    def save(self, path):
        np.savez_compressed(path, times=self.times, frequencies=self.frequencies, power=self.power)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z["times"], z["frequencies"], z["power"])


def from_samples(samples, sample_rate=2048, window_seconds=4.0, hop_seconds=0.5, band=(40, 450)):
    n = int(round(sample_rate * window_seconds))
    hop = int(round(sample_rate * hop_seconds))
    if n < 4 or hop < 1 or len(samples) < n or not 0 <= band[0] < band[1] < sample_rate / 2:
        raise ValueError("invalid STFT settings, frequency band, or recording shorter than one window")
    frequencies = rfftfreq(n, 1 / sample_rate)
    keep = (frequencies >= band[0]) & (frequencies <= band[1])
    starts = np.arange(0, len(samples) - n + 1, hop)
    power = np.empty((keep.sum(), len(starts)), dtype=np.float32)
    window = np.hanning(n)
    for begin in range(0, len(starts), 256):
        indices = starts[begin : begin + 256, None] + np.arange(n)
        spectrum = rfft(np.asarray(samples[indices]) * window, axis=1)
        # Scale a sinusoid to O(1), independent of FFT length.
        power[:, begin : begin + len(indices)] = (np.abs(spectrum[:, keep]) * 2 / window.sum()).T ** 2
    return Spectrogram((starts + n / 2) / sample_rate, frequencies[keep], power)


def decode_samples(path, sample_rate=2048):
    """Mono float32 samples at ``sample_rate``; sample k is at audio time k / sample_rate.

    The same ffmpeg conversion as ``decode``, so the samples align with its spectrogram times.
    One hour at 2048 Hz is about 30 MB.
    """
    with tempfile.TemporaryDirectory(prefix="doppler-audio-") as directory:
        pcm = directory + "/audio.f32"
        _ffmpeg_pcm(path, pcm, sample_rate)
        return np.fromfile(pcm, dtype="<f4").astype(np.float32)


def _ffmpeg_pcm(path, pcm, sample_rate):
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-y",
        pcm,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise ValueError("ffmpeg is required to decode audio/video") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(exc.stderr.decode(errors="replace")[:1000]) from exc


def decode(path, sample_rate=2048, window_seconds=4.0, hop_seconds=0.5, band=(40, 450)):
    with tempfile.TemporaryDirectory(prefix="doppler-audio-") as directory:
        pcm = directory + "/audio.f32"
        _ffmpeg_pcm(path, pcm, sample_rate)
        samples = np.memmap(pcm, dtype="<f4", mode="r")
        try:
            return from_samples(samples, sample_rate, window_seconds, hop_seconds, band)
        finally:
            del samples
