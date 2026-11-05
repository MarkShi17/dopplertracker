"""Chirp-compensated Doppler measurement from raw audio by complex demodulation.

A 4 s STFT smears a tone that sweeps several Hz per second near closest
approach.  Instead, mix harmonic h of the audio with exp(-i 2 pi h * integral of
the model frequency).  If the model is right the product is a stationary tone at
0 Hz; a small model error leaves a slowly varying residual delta(t).  After a
low-pass filter and decimation, delta is measured with short Hann-windowed FFTs
and log-parabolic sub-bin interpolation.  The observation is
f_model(t) + delta(t) / h in fundamental Hz.

Its uncertainty follows the Cramér-Rao bound for one tone in white noise.  With
a Hann window, rho = (peak power / mean noise power per bin) - 1 gives
sigma = sqrt(8 / rho) / (2 pi T) Hz for analysis length T.  Stationary hum lines,
which are moving chirps after demodulation, are masked so that the estimate
cannot lock onto them.  Alternatively ``remove_lines`` subtracts them from the
audio first, so that frames where the tone crosses a line stay usable.
"""

from dataclasses import dataclass

import numpy as np
from scipy.fft import fft, fftfreq, next_fast_len, rfft, rfftfreq
from scipy.ndimage import median_filter
from scipy.signal import filtfilt, iirnotch, resample_poly

BASEBAND_RATE = 128.0
# A noise-only bin exceeds 9x its mean power with probability exp(-9) ~ 1e-4, so
# the largest of the ~100 independent bins searched is a false peak ~1% of the time.
DETECTION_SNR = 8.0


@dataclass(frozen=True)
class Audio:
    """Mono samples; sample k is at audio time ``start + k / sample_rate`` (the spectrogram clock)."""

    samples: np.ndarray
    sample_rate: float = 2048.0
    start: float = 0.0

    def __post_init__(self):
        if np.ndim(self.samples) != 1 or not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("audio needs one-dimensional samples and a positive sample rate")


def stationary_lines(audio, span, band=(20.0, 1000.0), segment_seconds=8.0, threshold=4.0):
    """Frequencies of persistent narrow lines (hum) around ``span`` (audio seconds).

    The median periodogram over consecutive segments keeps lines present for most
    of the span and rejects an aircraft tone, which occupies any bin only briefly.
    """
    fs = audio.sample_rate
    n = int(round(segment_seconds * fs))
    first = max(0, int((span[0] - audio.start) * fs))
    last = min(len(audio.samples), int((span[1] - audio.start) * fs))
    starts = np.arange(first, last - n + 1, n)
    if len(starts) < 3:
        return np.empty(0)
    window = np.hanning(n)
    power = np.median(
        [np.abs(rfft(np.asarray(audio.samples[s : s + n], dtype=float) * window)) ** 2 for s in starts],
        axis=0,
    )
    frequencies = rfftfreq(n, 1 / fs)
    baseline = median_filter(power, size=int(4.0 * segment_seconds) | 1, mode="nearest")
    peak = (
        (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:]) & (power[1:-1] > threshold * baseline[1:-1])
    )
    index = np.flatnonzero(peak) + 1
    keep = (frequencies[index] >= band[0]) & (frequencies[index] <= band[1])
    return frequencies[index[keep]]


def _segments(samples, length, hop):
    starts = np.arange(0, len(samples) - length + 1, hop)
    return starts, samples[starts[:, None] + np.arange(length)]


def _line_amplitudes(segments, starts, frequencies, fs, ridge=1e-3):
    """Joint weighted least-squares complex amplitude of every line in every segment.

    The model is x(t) = sum_k a_k exp(i w_k t) + c.c. with t the audio sample
    time; negative-frequency images are ignored (lines lie well above 0 Hz).
    Returns a (segments, lines) array of a_k on that global time base.
    """
    length = segments.shape[1]
    taper = np.hanning(length + 2)[1:-1]
    local = np.arange(length) / fs
    omega = 2 * np.pi * np.asarray(frequencies, dtype=float)
    exponentials = np.exp(1j * local[:, None] * omega[None, :])
    basis = taper[:, None] * np.conj(exponentials)
    projection = segments @ basis
    # On segment-local time the Gram matrix is the same for every segment; the
    # ridge keeps nearly coincident lines (one line found in two bins) solvable.
    gram = basis.T @ exponentials
    gram = gram + ridge * np.mean(np.real(np.diag(gram))) * np.eye(len(omega))
    amplitude = np.linalg.solve(gram, projection.T).T
    return amplitude * np.exp(-1j * omega[None, :] * (starts[:, None] / fs))


def remove_lines(
    audio,
    lines,
    span,
    method="subtract",
    segment_seconds=4.0,
    median_segments=11,
    notch_width=0.3,
    margin=30.0,
):
    """Copy of ``audio`` over ``span`` (audio seconds, plus ``margin``) without stationary lines.

    ``subtract`` estimates each line's slowly varying complex amplitude by joint
    least squares on Hann segments (half-overlapping, ``segment_seconds`` long),
    refines its frequency from the median phase advance between segments, takes
    a running median over ``median_segments`` segments (so that a tone crossing
    the line for a few segments does not enter the estimate) and subtracts the
    interpolated line.  ``notch`` applies a zero-phase notch ``notch_width`` Hz
    wide at each line.  The returned ``Audio`` keeps the original clock.
    """
    if method not in ("subtract", "notch"):
        raise ValueError("line removal method must be subtract or notch")
    fs = audio.sample_rate
    first = max(0, int(np.floor((span[0] - margin - audio.start) * fs)))
    last = min(len(audio.samples), int(np.ceil((span[1] + margin - audio.start) * fs)))
    samples = np.asarray(audio.samples[first:last], dtype=float)
    lines = np.asarray(lines, dtype=float)
    lines = lines[(lines > 0) & (lines < 0.5 * fs - 1.0)]
    start = audio.start + first / fs
    if not len(lines) or not len(samples):
        return Audio(samples, fs, start)
    if method == "notch":
        cleaned = samples.copy()
        for frequency in lines:
            b, a = iirnotch(frequency, frequency / notch_width, fs)
            cleaned = filtfilt(b, a, cleaned)
        return Audio(cleaned, fs, start)
    length = int(round(segment_seconds * fs))
    hop = max(1, length // 2)
    if len(samples) < length + 2 * hop:
        return Audio(samples, fs, start)
    starts, segments = _segments(samples, length, hop)
    frequencies = lines.copy()
    for _ in range(2):
        amplitude = _line_amplitudes(segments, starts, frequencies, fs)
        # A frequency error df rotates the amplitude by 2 pi df hop / fs per segment.
        advance = np.angle(amplitude[1:] * np.conj(amplitude[:-1]))
        frequencies = frequencies + np.median(advance, axis=0) * fs / (2 * np.pi * hop)
    amplitude = _line_amplitudes(segments, starts, frequencies, fs)
    size = (min(int(median_segments), len(starts)) // 2) * 2 + 1
    smooth = median_filter(amplitude.real, size=(size, 1), mode="nearest") + 1j * median_filter(
        amplitude.imag, size=(size, 1), mode="nearest"
    )
    centres = (starts + 0.5 * (length - 1)) / fs
    times = np.arange(len(samples)) / fs
    cleaned = samples.copy()
    for k, frequency in enumerate(frequencies):
        value = np.interp(times, centres, smooth[:, k].real) + 1j * np.interp(
            times, centres, smooth[:, k].imag
        )
        cleaned -= 2 * np.real(value * np.exp(2j * np.pi * frequency * times))
    return Audio(cleaned, fs, start)


def _fill(values):
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return None
    grid = np.arange(len(values))
    return np.interp(grid, grid[finite], values[finite])


def demodulate(
    audio,
    times,
    guide,
    harmonic,
    half_band,
    window_seconds=2.0,
    lines=(),
    rate=BASEBAND_RATE,
    threshold=DETECTION_SNR,
):
    """Measure harmonic ``harmonic`` of the tone around ``guide`` at frame centres ``times``.

    ``guide(t)`` returns the model fundamental frequency in Hz (NaN where undefined).
    ``half_band`` is the search half-width in harmonic Hz and must cover the model
    error.  Returns fundamental-equivalent frequency and uncertainty (Hz), validity
    and baseband power SNR per frame.
    """
    times = np.asarray(times, dtype=float)
    fs = audio.sample_rate
    decimation = max(1, int(round(fs / rate)))
    rate = fs / decimation
    count = len(times)
    invalid = (np.full(count, np.nan), np.zeros(count, dtype=bool), np.full(count, np.nan), np.zeros(count))
    half_band = min(float(half_band), 0.4 * rate)
    margin = window_seconds / 2 + 1.0
    first = max(0, int(np.floor((times[0] - margin - audio.start) * fs)))
    last = min(len(audio.samples), int(np.ceil((times[-1] + margin - audio.start) * fs)))
    if last - first < decimation * 8:
        return invalid
    grid = np.arange(times[0] - margin, times[-1] + margin + 0.1, 0.1)
    model = np.asarray(guide(grid), dtype=float)
    filled = _fill(model)
    if filled is None:
        return invalid
    sample_times = audio.start + np.arange(first, last) / fs
    reference = harmonic * np.interp(sample_times, grid, filled)
    phase = 2 * np.pi * np.cumsum(reference) / fs
    mixed = np.asarray(audio.samples[first:last], dtype=float) * np.exp(-1j * phase)
    baseband = resample_poly(mixed, 1, decimation)
    base_times = sample_times[0] + np.arange(len(baseband)) * decimation / fs

    half = int(round(window_seconds * rate / 2))
    length = 2 * half + 1
    duration = length / rate
    centres = np.rint((times - base_times[0]) * rate).astype(int)
    inside = (centres - half >= 0) & (centres + half < len(baseband))
    centres = np.clip(centres, half, len(baseband) - half - 1)
    taper = np.hanning(length + 2)[1:-1]
    segments = baseband[centres[:, None] + np.arange(-half, half + 1)] * taper
    nfft = next_fast_len(16 * length)
    power = np.abs(fft(segments, nfft, axis=1)) ** 2
    offsets = fftfreq(nfft, 1 / rate)
    order = np.argsort(offsets)
    offsets, power = offsets[order], power[:, order]
    step = offsets[1] - offsets[0]

    # The model at each frame and its slope set where a stationary line falls in
    # baseband, and how far it spreads across one analysis window.
    centre_model = np.interp(times, grid, filled)
    slope = np.gradient(filled, grid)
    centre_slope = harmonic * np.abs(np.interp(times, grid, slope))
    lines = np.asarray(lines, dtype=float)
    blocked = np.zeros(power.shape, dtype=bool)
    # Hann main lobe half-width is 2/T; a chirping line also spreads by slope * T.
    guard = 2.5 / duration + 0.5 * centre_slope * duration
    positions = lines[None, :] - harmonic * centre_model[:, None]
    for column in positions.T:
        blocked |= np.abs(offsets[None, :] - column[:, None]) < guard[:, None]
    noise_zone = np.abs(offsets) <= min(0.45 * rate, half_band + 4.0)
    search = (np.abs(offsets) <= half_band)[None, :] & ~blocked
    candidates = np.where(search, power, -np.inf)
    index = np.argmax(candidates, axis=1)
    rows = np.arange(count)
    peak = power[rows, index]
    lower = np.log(np.maximum(power[rows, np.maximum(index - 1, 0)], 1e-300))
    upper = np.log(np.maximum(power[rows, np.minimum(index + 1, len(offsets) - 1)], 1e-300))
    middle = np.log(np.maximum(peak, 1e-300))
    curvature = lower - 2 * middle + upper
    shift = np.divide(0.5 * (lower - upper), curvature, out=np.zeros(count), where=curvature < -1e-12)
    delta = offsets[index] + np.clip(shift, -0.5, 0.5) * step

    exclude = (np.abs(offsets[None, :] - delta[:, None]) < 3.0 / duration) | blocked | ~noise_zone[None, :]
    noise = np.array([np.median(p[~e]) if (~e).sum() > 8 else np.nan for p, e in zip(power, exclude)])
    # The median of an exponential power distribution is ln 2 times its mean.
    noise = noise / np.log(2)
    snr = np.nan_to_num(peak / noise - 1.0, nan=0.0)
    sigma = np.sqrt(8.0 / np.maximum(snr, 1e-6)) / (2 * np.pi * duration)

    window_model = np.array(
        [np.isfinite(model[(grid >= t - duration / 2) & (grid <= t + duration / 2)]).all() for t in times]
    )
    near_line = np.any(np.abs(delta[:, None] - positions) < guard[:, None] + 1.0 / duration, axis=1)
    valid = (
        inside
        & window_model
        & np.isfinite(noise)
        & (snr >= threshold)
        & (np.abs(delta) < half_band - 2 * step)
        & ~near_line
    )
    frequency = centre_model + delta / harmonic
    return frequency, valid, sigma / harmonic, snr
