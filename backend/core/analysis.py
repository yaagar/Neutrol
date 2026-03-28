"""
analysis.py — Read-only signal analysis: FFT, PSD, band power, spectrogram.

These functions never modify the signal — they only compute data for visualization.
Signal processing uses SciPy and NumPy. MNE is used only in data_loader.py for I/O.

Called by analysis_routes.py in response to frontend requests from PreprocessPage.

Design principle: all functions are pure — same input always produces same output.
They can be tested independently without a running server.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import stft, welch

logger = logging.getLogger(__name__)

# EEG frequency band definitions (standard clinical/BCI conventions).
# Bands are listed in ascending frequency order for chart display.
EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5,  4.0),   # Deep sleep, unconsciousness
    "theta": (4.0,  8.0),   # Drowsiness, meditation, memory
    "alpha": (8.0,  13.0),  # Relaxed wakefulness, eyes closed — most visible rhythm
    "beta":  (13.0, 30.0),  # Active thinking, focus, motor activity
    "gamma": (30.0, 100.0), # High-level cognition, sensory processing
}


def compute_fft(
    signals: np.ndarray,
    sample_rate: int,
    channel_names: list[str] | None = None,
) -> dict:
    """
    Compute the FFT magnitude spectrum for each channel.

    FFT converts a time-domain signal into the frequency domain, showing which
    frequencies are present and how strong they are.

    Uses numpy.fft.rfft (real FFT) because EEG is real-valued — only positive
    frequencies matter. This halves the computation compared to full FFT.

    Args:
        signals:       Shape (n_channels, n_samples).
        sample_rate:   Sampling frequency in Hz. Determines the frequency axis.
        channel_names: Names for each channel row. Defaults to "ch_0", "ch_1", ...

    Returns:
        {
          "frequencies": [0.0, 0.5, 1.0, ...],           # in Hz
          "magnitudes":  {"Fp1": [...], "Fp2": [...]}     # one list per channel
        }
    """
    n_channels, n_samples = signals.shape
    names = channel_names or [f"ch_{i}" for i in range(n_channels)]

    # rfftfreq gives the positive frequency bins in Hz.
    # d = 1/sample_rate is the time step between samples.
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

    magnitudes: dict[str, list[float]] = {}
    for i, name in enumerate(names):
        # rfft of a real signal has N//2+1 complex output values.
        # abs() gives the magnitude of each frequency component.
        fft_vals = np.abs(np.fft.rfft(signals[i])).astype(float)
        magnitudes[name] = fft_vals.tolist()

    logger.debug("FFT computed: %d channels, %d freq bins", n_channels, len(freqs))

    return {
        "frequencies": freqs.tolist(),
        "magnitudes":  magnitudes,
    }


def compute_psd(
    signals: np.ndarray,
    sample_rate: int,
    nperseg: int = 256,
    channel_names: list[str] | None = None,
) -> dict:
    """
    Compute the Power Spectral Density using Welch's method.

    Why Welch over raw FFT?
      Welch divides the signal into overlapping segments, computes FFT on each,
      and averages them. This dramatically reduces noise in the estimate.
      Raw FFT on a long signal is noisy. Welch is smoother and much more reliable
      for EEG, especially for detecting band power.

    Args:
        signals:       Shape (n_channels, n_samples).
        sample_rate:   Sampling frequency in Hz.
        nperseg:       Samples per Welch segment. Longer = finer frequency resolution
                       but noisier. 256 samples at 256 Hz = 1-second segments.
        channel_names: Names for each channel row.

    Returns:
        {
          "frequencies": [...],                           # in Hz
          "power":       {"Fp1": [...], "Fp2": [...]}     # PSD values (V²/Hz)
        }
    """
    n_channels, n_samples = signals.shape
    names = channel_names or [f"ch_{i}" for i in range(n_channels)]

    # Cap nperseg at n_samples to avoid scipy errors on short signals.
    nperseg_actual = min(nperseg, n_samples)

    power: dict[str, list[float]] = {}
    freq_axis: np.ndarray | None = None

    for i, name in enumerate(names):
        # welch() returns (frequencies, power_spectrum_density)
        f, pxx = welch(signals[i], fs=sample_rate, nperseg=nperseg_actual)
        if freq_axis is None:
            freq_axis = f   # all channels share the same frequency axis
        power[name] = pxx.astype(float).tolist()

    logger.debug("PSD computed: %d channels, nperseg=%d", n_channels, nperseg_actual)

    return {
        "frequencies": (freq_axis.tolist() if freq_axis is not None else []),
        "power":       power,
    }


def compute_band_power(
    signals: np.ndarray,
    sample_rate: int,
    channel_names: list[str] | None = None,
) -> dict:
    """
    Compute power within each standard EEG frequency band, per channel.

    Uses Welch PSD internally (for noise reduction), then integrates (area under
    the PSD curve) within each band's frequency range using trapezoid integration.

    This gives a single scalar per band per channel — easy to compare across channels
    and sessions. Alpha power increasing when eyes close is the classic example.

    Args:
        signals:       Shape (n_channels, n_samples).
        sample_rate:   Sampling frequency in Hz.
        channel_names: Names for each channel row.

    Returns:
        {
          "channels": ["Fp1", "Fp2", ...],
          "bands":    {
            "delta": [ch0_power, ch1_power, ...],
            "theta": [...],
            "alpha": [...],
            "beta":  [...],
            "gamma": [...]
          }
        }
    """
    n_channels, n_samples = signals.shape
    names = channel_names or [f"ch_{i}" for i in range(n_channels)]

    # Cap nperseg to avoid scipy errors on short signals.
    # Minimum 4 samples required for any Welch estimate.
    nperseg = min(256, max(4, n_samples))

    # Initialise result structure — one list per band, one value per channel
    bands: dict[str, list[float]] = {band: [] for band in EEG_BANDS}

    for i in range(n_channels):
        f, pxx = welch(signals[i], fs=sample_rate, nperseg=nperseg)

        for band_name, (low, high) in EEG_BANDS.items():
            # Select the frequency bins that fall within this band's range.
            idx = np.logical_and(f >= low, f <= high)
            if idx.sum() > 0:
                # Integrate PSD over the band using the trapezoid rule.
                # This gives total power in Watts (or V² if signals are in Volts).
                band_power = float(trapezoid(pxx[idx], f[idx]))
            else:
                # Band is above Nyquist or below resolution — return zero.
                band_power = 0.0
            bands[band_name].append(round(band_power, 8))

    logger.debug("Band power computed: %d channels, %d bands", n_channels, len(EEG_BANDS))

    return {
        "channels": names,
        "bands":    bands,
    }


def compute_spectrogram(
    signals: np.ndarray,
    sample_rate: int,
    channel_index: int = 0,
) -> dict:
    """
    Compute a time-frequency spectrogram for one channel using STFT.

    The spectrogram shows how frequency content changes over time — you can see
    alpha waves appear when eyes close, disappear when eyes open. This is the
    most important analysis tool for understanding non-stationary EEG.

    Uses scipy.signal.stft (Short-Time Fourier Transform): many short FFTs at
    successive time points, stacked into a 2D power matrix.

    Args:
        signals:       Shape (n_channels, n_samples).
        sample_rate:   Sampling frequency in Hz.
        channel_index: Which channel to compute the spectrogram for (0-indexed).

    Returns:
        {
          "times":      [t0, t1, ...],         # time axis in seconds
          "frequencies": [0.0, 0.5, ...],      # frequency axis in Hz
          "power_db":   [[f0t0, f0t1, ...],    # 2D list: power_db[freq][time]
                         [f1t0, f1t1, ...], ...]
        }

    Raises:
        ValueError: If channel_index is out of range.
    """
    n_channels, n_samples = signals.shape

    if channel_index < 0 or channel_index >= n_channels:
        raise ValueError(
            f"channel_index {channel_index} is out of range "
            f"(signal has {n_channels} channels, indices 0–{n_channels - 1})."
        )

    channel = signals[channel_index]

    # nperseg controls the frequency resolution vs time resolution trade-off.
    # Longer segments → finer frequency resolution, coarser time resolution.
    # We use n_samples // 8 with a floor of 64 as a reasonable default.
    # Cap at 256 to avoid very long segments on long recordings.
    nperseg = min(256, max(64, n_samples // 8))
    # Ensure nperseg doesn't exceed signal length
    nperseg = min(nperseg, n_samples)

    # stft returns:
    #   f: frequency axis (n_freqs,)
    #   t: time axis (n_times,)
    #   Zxx: complex STFT coefficients (n_freqs, n_times)
    f, t, Zxx = stft(channel, fs=sample_rate, nperseg=nperseg)

    # Convert complex coefficients to power in decibels.
    # power = |STFT|² gives instantaneous power at each time-frequency bin.
    # log10 converts to dB scale — better for visualization since EEG power
    # spans many orders of magnitude.
    power = np.abs(Zxx) ** 2
    # Add tiny epsilon before log to avoid log(0) = -inf at silent bins
    power_db = 10.0 * np.log10(power + 1e-12)

    logger.debug(
        "Spectrogram computed: channel %d, nperseg=%d, shape=(%d freqs, %d times)",
        channel_index, nperseg, len(f), len(t),
    )

    return {
        "times":       t.tolist(),
        "frequencies": f.tolist(),
        "power_db":    power_db.tolist(),   # 2D list [freq_idx][time_idx]
    }


def compute_signal_stats(
    signals: np.ndarray,
    sample_rate: int,
    channel_names: list[str] | None = None,
) -> dict:
    """
    Compute per-channel signal quality metrics for the Analyze button.

    These metrics are standard in EEG quality assessment pipelines:
      - rms: Root mean square — overall signal power; very low = disconnected electrode
      - peak_to_peak: max - min amplitude — high values indicate artifacts
      - std: Standard deviation — signal variability around the mean
      - kurtosis: 4th statistical moment (Pearson). Normal Gaussian = 3.0.
                  > 5 strongly suggests spike artifacts (blinks, muscle bursts).
                  < 2 suggests a flat/saturated channel.
      - max_gradient: Maximum sample-to-sample change — detects electrode pop-offs
                      (sudden large jumps in voltage).
      - zero_crossing_rate: Sign changes per second — rough frequency proxy.
                            High ZCR → high-frequency content.

    Args:
        signals:       Shape (n_channels, n_samples).
        sample_rate:   Sampling frequency in Hz.
        channel_names: Names for each channel row.

    Returns:
        {"channels": [{"channel": name, "rms": ..., "peak_to_peak": ...,
                        "std": ..., "kurtosis": ..., "max_gradient": ...,
                        "zero_crossing_rate": ..., "artifact_warning": bool}, ...]}
    """
    from scipy.stats import kurtosis as scipy_kurtosis

    n_channels, n_samples = signals.shape
    names = channel_names or [f"ch_{i}" for i in range(n_channels)]

    results: list[dict] = []
    for i, name in enumerate(names):
        ch = signals[i].astype(np.float64)

        # Root mean square: overall signal power.
        rms = float(np.sqrt(np.mean(ch ** 2)))

        # Peak-to-peak: amplitude range. Large values indicate artifact spikes.
        peak_to_peak = float(np.max(ch) - np.min(ch))

        # Standard deviation: variability around the mean.
        std = float(np.std(ch))

        # Pearson kurtosis (fisher=False so normal distribution = 3.0).
        # kurtosis > 5 is common artifact threshold in EEG preprocessing.
        kurt = float(scipy_kurtosis(ch, fisher=False))

        # Max gradient: largest sample-to-sample change in absolute value.
        # Detects sudden jumps from electrode pop-off or clipping.
        diff = np.diff(ch)
        max_gradient = float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0

        # Zero crossing rate: sign changes per second.
        # Count where consecutive samples change sign (positive↔negative).
        signs = np.sign(ch)
        crossings = int(np.sum(np.diff(signs) != 0))
        zcr = crossings / (n_samples / sample_rate)

        # Flag likely artifact channels: high kurtosis OR very large amplitude.
        # Thresholds are conservative — flag for review, not hard rejection.
        artifact_warning = (kurt > 5.0) or (peak_to_peak > 200.0)

        results.append({
            "channel":            name,
            "rms":                round(rms, 4),
            "peak_to_peak":       round(peak_to_peak, 4),
            "std":                round(std, 4),
            "kurtosis":           round(kurt, 4),
            "max_gradient":       round(max_gradient, 4),
            "zero_crossing_rate": round(zcr, 2),
            "artifact_warning":   artifact_warning,
        })

    logger.debug("Signal stats computed: %d channels", n_channels)
    return {"channels": results}
