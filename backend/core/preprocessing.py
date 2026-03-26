"""
preprocessing.py — EEG preprocessing steps and the Pipeline class.

Design principles (from CLAUDE.md):
  - Each step is a pure class: same input always produces same output, no side effects.
  - All steps implement the PreprocessingStep interface: apply(), to_dict(), from_dict().
  - Steps can be serialized to/from the PipelineConfig JSON format, so a saved config
    exactly reproduces any previous result.
  - SlidingWindow is always the last step — it changes the return type from
    np.ndarray to list[np.ndarray].
  - EpochRejection is applied AFTER SlidingWindow — it filters out bad windows.

Calling convention for all steps:
  Input signals shape:  (n_channels, n_samples)
  Output signals shape: (n_channels, n_samples)   [except SlidingWindow, EpochRejection]

The optional channel_names parameter on apply() is used by ChannelSelector
to map channel names to array indices. Other steps ignore it.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt

from core.data_loader import DataSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PreprocessingStep — abstract base class
# ---------------------------------------------------------------------------

class PreprocessingStep:
    """
    Base class for all preprocessing steps.

    Every subclass must implement:
      apply()     — the actual signal transformation
      to_dict()   — serialize to the PipelineConfig JSON step format
      from_dict() — reconstruct from that same dict
    """

    def apply(
        self,
        signals: np.ndarray,
        sample_rate: int,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Apply this step to the signals.

        Args:
            signals:       Input array of shape (n_channels, n_samples).
            sample_rate:   Sampling frequency in Hz — needed by filter design functions.
            channel_names: Optional list of channel names, one per row of signals.
                           Used by ChannelSelector; ignored by all other steps.

        Returns:
            Transformed array, same shape (n_channels, n_samples).
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Serialize this step to the PipelineConfig JSON step format."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, config: dict) -> "PreprocessingStep":
        """Reconstruct a step from a PipelineConfig JSON step dict."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# BandpassFilter — keep only frequencies in [low_hz, high_hz]
# ---------------------------------------------------------------------------

class BandpassFilter(PreprocessingStep):
    """
    Zero-phase Butterworth bandpass filter.

    Why Butterworth order 4?
      - Butterworth has a maximally flat response in the passband — it doesn't distort
        signal amplitude within the frequency range we care about.
      - Order 4 is strong enough to reject noise outside the band (–80 dB/decade rolloff)
        without introducing ringing artifacts.

    Why sosfiltfilt instead of lfilter?
      - sosfiltfilt applies the filter forward then backward, which cancels out
        phase shift. Phase-correct filtering is critical for EEG: if events are
        shifted in time, they won't align across channels during analysis.
      - "sos" (second-order sections) format is numerically more stable than the
        traditional 'b, a' coefficient format for high-order filters.

    Typical EEG range: low_hz=1.0, high_hz=40.0
      - 1 Hz cutoff removes slow DC drift (electrode sweat potential)
      - 40 Hz cutoff removes EMG (muscle) artifacts above the gamma band
    """

    # Butterworth filter order — 4 is standard for EEG
    FILTER_ORDER = 4

    def __init__(self, low_hz: float, high_hz: float) -> None:
        """
        Args:
            low_hz:  Lower cutoff frequency in Hz (e.g. 1.0).
            high_hz: Upper cutoff frequency in Hz (e.g. 40.0).
        """
        if low_hz <= 0:
            raise ValueError(f"low_hz must be positive, got {low_hz}")
        if high_hz <= low_hz:
            raise ValueError(f"high_hz ({high_hz}) must be greater than low_hz ({low_hz})")
        self.low_hz = low_hz
        self.high_hz = high_hz

    def apply(
        self,
        signals: np.ndarray,
        sample_rate: int,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        """Apply the bandpass filter to every channel independently."""
        # Design the filter in second-order sections (sos) format.
        # We pass the cutoff frequencies as a fraction of the Nyquist frequency
        # by using fs=sample_rate — scipy handles the normalization internally.
        sos = butter(
            self.FILTER_ORDER,
            [self.low_hz, self.high_hz],
            btype="band",
            fs=sample_rate,
            output="sos",
        )

        # Apply the filter to each channel row.
        # sosfiltfilt operates along axis=-1 by default (last axis = time axis).
        # We iterate per channel to make the per-channel intent explicit.
        filtered = np.zeros_like(signals)
        for i, channel in enumerate(signals):
            filtered[i] = sosfiltfilt(sos, channel)

        return filtered

    def to_dict(self) -> dict:
        return {"step": "bandpass", "low_hz": self.low_hz, "high_hz": self.high_hz}

    @classmethod
    def from_dict(cls, config: dict) -> "BandpassFilter":
        return cls(low_hz=config["low_hz"], high_hz=config["high_hz"])


# ---------------------------------------------------------------------------
# NotchFilter — remove power-line interference at a specific frequency
# ---------------------------------------------------------------------------

class NotchFilter(PreprocessingStep):
    """
    IIR notch (band-stop) filter at a single frequency.

    Why a notch filter?
      Power line interference (60 Hz in North America, 50 Hz in Europe) creates
      a strong sinusoidal artifact in every channel. It must be removed before
      any frequency analysis, because a 60 Hz peak would dominate all spectra.

    Why iirnotch instead of a notch bandpass?
      iirnotch is designed specifically for this: it creates a very narrow stopband
      at exactly one frequency. The quality_factor (Q) controls the width:
      higher Q = narrower notch (removes fewer surrounding frequencies).
      Q=30 is standard for EEG — narrow enough to not disturb gamma-band signals.
    """

    DEFAULT_QUALITY_FACTOR = 30.0

    def __init__(self, freq_hz: float, quality_factor: float = DEFAULT_QUALITY_FACTOR) -> None:
        """
        Args:
            freq_hz:        Frequency to remove in Hz (e.g. 60.0 for North America).
            quality_factor: Controls notch width. Higher = narrower. Default 30.0.
        """
        if freq_hz <= 0:
            raise ValueError(f"freq_hz must be positive, got {freq_hz}")
        if quality_factor <= 0:
            raise ValueError(f"quality_factor must be positive, got {quality_factor}")
        self.freq_hz = freq_hz
        self.quality_factor = quality_factor

    def apply(
        self,
        signals: np.ndarray,
        sample_rate: int,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        """Apply the notch filter to every channel independently."""
        # iirnotch returns (b, a) coefficient arrays — the traditional IIR format.
        # We convert to sosfiltfilt-compatible format via sosfilt for stability,
        # but iirnotch is only order 2 so numerical issues are not a concern here.
        # We use sosfiltfilt by first building sos from b, a via tf2sos.
        from scipy.signal import sosfilt, tf2sos

        b, a = iirnotch(self.freq_hz, self.quality_factor, fs=sample_rate)

        # Convert to second-order sections for zero-phase filtering.
        # This keeps the same sosfiltfilt pattern as BandpassFilter for consistency.
        sos = tf2sos(b, a)

        filtered = np.zeros_like(signals)
        for i, channel in enumerate(signals):
            filtered[i] = sosfiltfilt(sos, channel)

        return filtered

    def to_dict(self) -> dict:
        return {
            "step": "notch",
            "freq_hz": self.freq_hz,
            "quality_factor": self.quality_factor,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "NotchFilter":
        return cls(
            freq_hz=config["freq_hz"],
            quality_factor=config.get("quality_factor", cls.DEFAULT_QUALITY_FACTOR),
        )


# ---------------------------------------------------------------------------
# Normalize — scale each channel to a standard range
# ---------------------------------------------------------------------------

class Normalize(PreprocessingStep):
    """
    Per-channel normalization.

    Why normalize?
      EEG amplitude varies enormously between subjects, sessions, and electrodes.
      Normalization makes channels comparable, which is important for CNNs that
      learn from relative amplitude patterns.

    Why per-channel (not across all channels at once)?
      Different electrodes have different baseline amplitudes — normalizing all
      channels together would let a noisy channel distort the scale of all others.

    Supported methods:
      'zscore' — subtract mean, divide by std. Output has mean≈0, std≈1.
                 Most common for EEG. Preserves relative amplitude within a channel.
      'minmax' — scale to [0, 1]. Less common; useful when the model expects non-negative input.
    """

    VALID_METHODS = ("zscore", "minmax")

    def __init__(self, method: str = "zscore") -> None:
        """
        Args:
            method: 'zscore' or 'minmax'. Default is 'zscore'.
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown normalization method '{method}'. "
                f"Supported: {self.VALID_METHODS}"
            )
        self.method = method

    def apply(
        self,
        signals: np.ndarray,
        sample_rate: int,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        """Normalize each channel independently."""
        normalized = np.zeros_like(signals, dtype=np.float32)

        for i, channel in enumerate(signals):
            if self.method == "zscore":
                std = channel.std()
                # Guard against flat channels (std=0) which would cause division by zero.
                # A flat channel means no signal — zscore of all-zeros is fine.
                if std < 1e-10:
                    normalized[i] = np.zeros_like(channel)
                else:
                    normalized[i] = (channel - channel.mean()) / std

            elif self.method == "minmax":
                ch_min, ch_max = channel.min(), channel.max()
                rng = ch_max - ch_min
                if rng < 1e-10:
                    normalized[i] = np.zeros_like(channel)
                else:
                    normalized[i] = (channel - ch_min) / rng

        return normalized

    def to_dict(self) -> dict:
        return {"step": "normalize", "method": self.method}

    @classmethod
    def from_dict(cls, config: dict) -> "Normalize":
        return cls(method=config.get("method", "zscore"))


# ---------------------------------------------------------------------------
# SlidingWindow — cut the signal into overlapping fixed-length windows
# ---------------------------------------------------------------------------

class SlidingWindow(PreprocessingStep):
    """
    Segment the signal into overlapping fixed-length windows.

    Why windowing?
      Neural networks need fixed-size inputs. A recording of arbitrary length
      must be cut into equal-length chunks ("windows") before it can be fed
      to a CNN or labeled by a human.

    Why overlapping windows?
      An event (blink, motor imagery) might start at any point in time — not just
      at window boundaries. Overlap ensures we don't miss events that straddle
      two adjacent windows.

    Example: window_sec=1.0, overlap_ratio=0.5, sample_rate=256
      window_samples = 256
      step_samples   = 128  (50% overlap)
      A 10-second recording → 19 windows  (every 0.5s, shifted by 128 samples)

    Note: This step changes the return type. While all other steps return
    np.ndarray, SlidingWindow returns list[np.ndarray]. This is why it must
    always be the last step in a pipeline.
    """

    def __init__(self, window_sec: float, overlap_ratio: float) -> None:
        """
        Args:
            window_sec:    Length of each window in seconds (e.g. 1.0).
            overlap_ratio: Fraction of overlap between adjacent windows (0.0–<1.0).
                           0.5 means 50% overlap.
        """
        if window_sec <= 0:
            raise ValueError(f"window_sec must be positive, got {window_sec}")
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError(
                f"overlap_ratio must be in [0.0, 1.0), got {overlap_ratio}"
            )
        self.window_sec = window_sec
        self.overlap_ratio = overlap_ratio

    def apply(
        self,
        signals: np.ndarray,
        sample_rate: int,
        channel_names: list[str] | None = None,
    ) -> list[np.ndarray]:  # type: ignore[override]
        """
        Cut signals into overlapping windows.

        Returns:
            List of np.ndarray, each of shape (n_channels, window_samples).
            Returns an empty list if the signal is shorter than one window.
        """
        window_samples = int(self.window_sec * sample_rate)
        # Step size: how many samples to advance between window starts.
        # (1 - overlap_ratio) fraction of the window length.
        step_samples = max(1, int(window_samples * (1.0 - self.overlap_ratio)))

        n_channels, n_samples = signals.shape

        if n_samples < window_samples:
            logger.warning(
                "Signal length (%d samples) is shorter than one window (%d samples). "
                "Returning empty window list.",
                n_samples, window_samples,
            )
            return []

        windows: list[np.ndarray] = []

        # Slide the window from the start of the signal to the last full window.
        start = 0
        while start + window_samples <= n_samples:
            window = signals[:, start : start + window_samples]
            windows.append(window.copy())  # copy so each window is independent
            start += step_samples

        logger.info(
            "SlidingWindow: %d windows of %d samples (step %d, overlap %.0f%%)",
            len(windows), window_samples, step_samples, self.overlap_ratio * 100,
        )
        return windows

    def to_dict(self) -> dict:
        return {
            "step": "window",
            "window_sec": self.window_sec,
            "overlap_ratio": self.overlap_ratio,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "SlidingWindow":
        return cls(
            window_sec=config["window_sec"],
            overlap_ratio=config["overlap_ratio"],
        )


# ---------------------------------------------------------------------------
# Rereferencing — change the voltage reference electrode
# ---------------------------------------------------------------------------

class Rereferencing(PreprocessingStep):
    """
    Re-reference EEG signals to a new voltage reference.

    Raw EEG is measured relative to a single physical reference electrode.
    This introduces bias — that electrode's own brain activity contaminates
    every channel. Re-referencing mathematically moves the reference point.

    Supported methods:
      'average': Subtract the mean of all channels from each channel.
                 This is the most common method in BCI research — it distributes
                 the reference signal equally and removes any single-electrode bias.
      'linked_mastoid': Subtract the mean of channels M1 and M2 (mastoid electrodes).
                        Used in clinical EEG. Requires channel_names to find M1/M2.
    """

    VALID_METHODS = ("average", "linked_mastoid")

    def __init__(self, method: str = "average") -> None:
        """
        Args:
            method: 'average' (common average reference) or 'linked_mastoid'.
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown rereferencing method '{method}'. "
                f"Supported: {self.VALID_METHODS}"
            )
        self.method = method

    def apply(
        self,
        signals: np.ndarray,
        sample_rate: int,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        """Apply re-referencing to all channels."""
        if self.method == "average":
            # Subtract the mean across all channels at each time point.
            # signals.mean(axis=0) has shape (n_samples,) — broadcasting works:
            # (n_channels, n_samples) - (n_samples,) = (n_channels, n_samples)
            return signals - signals.mean(axis=0)

        elif self.method == "linked_mastoid":
            if channel_names is None:
                raise ValueError(
                    "linked_mastoid rereferencing requires channel_names to find M1 and M2."
                )
            # Find M1 and M2 indices (case-insensitive search)
            names_lower = [n.lower() for n in channel_names]
            mastoid_indices = [
                i for i, n in enumerate(names_lower)
                if n in ("m1", "m2", "a1", "a2")  # A1/A2 are also common mastoid labels
            ]
            if not mastoid_indices:
                raise ValueError(
                    f"linked_mastoid rereferencing: could not find M1/M2 or A1/A2 channels. "
                    f"Available channels: {channel_names}"
                )
            # Reference = mean of the mastoid channels at each time point
            reference = signals[mastoid_indices].mean(axis=0)
            return signals - reference

        # Should never reach here after __init__ validation
        return signals

    def to_dict(self) -> dict:
        return {"step": "reref", "method": self.method}

    @classmethod
    def from_dict(cls, config: dict) -> "Rereferencing":
        return cls(method=config.get("method", "average"))


# ---------------------------------------------------------------------------
# ChannelSelector — keep only a specified subset of channels
# ---------------------------------------------------------------------------

class ChannelSelector(PreprocessingStep):
    """
    Keep only the specified EEG channels and discard the rest.

    Many EEG devices record 32–256 channels. For motor imagery tasks, you only
    need ~8–16 channels over the motor cortex (C3, Cz, C4 and neighbors).
    Fewer channels = faster training, less noise, smaller model.

    This step MUST be applied BEFORE windowing.

    The channel_names parameter in apply() is required — it maps the requested
    channel names to their indices in the signals array.
    """

    def __init__(self, channels: list[str]) -> None:
        """
        Args:
            channels: Ordered list of channel names to keep.
                      Example: ["C3", "Cz", "C4", "Fp1", "Fp2"]
        """
        if not channels:
            raise ValueError("ChannelSelector requires at least one channel name.")
        self.channels = channels

    def apply(
        self,
        signals: np.ndarray,
        sample_rate: int,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Select and return only the specified channels.

        Raises ValueError if any requested channel is not in channel_names.
        """
        if channel_names is None:
            raise ValueError(
                "ChannelSelector.apply() requires channel_names to be provided. "
                "The Pipeline.run() method should pass session.channel_names."
            )

        # Build a set for fast membership checking
        available = {name: i for i, name in enumerate(channel_names)}

        missing = [ch for ch in self.channels if ch not in available]
        if missing:
            raise ValueError(
                f"ChannelSelector: the following channels were not found in the session: "
                f"{missing}. Available channels: {list(available.keys())}"
            )

        # Extract the rows corresponding to the requested channels, in order
        indices = [available[ch] for ch in self.channels]
        return signals[indices, :]

    def to_dict(self) -> dict:
        return {"step": "channels", "channels": self.channels}

    @classmethod
    def from_dict(cls, config: dict) -> "ChannelSelector":
        return cls(channels=config.get("channels", []))


# ---------------------------------------------------------------------------
# EpochRejection — remove artifact-contaminated windows after windowing
# ---------------------------------------------------------------------------

class EpochRejection(PreprocessingStep):
    """
    Reject windowed EEG epochs (windows) that contain artifacts.

    Must be applied AFTER SlidingWindow. It operates on the list of windows,
    not on the continuous signal. Use Pipeline — it handles the ordering.

    Rule: a window is rejected if ANY sample in ANY channel exceeds ±threshold_uv.
    Spikes of ±100µV or more are almost always artifacts (blink, jaw clench,
    electrode pop), not brain signal. Training on them confuses the model.

    The apply() method is intentionally left as a placeholder — use apply_to_windows().
    """

    DEFAULT_THRESHOLD = 100.0   # microvolts (µV)

    def __init__(self, threshold_uv: float = DEFAULT_THRESHOLD) -> None:
        """
        Args:
            threshold_uv: Amplitude threshold in microvolts. Any window where
                          any channel exceeds ±threshold_uv is rejected.
                          Default 100.0 µV is standard for BCI research.
        """
        if threshold_uv <= 0:
            raise ValueError(f"threshold_uv must be positive, got {threshold_uv}")
        self.threshold_uv = threshold_uv

    def apply(
        self,
        signals: np.ndarray,
        sample_rate: int,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        """Not used directly — call apply_to_windows() for epoch rejection."""
        raise NotImplementedError(
            "EpochRejection must be applied AFTER SlidingWindow. "
            "The Pipeline automatically calls apply_to_windows() for this step."
        )

    def apply_to_windows(
        self,
        windows: list[np.ndarray],
        sample_rate: int,
    ) -> list[np.ndarray]:
        """
        Filter out windows that exceed the amplitude threshold.

        Args:
            windows: List of np.ndarray, each shape (n_channels, n_samples).
            sample_rate: Not used here (kept for consistent interface).

        Returns:
            Filtered list with artifact windows removed.
        """
        kept: list[np.ndarray] = []
        rejected_count = 0

        for window in windows:
            # Check if any sample in any channel exceeds the threshold
            max_amplitude = np.abs(window).max()
            if max_amplitude <= self.threshold_uv:
                kept.append(window)
            else:
                rejected_count += 1
                logger.debug(
                    "Rejected window: max amplitude %.1f µV exceeds threshold %.1f µV",
                    max_amplitude, self.threshold_uv,
                )

        logger.info(
            "EpochRejection: kept %d/%d windows (rejected %d with >.±%.1f µV)",
            len(kept), len(windows), rejected_count, self.threshold_uv,
        )
        return kept

    def to_dict(self) -> dict:
        return {"step": "reject", "threshold_uv": self.threshold_uv}

    @classmethod
    def from_dict(cls, config: dict) -> "EpochRejection":
        return cls(threshold_uv=config.get("threshold_uv", cls.DEFAULT_THRESHOLD))


# ---------------------------------------------------------------------------
# ICAFilter — Independent Component Analysis artifact removal
# ---------------------------------------------------------------------------

class ICAFilter(PreprocessingStep):
    """
    Remove artifacts using Independent Component Analysis (ICA).

    ICA mathematically separates the mixed EEG signal into statistically
    independent source signals ("components"). Some components correspond to
    artifacts: eye blinks appear as large deflections in frontal channels,
    heartbeat as a periodic signal across all channels.

    These artifact components can be identified by index and excluded —
    their contribution is subtracted from all channels simultaneously.

    Uses MNE's ICA implementation which applies FastICA under the hood.

    IMPORTANT: Apply AFTER bandpass filtering, BEFORE windowing and normalization.
    ICA on unfiltered signals is unreliable (DC drift dominates components).
    ICA on normalized signals changes the signal statistics ICA relies on.

    WARNING: ICA fitting is computationally expensive on long recordings.
             It may take several minutes on CPU for recordings > 5 minutes.
    """

    def __init__(
        self,
        n_components: int | None = None,
        exclude_indices: list[int] | None = None,
    ) -> None:
        """
        Args:
            n_components:    Number of ICA components to compute.
                             Default = number of channels (full decomposition).
            exclude_indices: Which component indices to remove (e.g. [0, 1]).
                             These are identified by the user inspecting component
                             topomaps (in future UI). If None, no components removed.
        """
        self.n_components    = n_components
        self.exclude_indices = exclude_indices or []

    def apply(
        self,
        signals: np.ndarray,
        sample_rate: int,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Fit ICA on the signals and apply it, optionally excluding artifact components.

        If exclude_indices is empty, returns the signal reconstructed from all
        components (effectively a no-op for the signal, but the ICA is still fit).
        """
        try:
            import mne
        except ImportError as exc:
            raise ImportError(
                "MNE-Python is required for ICA filtering. "
                "Install it with: pip install mne"
            ) from exc

        n_channels, n_samples = signals.shape
        n_components = self.n_components or n_channels

        # Cap n_components at n_channels (ICA cannot produce more components than inputs)
        n_components = min(n_components, n_channels)

        # Build an MNE RawArray so we can use MNE's ICA class.
        # MNE expects (n_channels, n_samples) shape — matches our convention.
        info = mne.create_info(
            ch_names=channel_names or [f"ch_{i}" for i in range(n_channels)],
            sfreq=float(sample_rate),
            ch_types="eeg",
        )
        raw = mne.io.RawArray(signals.astype(np.float64), info, verbose=False)

        # Fit ICA.
        # method='fastica' is the standard algorithm for EEG.
        # random_state=42 makes the decomposition reproducible.
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method="fastica",
            random_state=42,
            verbose=False,
        )

        logger.info(
            "Fitting ICA: %d components on signal (%d ch, %d samples)...",
            n_components, n_channels, n_samples,
        )
        ica.fit(raw, verbose=False)

        # Exclude specified artifact components
        if self.exclude_indices:
            ica.exclude = self.exclude_indices
            logger.info("ICA excluding components: %s", self.exclude_indices)

        # Apply ICA — reconstruct signal without the excluded components
        raw_cleaned = ica.apply(raw.copy(), verbose=False)

        # Extract the cleaned signal back to numpy
        cleaned = raw_cleaned.get_data().astype(np.float32)
        logger.info("ICA applied. Signal shape: %s", cleaned.shape)

        return cleaned

    def to_dict(self) -> dict:
        return {
            "step":            "ica",
            "n_components":    self.n_components,
            "exclude_indices": self.exclude_indices,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "ICAFilter":
        return cls(
            n_components=   config.get("n_components"),
            exclude_indices=config.get("exclude_indices", []),
        )


# ---------------------------------------------------------------------------
# STEP_REGISTRY — maps JSON "step" strings to their classes
# ---------------------------------------------------------------------------

# This registry is how Pipeline.from_dict() reconstructs a pipeline from saved JSON.
# If a new step type is added in the future, register it here.
STEP_REGISTRY: dict[str, type[PreprocessingStep]] = {
    "bandpass":  BandpassFilter,
    "notch":     NotchFilter,
    "normalize": Normalize,
    "window":    SlidingWindow,
    "reref":     Rereferencing,
    "reject":    EpochRejection,
    "channels":  ChannelSelector,
    "ica":       ICAFilter,
}


# ---------------------------------------------------------------------------
# Pipeline — orchestrates applying a sequence of steps to a DataSession
# ---------------------------------------------------------------------------

class Pipeline:
    """
    A sequence of preprocessing steps applied to a DataSession.

    Usage:
        pipeline = Pipeline([
            BandpassFilter(1.0, 40.0),
            NotchFilter(60.0),
            Normalize("zscore"),
            SlidingWindow(1.0, 0.5),
        ])
        windows = pipeline.run(session)   # list of (n_channels, 256) arrays

    The pipeline can be serialized to the PipelineConfig JSON format and
    reconstructed from it exactly, enabling reproducible experiments.
    """

    def __init__(self, steps: list[PreprocessingStep]) -> None:
        """
        Args:
            steps: Ordered list of preprocessing steps. SlidingWindow, if present,
                   must be last (enforced in run()).
        """
        self.steps = steps

    def run(self, session: DataSession) -> list[np.ndarray]:
        """
        Apply all steps to the session's signals and return windowed output.

        Step ordering (enforced internally regardless of config order):
          1. Pre-window steps (Bandpass, Notch, Normalize, Rereferencing, ChannelSelector,
             ICAFilter) — transform (n_channels, n_samples) → (n_channels, n_samples)
          2. SlidingWindow (if present) — produces list[np.ndarray]
          3. Post-window steps (EpochRejection) — filter the window list

        ChannelSelector tracking: after ChannelSelector runs, the pipeline tracks
        the new (reduced) channel list so subsequent steps receive the right names.

        Args:
            session: The DataSession to process.

        Returns:
            List of np.ndarray windows, each shape (n_channels, window_samples).
        """
        signals = session.signals.copy()  # don't mutate the original session data
        sample_rate = session.sample_rate
        # Track the current channel names — may be reduced by ChannelSelector
        channel_names = list(session.channel_names)

        # Partition steps by phase: pre-window, window, post-window
        pre_window_steps = [
            s for s in self.steps
            if not isinstance(s, (SlidingWindow, EpochRejection))
        ]
        window_step = next(
            (s for s in self.steps if isinstance(s, SlidingWindow)), None
        )
        post_window_steps = [
            s for s in self.steps if isinstance(s, EpochRejection)
        ]

        # Phase 1: Apply all pre-window steps on the full continuous signal.
        # Filtering the whole signal before windowing prevents edge effects
        # (filter transients) at every window boundary.
        for step in pre_window_steps:
            logger.info("Applying step: %s", step.to_dict())
            signals = step.apply(signals, sample_rate, channel_names)

            # If ChannelSelector ran, update the tracked channel names
            # to match the reduced channel set it selected.
            if isinstance(step, ChannelSelector):
                channel_names = step.channels

        # Phase 2: Apply windowing (changes return type to list[np.ndarray])
        if window_step is not None:
            logger.info("Applying step: %s", window_step.to_dict())
            windows = window_step.apply(signals, sample_rate, channel_names)
        else:
            # No windowing — wrap the full processed signal so the return type
            # is always list[np.ndarray], making downstream code uniform.
            windows = [signals]

        # Phase 3: Apply post-window steps (EpochRejection filters out bad windows)
        for step in post_window_steps:
            logger.info("Applying epoch rejection: %s", step.to_dict())
            windows = step.apply_to_windows(windows, sample_rate)

        return windows

    def to_dict(self) -> dict:
        """
        Serialize to the PipelineConfig JSON format from CLAUDE.md.
        Generates a fresh pipeline_id UUID each time — this is intentional:
        each saved config is a new unique artifact.
        """
        # session_id and sample_rate are placeholders here; they are filled in
        # by the route handler which knows the active session.
        return {
            "pipeline_id": str(uuid.uuid4()),
            "session_id":  None,   # filled in by pipeline_routes.py
            "sample_rate": None,   # filled in by pipeline_routes.py
            "steps":       [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, config: dict) -> "Pipeline":
        """
        Reconstruct a Pipeline from a PipelineConfig dict.

        Looks up each step's class in STEP_REGISTRY using the "step" key,
        then calls that class's from_dict() to restore its parameters.
        """
        steps: list[PreprocessingStep] = []
        for step_config in config.get("steps", []):
            step_name = step_config.get("step")
            step_class = STEP_REGISTRY.get(step_name)
            if step_class is None:
                raise ValueError(
                    f"Unknown preprocessing step '{step_name}'. "
                    f"Available steps: {list(STEP_REGISTRY.keys())}"
                )
            steps.append(step_class.from_dict(step_config))
        return cls(steps)
