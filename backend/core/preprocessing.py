"""
preprocessing.py — EEG preprocessing steps and the Pipeline class.

Design principles (from CLAUDE.md):
  - Each step is a pure class: same input always produces same output, no side effects.
  - All steps implement the PreprocessingStep interface: apply(), to_dict(), from_dict().
  - Steps can be serialized to/from the PipelineConfig JSON format, so a saved config
    exactly reproduces any previous result.
  - SlidingWindow is always the last step — it changes the return type from
    np.ndarray to list[np.ndarray].

Calling convention for all steps:
  Input signals shape:  (n_channels, n_samples)
  Output signals shape: (n_channels, n_samples)   [except SlidingWindow]
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

    def apply(self, signals: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply this step to the signals.

        Args:
            signals:     Input array of shape (n_channels, n_samples).
            sample_rate: Sampling frequency in Hz — needed by filter design functions.

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

    def apply(self, signals: np.ndarray, sample_rate: int) -> np.ndarray:
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

    def apply(self, signals: np.ndarray, sample_rate: int) -> np.ndarray:
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

    def apply(self, signals: np.ndarray, sample_rate: int) -> np.ndarray:
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

    def apply(self, signals: np.ndarray, sample_rate: int) -> list[np.ndarray]:  # type: ignore[override]
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
# STEP_REGISTRY — maps JSON "step" strings to their classes
# ---------------------------------------------------------------------------

# This registry is how Pipeline.from_dict() reconstructs a pipeline from saved JSON.
# If a new step type is added in the future, register it here.
STEP_REGISTRY: dict[str, type[PreprocessingStep]] = {
    "bandpass":  BandpassFilter,
    "notch":     NotchFilter,
    "normalize": Normalize,
    "window":    SlidingWindow,
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

        The pipeline applies steps in order:
          1. All non-SlidingWindow steps transform (n_channels, n_samples) → (n_channels, n_samples)
          2. SlidingWindow (if present, always last) produces list[np.ndarray]
          3. If no SlidingWindow, wrap the final signal in a single-element list
             so the return type is always list[np.ndarray]

        Args:
            session: The DataSession to process.

        Returns:
            List of np.ndarray windows, each shape (n_channels, window_samples).
        """
        signals = session.signals.copy()  # don't mutate the original session data
        sample_rate = session.sample_rate

        # Separate SlidingWindow from the other steps.
        # We process it last so all filtering happens on the full-length signal.
        # Filtering windowed segments separately would cause edge effects at each
        # window boundary.
        filter_steps = [s for s in self.steps if not isinstance(s, SlidingWindow)]
        window_step = next((s for s in self.steps if isinstance(s, SlidingWindow)), None)

        # Apply all filter/normalization steps sequentially
        for step in filter_steps:
            logger.info("Applying step: %s", step.to_dict())
            signals = step.apply(signals, sample_rate)

        # Apply windowing last (changes return type)
        if window_step is not None:
            return window_step.apply(signals, sample_rate)
        else:
            # No windowing — wrap the full processed signal so the return type
            # is always list[np.ndarray], making downstream code uniform.
            return [signals]

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
