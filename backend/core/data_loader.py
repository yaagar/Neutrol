"""
data_loader.py — Load EEG files (CSV or EDF) into DataSession objects.

This is a pure-Python module: no HTTP, no FastAPI, no routes.
It receives file paths and returns DataSession objects.
The API layer (data_routes.py) is the only caller.

Why keep I/O logic here and not in the route?
  If you ever want to load a file from a script or a test, you can call these
  functions directly without running a web server.
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataSession — the canonical container for a loaded EEG recording.
# Defined exactly as in CLAUDE.md. Do not add or remove fields.
# ---------------------------------------------------------------------------

@dataclass
class DataSession:
    """
    Holds everything about one loaded EEG recording.

    Shape convention: signals is always (n_channels, n_samples).
    This matches MNE-Python's convention and avoids transposing mid-pipeline.
    """

    session_id: str           # UUID generated at load time — uniquely identifies this recording
    signals: np.ndarray       # Shape: (n_channels, n_samples) — the raw voltage data
    sample_rate: int          # Samples per second, e.g. 256 Hz
    channel_names: list[str]  # One name per row of signals, e.g. ["Fp1", "C3", ...]
    duration_seconds: float   # Total recording length in seconds
    file_path: str            # Path to the original file (for reference/logging)
    created_at: str           # ISO-8601 UTC timestamp of when this session was loaded
    metadata: dict = field(default_factory=dict)  # Optional extras (unused in MVP)


# ---------------------------------------------------------------------------
# LabeledWindow — one labeled EEG segment ready for training.
# Defined exactly as in CLAUDE.md. Do not add or remove fields.
# ---------------------------------------------------------------------------

@dataclass
class LabeledWindow:
    """
    A single EEG window that has been assigned a class label by the user.

    These are the training examples. After labeling, they are passed to
    the trainer (Session 4) which feeds them into the CNN.

    Why store `data` (numpy array) here?
      The trainer needs the actual signal values, not just metadata.
      The label routes store LabeledWindow objects in memory and expose
      only JSON-safe metadata over the API.
    """

    window_id: str          # UUID — unique identifier for this window
    data: np.ndarray        # Shape: (n_channels, n_samples_in_window)
    label: str              # e.g. "rest", "left", "right", "blink"
    start_time_sec: float   # Where in the original recording this window starts
    session_id: str         # Which session this came from


# ---------------------------------------------------------------------------
# load_csv — read a .csv EEG file into a DataSession
# ---------------------------------------------------------------------------

def load_csv(file_path: str, sample_rate: Optional[int] = None) -> DataSession:
    """
    Load a CSV EEG file.

    Expected format:
      - Optional first line: # sample_rate=256
      - Header row: channel names (e.g. Fp1,Fp2,C3,...)
      - Data rows: one row per time step, one column per channel

    Args:
        file_path:   Path to the .csv file.
        sample_rate: Override sample rate. If None, the file must declare it
                     in a comment on the first line.

    Returns:
        A DataSession with signals of shape (n_channels, n_samples).

    Raises:
        ValueError: If no sample rate can be determined.
        FileNotFoundError: If file_path does not exist.
    """
    logger.info("Loading CSV: %s", file_path)

    # Read the raw text so we can inspect the first line before pandas or numpy sees it
    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # Check if the first line is a sample rate comment (e.g. "# sample_rate=256")
    header_offset = 0  # how many lines to skip before the column-name row
    if raw_lines and raw_lines[0].startswith("# sample_rate="):
        # Parse the integer after the "=" sign
        try:
            declared_rate = int(raw_lines[0].strip().split("=")[1])
            # Only use the declared rate if the caller didn't override it
            if sample_rate is None:
                sample_rate = declared_rate
        except (IndexError, ValueError):
            logger.warning("Could not parse sample_rate from comment line: %s", raw_lines[0])
        header_offset = 1  # the actual CSV header is on line 2

    # After checking the comment, we must know the sample rate
    if sample_rate is None:
        raise ValueError(
            "Sample rate could not be determined. "
            "Either include '# sample_rate=<hz>' as the first line of the CSV, "
            "or pass sample_rate explicitly to load_csv()."
        )

    # Parse the CSV using numpy — skip the comment line if present.
    # np.genfromtxt handles missing values gracefully; we want strict loading here
    # so we use np.loadtxt with skip_header.
    #
    # skip_header skips the comment line (if any); names=True reads the next line
    # as column headers.
    data_with_names = np.genfromtxt(
        file_path,
        delimiter=",",
        skip_header=header_offset,  # skip the # sample_rate= comment if present
        names=True,                 # treat the next row as column names
        encoding="utf-8",
    )

    # genfromtxt with names=True returns a structured array.
    # Extract column names and convert to a regular 2-D float array.
    channel_names: list[str] = list(data_with_names.dtype.names)

    # Stack columns into a 2-D array of shape (n_samples, n_channels),
    # then transpose to (n_channels, n_samples) — our canonical shape.
    signals: np.ndarray = np.column_stack(
        [data_with_names[name] for name in channel_names]
    ).T.astype(np.float32)

    n_channels, n_samples = signals.shape
    duration_seconds = n_samples / sample_rate

    logger.info(
        "Loaded CSV: %d channels, %d samples, %.2fs @ %dHz",
        n_channels, n_samples, duration_seconds, sample_rate,
    )

    return DataSession(
        session_id=str(uuid.uuid4()),
        signals=signals,
        sample_rate=sample_rate,
        channel_names=channel_names,
        duration_seconds=duration_seconds,
        file_path=file_path,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# load_edf — read an EDF file into a DataSession using MNE-Python
# ---------------------------------------------------------------------------

def load_edf(file_path: str) -> DataSession:
    """
    Load an EDF (European Data Format) EEG file using MNE-Python.

    EDF is the standard binary format for clinical EEG recordings.
    MNE handles the binary parsing, channel metadata, and unit conversions.

    Args:
        file_path: Path to the .edf file.

    Returns:
        A DataSession with signals of shape (n_channels, n_samples).
    """
    # Import MNE here (not at module top) so the rest of the module works even
    # if MNE is not installed — CSV loading doesn't require MNE.
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "MNE-Python is required to load EDF files. "
            "Install it with: pip install mne"
        ) from exc

    logger.info("Loading EDF: %s", file_path)

    # preload=True loads the entire signal into memory immediately.
    # Without it, MNE uses lazy loading which complicates downstream processing.
    # verbose=False suppresses MNE's own progress output.
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # raw.get_data() already returns (n_channels, n_samples) — no transpose needed.
    signals: np.ndarray = raw.get_data().astype(np.float32)

    # sfreq is MNE's attribute for sampling frequency (a float); we want an int
    sample_rate: int = int(raw.info["sfreq"])

    channel_names: list[str] = raw.info["ch_names"]

    n_channels, n_samples = signals.shape
    duration_seconds = n_samples / sample_rate

    logger.info(
        "Loaded EDF: %d channels, %d samples, %.2fs @ %dHz",
        n_channels, n_samples, duration_seconds, sample_rate,
    )

    return DataSession(
        session_id=str(uuid.uuid4()),
        signals=signals,
        sample_rate=sample_rate,
        channel_names=channel_names,
        duration_seconds=duration_seconds,
        file_path=file_path,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# get_session_summary — serialize a DataSession to a JSON-safe dict
# ---------------------------------------------------------------------------

def get_session_summary(session: DataSession) -> dict:
    """
    Return a JSON-serializable summary of a DataSession.

    We deliberately exclude `session.signals` (a numpy array) because:
      1. It can be many megabytes — too large to send over the API.
      2. The frontend doesn't need the raw samples; it needs metadata to display
         in the session list and to identify which session to preprocess.

    The signals stay in the SessionStore (server-side) and are accessed by
    session_id when the pipeline runs.

    Returns:
        A plain dict with only JSON-safe types (str, int, float, list[str]).
    """
    n_channels, n_samples = session.signals.shape

    return {
        "session_id":       session.session_id,
        "channel_names":    session.channel_names,
        "sample_rate":      session.sample_rate,
        "duration_seconds": session.duration_seconds,
        "n_channels":       n_channels,
        "n_samples":        n_samples,
        "file_path":        session.file_path,
        "created_at":       session.created_at,
    }
