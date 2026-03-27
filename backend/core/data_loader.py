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

import csv
import os
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
# EventMarker — a single event parsed from an event/annotation file.
# Defined exactly as in CLAUDE.md.
# ---------------------------------------------------------------------------

@dataclass
class EventMarker:
    """
    A single timestamped event parsed from an event or annotation file.

    Many EEG datasets store class labels separately from the signal.
    A marker file lists "at 3.5 seconds, event T1 occurred, lasting 4 seconds."
    Parsing these markers lets us skip manual window-by-window labeling.

    onset_sec:   When the event starts, in seconds from the recording start.
    duration_sec: How long the event lasts. 0.0 = instantaneous marker.
    label:        The class name, e.g. "T1", "T2", "rest", "left".
    description:  The raw annotation string from the file (for debugging).
    """

    onset_sec:    float   # seconds from recording start
    duration_sec: float   # 0.0 if instantaneous
    label:        str     # the class label
    description:  str     # raw annotation string from the file


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
    run_id: str = ""        # Which pipeline run this window came from


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


# ---------------------------------------------------------------------------
# load_edf_with_events — load EDF and extract embedded annotations as events
# ---------------------------------------------------------------------------

def load_edf_with_events(file_path: str) -> tuple[DataSession, list[EventMarker]]:
    """
    Load an EDF file and extract any embedded event annotations.

    Many clinical and research EDF files embed event markers (annotations) directly
    in the file. MNE reads these automatically with read_raw_edf and
    events_from_annotations(). This function returns both the DataSession
    (same as load_edf) and a list of EventMarker objects.

    If the EDF has no annotations, returns the session + an empty list.

    Args:
        file_path: Path to the .edf file.

    Returns:
        (DataSession, list[EventMarker]) — session with signal data, and any events.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "MNE-Python is required to load EDF files. "
            "Install it with: pip install mne"
        ) from exc

    logger.info("Loading EDF with events: %s", file_path)

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Build the DataSession (same logic as load_edf)
    signals: np.ndarray = raw.get_data().astype(np.float32)
    sample_rate: int = int(raw.info["sfreq"])
    channel_names: list[str] = raw.info["ch_names"]
    n_channels, n_samples = signals.shape
    duration_seconds = n_samples / sample_rate

    session = DataSession(
        session_id=str(uuid.uuid4()),
        signals=signals,
        sample_rate=sample_rate,
        channel_names=channel_names,
        duration_seconds=duration_seconds,
        file_path=file_path,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # Extract annotations (embedded event markers) from the EDF file.
    # mne.events_from_annotations returns:
    #   events: np.ndarray shape (n_events, 3) — [sample_idx, prev_id, event_id]
    #   event_id: dict mapping event description strings to integer IDs
    markers: list[EventMarker] = []
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        # Invert the event_id dict to map integer → description string
        id_to_desc = {v: k for k, v in event_id.items()}

        for event in events:
            sample_idx = int(event[0])
            ev_id      = int(event[2])
            onset_sec  = sample_idx / sample_rate
            description = id_to_desc.get(ev_id, str(ev_id))

            # Check if the raw annotations have a duration for this event.
            # MNE stores onsets, durations, and descriptions in raw.annotations.
            duration_sec = 0.0
            for ann in raw.annotations:
                if abs(ann["onset"] - onset_sec) < 1e-3:
                    duration_sec = float(ann["duration"])
                    break

            markers.append(EventMarker(
                onset_sec=onset_sec,
                duration_sec=duration_sec,
                label=description,
                description=description,
            ))

        logger.info("Extracted %d event markers from EDF annotations", len(markers))

    except Exception as exc:
        # No annotations or annotation parsing failed — not an error
        logger.info("No annotations found in EDF (or parse error): %s", exc)

    return session, markers


# ---------------------------------------------------------------------------
# parse_event_file — parse a standalone event/annotation file
# ---------------------------------------------------------------------------

def parse_event_file(
    file_path: str,
    sample_rate: int = 256,
    format: str = "auto",
) -> list[EventMarker]:
    """
    Parse a standalone event file into EventMarker objects.

    Supported formats (detected automatically from file extension if format='auto'):
      'bids_tsv'   — .tsv file with columns: onset, duration, trial_type
      'mne_eve'    — .eve file: tab-separated sample_index, prev_id, event_id
      'vmrk'       — BrainVision .vmrk marker file (text format)

    Args:
        file_path:   Path to the event file.
        sample_rate: Hz — needed to convert sample indices to seconds for .eve files.
        format:      'auto' to detect from extension, or specify explicitly.

    Returns:
        List of EventMarker objects, sorted by onset_sec.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if format == "auto":
        if ext == ".tsv":
            format = "bids_tsv"
        elif ext == ".eve":
            format = "mne_eve"
        elif ext == ".vmrk":
            format = "vmrk"
        elif ext == ".event":
            format = "edf_event"
        else:
            raise ValueError(
                f"Cannot detect event file format from extension '{ext}'. "
                "Supported: .tsv (BIDS), .eve (MNE), .vmrk (BrainVision), .event (EDF). "
                "Pass format='bids_tsv'|'mne_eve'|'vmrk'|'edf_event' to specify manually."
            )

    if format == "bids_tsv":
        return _parse_bids_tsv(file_path)
    elif format == "mne_eve":
        return _parse_mne_eve(file_path, sample_rate)
    elif format == "vmrk":
        return _parse_vmrk(file_path, sample_rate)
    elif format == "edf_event":
        return _parse_edf_event(file_path)
    else:
        raise ValueError(
            f"Unknown event file format '{format}'. "
            "Supported: 'bids_tsv', 'mne_eve', 'vmrk', 'edf_event'."
        )


def _parse_edf_event(file_path: str) -> list[EventMarker]:
    """
    Parse an MNE-style .edf.event binary file.

    This format is produced when MNE exports the annotations from an EDF+ file.
    It is a mixed binary/text format: binary timestamp bytes are interspersed with
    plain ASCII text records in the form "<label> duration: <seconds>".

    Because decoding the binary timestamp bytes is non-trivial (they appear to be
    a custom internal MNE format), we extract labels and durations from the text
    content and compute onset times cumulatively. This is valid for datasets like
    PhysioNet where annotations are back-to-back with no gaps.

    Format found in the file (example):
      ... binary bytes ... T0 duration: 4.2 ... T1 duration: 4.1 ...

    Returns:
        List of EventMarker objects with cumulative onset times, sorted by onset.
    """
    import re

    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    # Extract all label+duration pairs from the binary blob using regex.
    # The text is ASCII-encoded inside the binary file.
    # Pattern matches things like: "T0 duration: 4.2" or "T1 duration: 4.10"
    events_raw = re.findall(rb'([A-Za-z][A-Za-z0-9_]*)\s+duration:\s+([0-9]+(?:\.[0-9]+)?)', raw_bytes)

    if not events_raw:
        # Try a looser pattern — some files might omit the "duration:" keyword
        # and just list the label followed by a float
        events_raw = re.findall(rb'([A-Za-z][A-Za-z0-9_/]*)\s+([0-9]+\.[0-9]+)', raw_bytes)

    if not events_raw:
        logger.warning("No events found in .edf.event file: %s", file_path)
        return []

    # Compute cumulative onset times: events are assumed to be sequential
    # (each starts immediately after the previous one ends).
    # This holds for PhysioNet and most BCI paradigms where rest and task
    # periods fill the entire recording.
    markers: list[EventMarker] = []
    onset = 0.0

    for label_bytes, duration_bytes in events_raw:
        label       = label_bytes.decode("ascii", errors="replace").strip()
        duration_sec = float(duration_bytes.decode("ascii"))

        markers.append(EventMarker(
            onset_sec=   round(onset, 4),
            duration_sec=duration_sec,
            label=       label,
            description= f"{label} duration: {duration_sec}",
        ))
        onset += duration_sec

    logger.info(
        "Parsed %d events from .edf.event file: %s (total %.2fs)",
        len(markers), file_path, onset,
    )
    return markers


def _parse_bids_tsv(file_path: str) -> list[EventMarker]:
    """Parse a BIDS-format TSV event file (onset, duration, trial_type columns)."""
    markers: list[EventMarker] = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                onset    = float(row.get("onset", 0))
                duration = float(row.get("duration", 0))
                label    = row.get("trial_type", row.get("event_type", "unknown")).strip()
                markers.append(EventMarker(
                    onset_sec=onset,
                    duration_sec=duration,
                    label=label,
                    description=str(row),
                ))
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping malformed BIDS TSV row: %s — %s", row, exc)

    logger.info("Parsed %d events from BIDS TSV: %s", len(markers), file_path)
    return sorted(markers, key=lambda m: m.onset_sec)


def _parse_mne_eve(file_path: str, sample_rate: int) -> list[EventMarker]:
    """Parse an MNE .eve file (tab-separated: sample_idx, prev_id, event_id)."""
    markers: list[EventMarker] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                sample_idx = int(parts[0])
                event_id   = int(parts[2])
                onset_sec  = sample_idx / sample_rate
                label      = str(event_id)
                markers.append(EventMarker(
                    onset_sec=onset_sec,
                    duration_sec=0.0,
                    label=label,
                    description=line,
                ))
            except ValueError as exc:
                logger.warning("Skipping malformed .eve line: %s — %s", line, exc)

    logger.info("Parsed %d events from MNE .eve: %s", len(markers), file_path)
    return sorted(markers, key=lambda m: m.onset_sec)


def _parse_vmrk(file_path: str, sample_rate: int) -> list[EventMarker]:
    """Parse a BrainVision .vmrk marker file."""
    markers: list[EventMarker] = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            # Marker lines look like: Mk1=Stimulus,S  1,1024,1,0
            if not line.startswith("Mk") or "=" not in line:
                continue
            try:
                # Format: Mk<n>=<type>,<description>,<sample>,<length>,<channel>
                parts = line.split("=", 1)[1].split(",")
                if len(parts) < 3:
                    continue
                description = parts[1].strip()
                sample_idx  = int(parts[2].strip())
                duration_samples = int(parts[3].strip()) if len(parts) > 3 else 0

                onset_sec    = sample_idx / sample_rate
                duration_sec = duration_samples / sample_rate
                label        = description

                markers.append(EventMarker(
                    onset_sec=onset_sec,
                    duration_sec=duration_sec,
                    label=label,
                    description=line,
                ))
            except (ValueError, IndexError) as exc:
                logger.warning("Skipping malformed .vmrk line: %s — %s", line, exc)

    logger.info("Parsed %d events from .vmrk: %s", len(markers), file_path)
    return sorted(markers, key=lambda m: m.onset_sec)


# ---------------------------------------------------------------------------
# apply_events_to_windows — match event markers to windowed signal segments
# ---------------------------------------------------------------------------

def apply_events_to_windows(
    windows: list[np.ndarray],
    events: list[EventMarker],
    sample_rate: int,
    window_sec: float,
    session_id: str = "",
    run_id: str = "",
    step_samples: int | None = None,
    signal_offset_sec: float = 0.0,
) -> list[LabeledWindow]:
    """
    Match event markers to windowed signal segments and produce LabeledWindows.

    For each window, this function looks for an event whose onset falls within
    the window's time range. If found, the window is labeled with that event.
    Windows with no matching event get label 'unknown'.

    Rule: an event is assigned to a window if its onset_sec falls within
    [window_start_sec, window_start_sec + window_sec).

    Args:
        windows:      List of np.ndarray windows, each shape (n_channels, n_samples).
        events:       Sorted list of EventMarker objects.
        sample_rate:  Hz — used to compute window start times.
        window_sec:         Duration of each window in seconds.
        session_id:         UUID of the source session (for LabeledWindow.session_id).
        run_id:             UUID of the pipeline run (for LabeledWindow.run_id).
        step_samples:       How many samples between window starts (for non-overlapping windows,
                            step = window_samples; for 50% overlap, step = window_samples // 2).
                            If None, defaults to window_samples (no overlap).
        signal_offset_sec:  Seconds to add to each window's start time. Non-zero when the
                            pipeline includes a TruncateSignal step that removed the first N
                            seconds — event markers are in original-recording time, but windows
                            are indexed from 0 within the truncated signal.

    Returns:
        List of LabeledWindow objects, one per input window.
    """
    if not windows:
        return []

    window_samples = windows[0].shape[1]
    if step_samples is None:
        step_samples = window_samples  # no overlap by default

    labeled: list[LabeledWindow] = []

    for i, window in enumerate(windows):
        # Calculate the time range this window covers.
        # signal_offset_sec shifts window times back into original-recording time
        # so they align with event marker onsets (which use original recording time).
        start_sec = signal_offset_sec + (i * step_samples) / sample_rate
        end_sec   = start_sec + window_sec

        # Find the event that is ACTIVE during this window's start time.
        # An event is active if the window starts within the event's time span:
        #   event.onset_sec <= window_start < event.onset_sec + event.duration_sec
        # This correctly labels all windows within a 4-second event, not just the first.
        matched_label = "unknown"
        for event in events:
            if event.onset_sec > start_sec:
                break  # events are sorted — all remaining start later than this window
            event_end = event.onset_sec + event.duration_sec
            if event.onset_sec <= start_sec < event_end:
                matched_label = event.label
                # Don't break — a later event might also cover this window (take last match)

        labeled.append(LabeledWindow(
            window_id=     str(uuid.uuid4()),
            data=          window,
            label=         matched_label,
            start_time_sec=start_sec,
            session_id=    session_id,
            run_id=        run_id,
        ))

    known    = sum(1 for lw in labeled if lw.label != "unknown")
    unknown  = len(labeled) - known
    logger.info(
        "Applied events to %d windows: %d labeled, %d unknown",
        len(labeled), known, unknown,
    )
    return labeled
