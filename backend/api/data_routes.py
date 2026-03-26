"""
data_routes.py — HTTP endpoints for EEG file upload and session management.

This file is the "thin API layer": it receives HTTP requests, validates them,
calls functions from core/data_loader.py, and returns APIResponse dicts.
There is no signal processing logic here.

Registered in main.py with prefix /api/data.
"""

from __future__ import annotations

import logging
import os
import shutil

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from fastapi import Form
from core.data_loader import (
    EventMarker,
    get_session_summary,
    load_csv,
    load_edf,
    load_edf_with_events,
    parse_event_file,
)
from sessions.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter()
store = SessionStore()  # always returns the singleton — same object as in all other route files

# Directory where uploaded files are temporarily saved before loading.
# Using a subdirectory keeps uploads isolated from source code.
TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "temp")


# ---------------------------------------------------------------------------
# Helper: build a standard APIResponse dict
# ---------------------------------------------------------------------------

def api_response(success: bool, data=None, error: str | None = None) -> dict:
    """
    Build the standard APIResponse shape defined in CLAUDE.md.
    Every endpoint returns this — the frontend always knows where to look.
    """
    return {"success": success, "data": data, "error": error}


# ---------------------------------------------------------------------------
# POST /api/data/upload
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_file(file: UploadFile) -> JSONResponse:
    """
    Accept an uploaded EEG file, save it temporarily, load it into a DataSession,
    store the session in SessionStore, and return the session summary.

    Supported file types: .csv, .edf
    """
    # Ensure the temp directory exists (it may not on a fresh clone)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Build the temp file path from the original filename.
    # This preserves the extension so we can detect the file type below.
    filename = file.filename or "upload"
    temp_path = os.path.join(TEMP_DIR, filename)

    # Write the uploaded bytes to disk.
    # We must save to disk (not a BytesIO buffer) because MNE's EDF reader
    # requires a real file path, not a file-like object.
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"Could not save uploaded file: {exc}"),
        )

    # Detect file type from the extension (lowercase for robustness)
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".csv":
            # load_csv will parse the sample rate from the # sample_rate= comment.
            session = load_csv(temp_path)
        elif ext == ".edf":
            session = load_edf(temp_path)
        else:
            # Return a clear error for unsupported formats — not a 500 (server error),
            # but a 400 (bad request from the user).
            return JSONResponse(
                status_code=400,
                content=api_response(
                    False,
                    error=f"Unsupported file type '{ext}'. Supported formats: .csv, .edf",
                ),
            )
    except ValueError as exc:
        # ValueError from load_csv means missing sample rate or parse failure
        return JSONResponse(
            status_code=400,
            content=api_response(False, error=str(exc)),
        )
    except Exception as exc:
        logger.error("Error loading file %s: %s", filename, exc)
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"Failed to load file: {exc}"),
        )

    # Store in the singleton so all other routes can access it by session_id
    store.sessions[session.session_id] = session
    logger.info("Session created: %s", session.session_id)

    return JSONResponse(
        status_code=200,
        content=api_response(True, data=get_session_summary(session)),
    )


# ---------------------------------------------------------------------------
# GET /api/data/sessions
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def list_sessions() -> JSONResponse:
    """
    Return metadata for all currently loaded sessions.
    Signal arrays are never included — only lightweight summary dicts.
    """
    summaries = [get_session_summary(s) for s in store.sessions.values()]
    return JSONResponse(
        status_code=200,
        content=api_response(True, data=summaries),
    )


# ---------------------------------------------------------------------------
# GET /api/data/{session_id}/preview
# ---------------------------------------------------------------------------

@router.get("/{session_id}/preview")
async def get_session_preview(session_id: str, max_samples: int = 512) -> JSONResponse:
    """
    Return a downsampled slice of the session's signal data formatted for Recharts.

    Why a separate endpoint instead of including signals in the summary?
      The full signal array can be megabytes — not safe to include in every
      metadata response. This endpoint exists only when the frontend explicitly
      needs to render a chart.

    Why downsample to max_samples (default 512)?
      Recharts renders every data point as a DOM element. At 256 Hz for 10 seconds
      that's 2560 points — fine here, but a 5-minute recording would be 76,800 points
      which would freeze the browser. Downsampling keeps the chart fast at any length.

    Returns data in Recharts LineChart format:
      [{ "time": 0.0, "Fp1": 0.12, "Fp2": -0.03, ... }, ...]
    """
    session = store.sessions.get(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Session '{session_id}' not found."),
        )

    import numpy as np

    signals = session.signals          # shape: (n_channels, n_samples)
    n_channels, n_samples = signals.shape

    # Calculate the downsampling step — how many original samples to skip per output point.
    # If the recording is already short enough, step=1 (no downsampling).
    step = max(1, n_samples // max_samples)

    # Take every `step`-th sample along the time axis
    downsampled = signals[:, ::step]   # shape: (n_channels, downsampled_samples)
    n_out = downsampled.shape[1]

    # Build the time axis in seconds for the X axis of the chart
    time_axis = [round(i * step / session.sample_rate, 4) for i in range(n_out)]

    # Convert to Recharts format: list of dicts, one dict per time point
    # Each dict has "time" plus one key per channel name
    points = []
    for i in range(n_out):
        point = {"time": time_axis[i]}
        for ch_idx, ch_name in enumerate(session.channel_names):
            # Round to 6 decimal places — enough precision for EEG, avoids huge JSON
            point[ch_name] = round(float(downsampled[ch_idx, i]), 6)
        points.append(point)

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "points":        points,
            "channel_names": session.channel_names,
            "sample_rate":   session.sample_rate,
            "n_samples_original": n_samples,
            "n_samples_preview":  n_out,
        }),
    )


# ---------------------------------------------------------------------------
# GET /api/data/{session_id}
# ---------------------------------------------------------------------------

@router.get("/{session_id}")
async def get_session(session_id: str) -> JSONResponse:
    """
    Return metadata for a single session by its UUID.
    Returns HTTP 404 if the session_id is not in the store.
    """
    session = store.sessions.get(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=api_response(
                False,
                error=f"Session '{session_id}' not found. "
                      "It may not have been uploaded, or the server was restarted.",
            ),
        )
    return JSONResponse(
        status_code=200,
        content=api_response(True, data=get_session_summary(session)),
    )


# ---------------------------------------------------------------------------
# DELETE /api/data/{session_id}
# ---------------------------------------------------------------------------

@router.delete("/{session_id}")
async def delete_session(session_id: str) -> JSONResponse:
    """
    Remove a session from the store by its UUID.
    Also removes any pipeline runs associated with this session.
    Returns 404 if the session does not exist.
    """
    if session_id not in store.sessions:
        return JSONResponse(
            status_code=404,
            content=api_response(
                False,
                error=f"Session '{session_id}' not found.",
            ),
        )

    del store.sessions[session_id]

    # Clean up any pipeline runs that were generated from this session
    # so we don't accumulate stale data in memory.
    # pipeline_runs now stores dicts: {"session_id": ..., "windows": ..., ...}
    runs_to_remove = [
        run_id for run_id, run in store.pipeline_runs.items()
        if isinstance(run, dict) and run.get("session_id") == session_id
    ]
    for run_id in runs_to_remove:
        del store.pipeline_runs[run_id]

    logger.info("Session deleted: %s", session_id)

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={"deleted_session_id": session_id}),
    )


# ---------------------------------------------------------------------------
# POST /api/data/upload-events
# ---------------------------------------------------------------------------

@router.post("/upload-events")
async def upload_events(
    file: UploadFile,
    session_id: str = Form(...),
) -> JSONResponse:
    """
    Accept a standalone event/annotation file, parse it into EventMarkers,
    and store them for the given session.

    Supported formats: .tsv (BIDS), .eve (MNE), .vmrk (BrainVision).
    The format is auto-detected from the file extension.

    After uploading events, call POST /api/labels/from-events to automatically
    label the windowed data from a pipeline run using these markers.
    """
    # Validate session exists before parsing the file
    session = store.sessions.get(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=api_response(
                False,
                error=f"Session '{session_id}' not found. Upload the EEG file first.",
            ),
        )

    # Save the event file to temp/ with a unique prefix to avoid collisions
    import uuid as _uuid
    os.makedirs(TEMP_DIR, exist_ok=True)
    filename = file.filename or "events"
    unique_name = f"{_uuid.uuid4().hex[:8]}_{filename}"
    temp_path = os.path.join(TEMP_DIR, unique_name)

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"Could not save event file: {exc}"),
        )

    # Parse the event file using the session's sample rate for unit conversion
    try:
        markers = parse_event_file(temp_path, sample_rate=session.sample_rate)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content=api_response(False, error=str(exc)),
        )
    except Exception as exc:
        logger.error("Error parsing event file %s: %s", filename, exc)
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"Failed to parse event file: {exc}"),
        )

    # Store in singleton — accessible by label_routes.py for from-events endpoint
    store.event_markers[session_id] = markers

    logger.info(
        "Uploaded %d event markers for session %s (file: %s)",
        len(markers), session_id, filename,
    )

    # Return summary of parsed markers
    label_counts: dict[str, int] = {}
    for m in markers:
        label_counts[m.label] = label_counts.get(m.label, 0) + 1

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "session_id":   session_id,
            "n_markers":    len(markers),
            "label_counts": label_counts,
            "markers": [
                {
                    "onset_sec":    m.onset_sec,
                    "duration_sec": m.duration_sec,
                    "label":        m.label,
                }
                for m in markers[:20]   # first 20 for preview (don't send all)
            ],
        }),
    )
