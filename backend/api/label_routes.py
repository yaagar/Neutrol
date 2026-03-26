"""
label_routes.py — HTTP endpoints for saving and loading EEG segment labels.

The labeling workflow:
  1. User has a pipeline run (list of windows in store.pipeline_runs)
  2. User assigns a label to each window in the UI
  3. Frontend sends POST /api/labels/save with {session_id, run_id, labels: [...]}
  4. This route looks up the actual numpy windows from the pipeline run,
     wraps each one in a LabeledWindow dataclass, and stores in labeled_data

The numpy data stays in memory — only JSON-safe metadata is sent over the API.

Registered in main.py with prefix /api/labels.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.data_loader import LabeledWindow
from sessions.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter()
store = SessionStore()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def api_response(success: bool, data=None, error: str | None = None) -> dict:
    """Standard APIResponse shape from CLAUDE.md."""
    return {"success": success, "data": data, "error": error}


def labeled_window_summary(lw: LabeledWindow) -> dict:
    """
    Return a JSON-safe summary of a LabeledWindow.
    Omits the numpy `data` field — same reason as get_session_summary().
    """
    return {
        "window_id":      lw.window_id,
        "label":          lw.label,
        "start_time_sec": lw.start_time_sec,
        "session_id":     lw.session_id,
    }


# ---------------------------------------------------------------------------
# Request body models
# ---------------------------------------------------------------------------

class WindowLabel(BaseModel):
    """One label assignment: which window got which label."""
    window_index:  int    # index into the pipeline run's windows list
    label:         str    # e.g. "rest", "left", "right", "blink"
    start_time_sec: float = 0.0  # approximate position in the original recording


class SaveLabelsRequest(BaseModel):
    """Body for POST /api/labels/save."""
    session_id: str
    run_id:     str
    labels:     list[WindowLabel]


# ---------------------------------------------------------------------------
# POST /api/labels/save
# ---------------------------------------------------------------------------

@router.post("/save")
async def save_labels(request: SaveLabelsRequest) -> JSONResponse:
    """
    Save a list of labeled windows for a session.

    The frontend sends only {window_index, label, start_time_sec} — lightweight.
    This route looks up the actual numpy window data from store.pipeline_runs
    and builds proper LabeledWindow objects before storing them.

    Why look up numpy data here instead of having the frontend send it?
      Sending numpy arrays as nested JSON lists over HTTP is slow and wasteful.
      The data is already in server memory in store.pipeline_runs. We just
      need the index to look it up.
    """
    # Validate session exists
    if request.session_id not in store.sessions:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Session '{request.session_id}' not found."),
        )

    # Validate pipeline run exists
    run = store.pipeline_runs.get(request.run_id)
    if run is None or not isinstance(run, dict):
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Pipeline run '{request.run_id}' not found. Run the pipeline first."),
        )

    # Validate all window indices are within range
    n_windows = run["n_windows"]
    for lbl in request.labels:
        if lbl.window_index < 0 or lbl.window_index >= n_windows:
            return JSONResponse(
                status_code=400,
                content=api_response(
                    False,
                    error=f"Window index {lbl.window_index} out of range (0-{n_windows - 1}).",
                ),
            )

    # Build LabeledWindow objects — one per label assignment
    labeled_windows: list[LabeledWindow] = []
    for lbl in request.labels:
        window_data = run["windows"][lbl.window_index]  # numpy array (n_channels, n_samples)
        labeled_windows.append(LabeledWindow(
            window_id=     str(uuid.uuid4()),
            data=          window_data,
            label=         lbl.label.strip(),
            start_time_sec=lbl.start_time_sec,
            session_id=    request.session_id,
        ))

    # Store in session_id → list[LabeledWindow] dict.
    # Overwrite any previous labels for this session — the UI always sends the
    # full current set when the user hits "Save".
    store.labeled_data[request.session_id] = labeled_windows

    # Build a summary of label counts for the response (useful for the UI to confirm)
    label_counts: dict[str, int] = {}
    for lw in labeled_windows:
        label_counts[lw.label] = label_counts.get(lw.label, 0) + 1

    logger.info(
        "Saved %d labels for session %s: %s",
        len(labeled_windows), request.session_id, label_counts,
    )

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "session_id":   request.session_id,
            "total_labeled": len(labeled_windows),
            "label_counts": label_counts,
        }),
    )


# ---------------------------------------------------------------------------
# GET /api/labels/{session_id}
# ---------------------------------------------------------------------------

@router.get("/{session_id}")
async def get_labels(session_id: str) -> JSONResponse:
    """
    Return all saved labels for a session (metadata only — no numpy data).
    Returns an empty list if no labels have been saved yet.
    """
    labeled_windows = store.labeled_data.get(session_id, [])
    summaries = [labeled_window_summary(lw) for lw in labeled_windows]

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "session_id":    session_id,
            "total_labeled": len(summaries),
            "labels":        summaries,
        }),
    )


# ---------------------------------------------------------------------------
# DELETE /api/labels/{session_id}
# ---------------------------------------------------------------------------

@router.delete("/{session_id}")
async def delete_labels(session_id: str) -> JSONResponse:
    """
    Clear all saved labels for a session.
    Used when the user wants to start labeling over from scratch.
    """
    if session_id not in store.labeled_data:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"No labels found for session '{session_id}'."),
        )

    count = len(store.labeled_data[session_id])
    del store.labeled_data[session_id]

    logger.info("Deleted %d labels for session %s", count, session_id)

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "session_id":   session_id,
            "deleted_count": count,
        }),
    )
