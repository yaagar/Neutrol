"""
pipeline_routes.py — HTTP endpoints for running and managing preprocessing pipelines.

This file is the thin API layer for pipeline operations.
All actual signal processing lives in core/preprocessing.py.

Registered in main.py with prefix /api/pipeline.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.preprocessing import STEP_REGISTRY, Pipeline
from sessions.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter()
store = SessionStore()  # same singleton used everywhere

# Directory where pipeline config JSON files are saved
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")


# ---------------------------------------------------------------------------
# Helper: standard APIResponse
# ---------------------------------------------------------------------------

def api_response(success: bool, data=None, error: str | None = None) -> dict:
    """Standard APIResponse shape from CLAUDE.md."""
    return {"success": success, "data": data, "error": error}


# ---------------------------------------------------------------------------
# Request body models (Pydantic validates the JSON body automatically)
# ---------------------------------------------------------------------------

class RunPipelineRequest(BaseModel):
    """Body for POST /api/pipeline/run."""
    session_id: str
    pipeline_config: dict   # PipelineConfig JSON dict from CLAUDE.md
    name: str = ""          # Optional human-readable name for this run


class SavePipelineRequest(BaseModel):
    """Body for POST /api/pipeline/save."""
    pipeline_id: str | None = None
    session_id: str | None = None
    sample_rate: int | None = None
    steps: list[dict] = []


# ---------------------------------------------------------------------------
# POST /api/pipeline/run
# ---------------------------------------------------------------------------

@router.post("/run")
async def run_pipeline(request: RunPipelineRequest) -> JSONResponse:
    """
    Run a preprocessing pipeline on a loaded session and store the windows.

    Storage format changed from Session 1's tuple to a dict so we can
    list runs and serve individual windows from GET /api/pipeline/runs/*.
    """
    session = store.sessions.get(request.session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=api_response(
                False,
                error=f"Session '{request.session_id}' not found. Upload a file first.",
            ),
        )

    try:
        pipeline = Pipeline.from_dict(request.pipeline_config)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content=api_response(False, error=f"Invalid pipeline config: {exc}"),
        )

    try:
        windows = pipeline.run(session)
    except Exception as exc:
        logger.error("Pipeline run failed for session %s: %s", request.session_id, exc)
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"Pipeline failed: {exc}"),
        )

    run_id = str(uuid.uuid4())
    window_shape = list(windows[0].shape) if windows else [0, 0]
    steps_applied = [step.to_dict()["step"] for step in pipeline.steps]

    # Determine signal_offset_sec from any TruncateSignal step.
    # This offset is passed to apply_events_to_windows so event markers
    # (in original-recording time) align with windows (in truncated-signal time).
    signal_offset_sec = 0.0
    for step_cfg in request.pipeline_config.get("steps", []):
        if step_cfg.get("step") == "truncate":
            signal_offset_sec = float(step_cfg.get("start_sec", 0.0))
            break

    # Store as a dict (not a tuple) so /runs and /runs/{id}/window/{i} can read metadata
    # without loading all the numpy arrays into the response.
    store.pipeline_runs[run_id] = {
        "session_id":       request.session_id,
        "name":             request.name,
        "windows":          windows,           # list[np.ndarray] — never sent over API
        "n_windows":        len(windows),
        "window_shape":     window_shape,
        "steps_applied":    steps_applied,
        "signal_offset_sec": signal_offset_sec,
        "created_at":       datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Pipeline run %s ('%s'): %d windows of shape %s for session %s",
        run_id, request.name, len(windows), window_shape, request.session_id,
    )

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "run_id":            run_id,
            "name":              request.name,
            "n_windows":         len(windows),
            "window_shape":      window_shape,
            "steps_applied":     steps_applied,
            "signal_offset_sec": signal_offset_sec,
        }),
    )


# ---------------------------------------------------------------------------
# GET /api/pipeline/runs
# ---------------------------------------------------------------------------

@router.get("/runs")
async def list_runs() -> JSONResponse:
    """
    Return metadata for all pipeline runs currently in memory.
    Optionally filter by session_id via query param: /api/pipeline/runs?session_id=abc
    """
    from fastapi import Request
    results = []
    for run_id, run in store.pipeline_runs.items():
        # Guard: skip any legacy tuple entries (shouldn't exist after this session)
        if not isinstance(run, dict):
            continue
        results.append({
            "run_id":            run_id,
            "name":              run.get("name", ""),
            "session_id":        run["session_id"],
            "n_windows":         run["n_windows"],
            "window_shape":      run["window_shape"],
            "steps_applied":     run["steps_applied"],
            "signal_offset_sec": run.get("signal_offset_sec", 0.0),
            "created_at":        run["created_at"],
        })
    return JSONResponse(
        status_code=200,
        content=api_response(True, data=results),
    )


# ---------------------------------------------------------------------------
# GET /api/pipeline/runs/{run_id}/window/{window_index}
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/window/{window_index}")
async def get_window(run_id: str, window_index: int) -> JSONResponse:
    """
    Return a single window's signal data in Recharts-compatible format.

    This is what the LabelPage calls when navigating between windows.
    The window data (numpy array) is converted to a list of dicts here
    so React can feed it straight into WaveformChart.

    Why not send all windows at once?
      For a 10-minute recording at 256Hz with 50% overlap, that's ~1200 windows
      each with 256 × 8 = 2048 floats. That's ~10 MB of JSON — too slow to load.
      Fetching one window at a time keeps navigation instant.
    """
    run = store.pipeline_runs.get(run_id)
    if run is None or not isinstance(run, dict):
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Pipeline run '{run_id}' not found."),
        )

    n_windows = run["n_windows"]
    if window_index < 0 or window_index >= n_windows:
        return JSONResponse(
            status_code=400,
            content=api_response(
                False,
                error=f"Window index {window_index} out of range (0–{n_windows - 1}).",
            ),
        )

    # Get the numpy window: shape (n_channels, window_samples)
    window = run["windows"][window_index]
    n_channels, window_samples = window.shape

    # Look up channel names from the original session
    session = store.sessions.get(run["session_id"])
    channel_names = session.channel_names if session else [str(i) for i in range(n_channels)]
    sample_rate = session.sample_rate if session else 256

    # Build time axis — each window starts at t=0 for display purposes.
    # The absolute start time in the recording is stored separately.
    time_step = 1.0 / sample_rate
    points = []
    for i in range(window_samples):
        point = {"time": round(i * time_step, 4)}
        for ch_idx, ch_name in enumerate(channel_names):
            point[ch_name] = round(float(window[ch_idx, i]), 6)
        points.append(point)

    # Approximate start time in the original recording.
    # We use window_index * step where step = window_samples * (1 - overlap).
    # Since we don't store overlap here, we approximate with 50% overlap as default.
    # This is display-only — not used for training.
    approx_start_sec = round(window_index * window_samples * 0.5 / sample_rate, 3)

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "run_id":          run_id,
            "window_index":    window_index,
            "total_windows":   n_windows,
            "channel_names":   channel_names,
            "sample_rate":     sample_rate,
            "start_time_sec":  approx_start_sec,
            "points":          points,
        }),
    )


# ---------------------------------------------------------------------------
# GET /api/pipeline/steps
# ---------------------------------------------------------------------------

@router.get("/steps")
async def list_steps() -> JSONResponse:
    """
    Return all available preprocessing steps with their parameter metadata.
    The frontend uses this to build the pipeline config form dynamically.
    """
    steps_info = []

    for step_name, step_class in STEP_REGISTRY.items():
        sig = inspect.signature(step_class.__init__)
        params = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            if param.annotation is inspect.Parameter.empty:
                type_str = "any"
            else:
                type_str = getattr(param.annotation, "__name__", str(param.annotation))

            if param.default is inspect.Parameter.empty:
                default = None
                required = True
            else:
                default = param.default
                required = False

            params.append({
                "name":     param_name,
                "type":     type_str,
                "default":  default,
                "required": required,
            })

        steps_info.append({
            "step":        step_name,
            "class_name":  step_class.__name__,
            "description": (step_class.__doc__ or "").strip().split("\n")[0],
            "parameters":  params,
        })

    return JSONResponse(
        status_code=200,
        content=api_response(True, data=steps_info),
    )


# ---------------------------------------------------------------------------
# POST /api/pipeline/save
# ---------------------------------------------------------------------------

@router.post("/save")
async def save_pipeline(request: SavePipelineRequest) -> JSONResponse:
    """Save a pipeline configuration to a JSON file in the configs/ directory."""
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    pipeline_id = request.pipeline_id or str(uuid.uuid4())

    config_dict = {
        "pipeline_id": pipeline_id,
        "session_id":  request.session_id,
        "sample_rate": request.sample_rate,
        "steps":       request.steps,
        "saved_at":    datetime.now(timezone.utc).isoformat(),
    }

    file_path = os.path.join(CONFIGS_DIR, f"{pipeline_id}.json")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"Could not save pipeline config: {exc}"),
        )

    logger.info("Pipeline config saved: %s", file_path)

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={"pipeline_id": pipeline_id, "file_path": file_path}),
    )


# ---------------------------------------------------------------------------
# GET /api/pipeline/saved
# ---------------------------------------------------------------------------

@router.get("/saved")
async def list_saved_pipelines() -> JSONResponse:
    """Scan the configs/ directory and return metadata for all saved pipeline configs."""
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    results = []
    for filename in os.listdir(CONFIGS_DIR):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(CONFIGS_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            results.append({
                "filename":    filename,
                "pipeline_id": config.get("pipeline_id", filename.replace(".json", "")),
                "step_count":  len(config.get("steps", [])),
                "saved_at":    config.get("saved_at"),
            })
        except Exception as exc:
            logger.warning("Could not read config file %s: %s", filename, exc)

    return JSONResponse(
        status_code=200,
        content=api_response(True, data=results),
    )
