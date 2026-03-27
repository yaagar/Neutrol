"""
train_routes.py — HTTP + WebSocket endpoints for model training.

Training flow:
  1. Frontend calls POST /api/train/start with a ModelConfig dict → gets job_id
  2. A daemon thread starts running trainer.run_training(job_id, ...)
  3. Frontend connects WebSocket to /api/train/ws/{job_id}/progress
  4. WebSocket handler polls store.training_jobs[job_id]["updates"] and forwards
     new entries to the browser as JSON messages after each epoch
  5. When training finishes, a {"type": "done"} message is sent and the socket closes

Registered in main.py with prefix /api/train.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.models import (
    ACTIVATION_OPTIONS, LAYER_TYPES_SCHEMA,
    estimate_output_shape, validate_model_config,
)
from core.trainer import run_training
from sessions.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter()
store = SessionStore()

# How often (seconds) the WebSocket handler polls for new training updates.
# 0.2s gives a smooth real-time feel without hammering the event loop.
WS_POLL_INTERVAL = 0.2

# Maximum seconds to keep a WebSocket open waiting for training to finish.
# 30 minutes is generous for any reasonable dataset size on CPU.
WS_TIMEOUT_SEC = 1800


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def api_response(success: bool, data=None, error: str | None = None) -> dict:
    """Standard APIResponse shape from CLAUDE.md."""
    return {"success": success, "data": data, "error": error}


# ---------------------------------------------------------------------------
# Request body model
# ---------------------------------------------------------------------------

class StartTrainingRequest(BaseModel):
    """Body for POST /api/train/start."""
    session_id:      str
    training_config: dict        # ModelConfig JSON dict from CLAUDE.md
    run_id:          str | None = None  # Which pipeline run to train on (keyed in labeled_data)
    model_name:      str = ""    # Optional human-readable name for this trained model


class RenameModelRequest(BaseModel):
    """Body for PATCH /api/train/{job_id}/name."""
    model_name: str


class ValidateConfigRequest(BaseModel):
    """Body for POST /api/train/validate-config."""
    layers:         list   # list of layer config dicts
    input_channels: int
    input_samples:  int
    classes:        list[str]


# ---------------------------------------------------------------------------
# GET /api/train/jobs — list all training jobs
# ---------------------------------------------------------------------------

@router.get("/jobs")
async def list_jobs() -> JSONResponse:
    """
    Return a summary of all training jobs (running, done, or failed).
    Used by the Inference page to populate the model selector dropdown.
    """
    jobs = []
    for job_id, job in store.training_jobs.items():
        history = job.get("history", [])
        final   = history[-1] if history else {}
        jobs.append({
            "job_id":         job_id,
            "session_id":     job.get("session_id"),
            "model_name":     job.get("model_name", ""),
            "status":         job.get("status"),
            "started_at":     job.get("started_at"),
            "final_val_acc":  final.get("val_acc"),
            "epochs_run":     len(history),
            "classes":        job.get("class_names", []),
            "model_path":     job.get("model_path"),
        })
    # Most recent first
    jobs.sort(key=lambda j: j["started_at"] or "", reverse=True)
    return JSONResponse(status_code=200, content=api_response(True, data={"jobs": jobs}))


# ---------------------------------------------------------------------------
# GET /api/train/layer-types
# ---------------------------------------------------------------------------

@router.get("/layer-types")
async def get_layer_types() -> JSONResponse:
    """
    Return all supported layer types with their parameters, types, defaults,
    and plain-English descriptions.

    Used by the frontend 'Add Layer' dropdown and the parameter editor for
    each layer card.
    """
    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "layer_types":        LAYER_TYPES_SCHEMA,
            "activation_options": ACTIVATION_OPTIONS,
        }),
    )


# ---------------------------------------------------------------------------
# POST /api/train/validate-config
# ---------------------------------------------------------------------------

@router.post("/validate-config")
async def validate_config(request: ValidateConfigRequest) -> JSONResponse:
    """
    Validate a model architecture and compute the tensor shape after each layer.

    Returns:
      - valid: bool
      - errors: list of human-readable error strings (empty if valid)
      - layer_shapes: list of tensor shapes (one per layer, None if shape estimation failed)
    """
    config = {
        "layers":         request.layers,
        "input_channels": request.input_channels,
        "input_samples":  request.input_samples,
        "classes":        request.classes,
    }

    errors = validate_model_config(config)

    # Only run shape estimation if there are no structural errors — a broken
    # config would produce misleading or crashing shape estimates
    if not errors:
        try:
            shapes = estimate_output_shape(config)
        except Exception as exc:
            logger.error("Shape estimation error: %s", exc)
            shapes = [None] * len(request.layers)
    else:
        shapes = [None] * len(request.layers)

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "valid":        len(errors) == 0,
            "errors":       errors,
            "layer_shapes": shapes,
        }),
    )


# ---------------------------------------------------------------------------
# POST /api/train/start
# ---------------------------------------------------------------------------

@router.post("/start")
async def start_training(request: StartTrainingRequest) -> JSONResponse:
    """
    Validate the request, initialise a training job entry, and start training
    in a background daemon thread.

    Returns the job_id immediately — training continues in the background.
    The frontend uses the job_id to connect to the WebSocket for live updates.
    """
    # Validate session exists
    if request.session_id not in store.sessions:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Session '{request.session_id}' not found."),
        )

    # Validate labeled data exists.
    # If run_id is provided, look up labeled_data by run_id (preferred — specific run).
    # Otherwise aggregate all labeled windows for the session (fallback for older clients).
    if request.run_id:
        labeled_windows = store.labeled_data.get(request.run_id, [])
    else:
        labeled_windows = []
        for lw_list in store.labeled_data.values():
            for lw in lw_list:
                if lw.session_id == request.session_id:
                    labeled_windows.append(lw)

    if len(labeled_windows) == 0:
        return JSONResponse(
            status_code=400,
            content=api_response(
                False,
                error="No labeled windows found for this session. "
                      "Go to the Label page and label some windows first.",
            ),
        )

    # Auto-detect classes from labels if not provided in training_config
    model_config = dict(request.training_config)
    if "classes" not in model_config or not model_config["classes"]:
        # Deduplicate and sort so class order is deterministic
        model_config["classes"] = sorted(set(lw.label for lw in labeled_windows))

    # Set defaults for any missing config fields
    model_config.setdefault("model_type",    "CNN")
    model_config.setdefault("learning_rate", 0.001)
    model_config.setdefault("epochs",        50)
    model_config.setdefault("batch_size",    32)

    # Get input shape from the first labeled window
    first_window = labeled_windows[0]
    n_channels, n_samples = first_window.data.shape
    model_config.setdefault("input_channels", n_channels)
    model_config.setdefault("input_samples",  n_samples)

    # Create the job entry before starting the thread so the WebSocket
    # handler can find it immediately (even before the first epoch completes)
    job_id = str(uuid.uuid4())
    store.training_jobs[job_id] = {
        "job_id":        job_id,
        "session_id":    request.session_id,
        "run_id":        request.run_id,
        "model_name":    request.model_name,
        "model_config":  model_config,
        "status":        "running",
        "current_epoch": 0,
        "updates":       [],    # list of progress dicts appended by trainer.py
        "history":       [],    # copy of per-epoch metrics
        "model_path":    None,
        "started_at":    datetime.now(timezone.utc).isoformat(),
    }

    # Start training in a daemon thread.
    # daemon=True means the thread will not block the process from exiting.
    thread = threading.Thread(
        target=run_training,
        args=(job_id, request.session_id, model_config),
        daemon=True,
        name=f"train-{job_id[:8]}",
    )
    thread.start()

    logger.info(
        "Training job %s started for session %s: %s",
        job_id, request.session_id, model_config,
    )

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "job_id":      job_id,
            "status":      "running",
            "model_config": model_config,
        }),
    )


# ---------------------------------------------------------------------------
# GET /api/train/{job_id}/status
# ---------------------------------------------------------------------------

@router.get("/{job_id}/status")
async def get_training_status(job_id: str) -> JSONResponse:
    """
    Polling fallback for training progress.
    Returns the current epoch, status, and latest metrics.
    The frontend primarily uses the WebSocket, but this is useful for
    reconnection after a page refresh.
    """
    job = store.training_jobs.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Training job '{job_id}' not found."),
        )

    total_epochs = job["model_config"].get("epochs", 50)
    latest = job["history"][-1] if job["history"] else {}

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "job_id":        job_id,
            "status":        job["status"],
            "current_epoch": job["current_epoch"],
            "total_epochs":  total_epochs,
            "latest_metrics": latest,
            "started_at":    job["started_at"],
        }),
    )


# ---------------------------------------------------------------------------
# GET /api/train/{job_id}/results
# ---------------------------------------------------------------------------

@router.get("/{job_id}/results")
async def get_training_results(job_id: str) -> JSONResponse:
    """
    Return the full training history and final results once training is complete.
    Returns 400 if training is still in progress.
    """
    job = store.training_jobs.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Training job '{job_id}' not found."),
        )

    if job["status"] == "running":
        return JSONResponse(
            status_code=400,
            content=api_response(False, error="Training is still in progress."),
        )

    # Find the "done" update which has the final metrics and confusion matrix
    done_update = next(
        (u for u in job["updates"] if u.get("type") == "done"),
        None,
    )

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "job_id":           job_id,
            "status":           job["status"],
            "history":          job["history"],
            "final_metrics":    done_update,
            "model_path":       job.get("model_path"),
            "model_config":     job["model_config"],
        }),
    )


# ---------------------------------------------------------------------------
# PATCH /api/train/{job_id}/name — rename a trained model
# ---------------------------------------------------------------------------

@router.patch("/{job_id}/name")
async def rename_model(job_id: str, request: RenameModelRequest) -> JSONResponse:
    """
    Update the human-readable name for a training job / saved model.

    This lets the user give their models memorable names (e.g. "Motor Imagery v2")
    instead of the opaque UUID that appears in the Inference page dropdown.
    """
    job = store.training_jobs.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Training job '{job_id}' not found."),
        )

    job["model_name"] = request.model_name.strip()
    logger.info("Renamed model %s → '%s'", job_id, job["model_name"])

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={"job_id": job_id, "model_name": job["model_name"]}),
    )


# ---------------------------------------------------------------------------
# WebSocket /api/train/ws/{job_id}/progress
# ---------------------------------------------------------------------------

@router.websocket("/ws/{job_id}/progress")
async def training_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """
    Stream training progress updates to the frontend as JSON messages.

    How it works:
      - Training runs in a background thread (started by POST /start)
      - That thread appends dicts to store.training_jobs[job_id]["updates"]
      - This WebSocket handler polls that list every WS_POLL_INTERVAL seconds
      - Any new entries are forwarded to the browser immediately
      - When a {"type": "done"} or {"type": "error"} message is seen, the socket closes

    Message types the frontend receives:
      {"type": "progress", "epoch": int, "total_epochs": int,
       "train_loss": float, "val_loss": float, "train_acc": float, "val_acc": float}
      {"type": "done", "final_val_acc": float, "confusion_matrix": [...], ...}
      {"type": "error", "message": str}
    """
    await websocket.accept()
    logger.info("WebSocket connected for training job %s", job_id)

    # Check the job exists before entering the loop
    if job_id not in store.training_jobs:
        await websocket.send_json({"type": "error", "message": f"Job '{job_id}' not found."})
        await websocket.close()
        return

    last_sent_idx = 0    # index into job["updates"] — tracks what we've already sent
    elapsed = 0.0

    try:
        while elapsed < WS_TIMEOUT_SEC:
            job = store.training_jobs.get(job_id)
            if not job:
                break

            updates = job.get("updates", [])

            # Forward any new updates that appeared since our last check
            while last_sent_idx < len(updates):
                await websocket.send_json(updates[last_sent_idx])
                last_sent_idx += 1

            # If training finished (or failed), we're done
            if job.get("status") in ("done", "failed"):
                break

            # Wait before polling again — yields control to the event loop
            # so other requests can be handled while we wait
            await asyncio.sleep(WS_POLL_INTERVAL)
            elapsed += WS_POLL_INTERVAL

        if elapsed >= WS_TIMEOUT_SEC:
            await websocket.send_json({"type": "error", "message": "Training timed out."})

    except WebSocketDisconnect:
        # Browser closed the tab or navigated away — normal, not an error
        logger.info("WebSocket disconnected by client for job %s", job_id)
    except Exception as exc:
        logger.error("WebSocket error for job %s: %s", job_id, exc)
    finally:
        logger.info("WebSocket closed for training job %s", job_id)
