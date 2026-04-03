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

import shutil
import tempfile

from fastapi import APIRouter, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.models import (
    ACTIVATION_OPTIONS, LAYER_TYPES_SCHEMA,
    estimate_output_shape, validate_model_config,
)
import os

import torch

from core.trainer import run_training
from core.classical_ml import run_classical_training, SUPPORTED_ALGORITHMS

EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exports")
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
    session_id:           str
    training_config:      dict        # ModelConfig JSON dict from CLAUDE.md
    run_id:               str | None = None
    model_name:           str        = ""
    augmentation_config:  dict | None = None  # Optional augmentation settings


class RenameModelRequest(BaseModel):
    """Body for PATCH /api/train/{job_id}/name."""
    model_name: str


class FinetuneRequest(BaseModel):
    """Body for POST /api/train/finetune."""
    base_job_id:          str           # existing trained model to start from
    session_id:           str
    run_id:               str | None = None
    model_name:           str        = ""
    augmentation_config:  dict | None = None


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
        # Classical ML jobs store final_val_acc directly; DL jobs have it in history
        final_acc = job.get("final_val_acc") or final.get("val_acc")
        jobs.append({
            "job_id":         job_id,
            "session_id":     job.get("session_id"),
            "model_name":     job.get("model_name", ""),
            "status":         job.get("status"),
            "started_at":     job.get("started_at"),
            "final_val_acc":  final_acc,
            "epochs_run":     len(history),
            "classes":        job.get("class_names", []),
            "model_path":     job.get("model_path"),
            "ml_type":        job.get("ml_type"),       # "classical" | None
            "algorithm":      job.get("algorithm"),     # classical ML only
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
        "job_id":         job_id,
        "session_id":     request.session_id,
        "run_id":         request.run_id,
        "model_name":     request.model_name,
        "model_config":   model_config,
        "status":         "running",
        "current_epoch":  0,
        "updates":        [],    # list of progress dicts appended by trainer.py
        "history":        [],    # copy of per-epoch metrics
        "model_path":     None,
        "started_at":     datetime.now(timezone.utc).isoformat(),
        "finetuned_from": None,  # None = training from scratch
    }

    # Start training in a daemon thread.
    # daemon=True means the thread will not block the process from exiting.
    # Pass run_id so the trainer looks up labeled_data by run_id (not session_id).
    # labeled_data is keyed by run_id — looking up by session_id always returns [].
    thread = threading.Thread(
        target=run_training,
        args=(job_id, request.session_id, model_config),
        kwargs={
            "run_id":              request.run_id,
            "augmentation_config": request.augmentation_config,
        },
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

    # Classical ML jobs don't have model_config or epoch history
    model_config  = job.get("model_config") or {}
    total_epochs  = model_config.get("epochs", 0) if model_config else 0
    latest        = job.get("history", [])
    latest_metric = latest[-1] if latest else {}

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "job_id":        job_id,
            "status":        job["status"],
            "current_epoch": job.get("current_epoch", 0),
            "total_epochs":  total_epochs,
            "latest_metrics": latest_metric,
            "started_at":    job.get("started_at"),
            "algorithm":     job.get("algorithm"),       # classical ML only
            "ml_type":       job.get("ml_type"),         # "classical" | None
            "cv_scores":     job.get("cv_scores", []),   # classical ML only
            "updates":       job.get("updates", []),     # for classical ML done check
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
            "history":          job.get("history", []),
            "final_metrics":    done_update,
            "model_path":       job.get("model_path"),
            "model_config":     job.get("model_config"),  # None for classical ML
            "ml_type":          job.get("ml_type"),       # "classical" | None
            "algorithm":        job.get("algorithm"),     # classical ML only
            "cv_scores":        job.get("cv_scores", []), # classical ML only
            "final_val_acc":    done_update.get("final_val_acc") if done_update else None,
            "confusion_matrix": done_update.get("confusion_matrix") if done_update else None,
            "roc_data":         done_update.get("roc_data") if done_update else None,
            "class_names":      done_update.get("class_names") if done_update else job.get("class_names", []),
            # Unsupervised ML: inertia (KMeans) or log-likelihood (GMM); None for supervised
            "fit_metric":       done_update.get("fit_metric") if done_update else None,
            "fit_metric_name":  done_update.get("fit_metric_name") if done_update else None,
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
# DELETE /api/train/{job_id} — delete a training job and its .pt checkpoint
# ---------------------------------------------------------------------------

@router.delete("/{job_id}")
async def delete_model(job_id: str) -> JSONResponse:
    """
    Remove a training job from the in-memory store and delete its .pt file.

    Used by the Manage Models UI so users can clean up old or unwanted models.
    Running jobs cannot be deleted — stop them first.
    """
    job = store.training_jobs.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Training job '{job_id}' not found."),
        )

    if job.get("status") == "running":
        return JSONResponse(
            status_code=400,
            content=api_response(False, error="Cannot delete a running training job. Wait for it to finish."),
        )

    # Remove the .pt checkpoint file if it exists
    model_path = os.path.join(EXPORTS_DIR, f"{job_id}.pt")
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
            logger.info("Deleted checkpoint: %s", model_path)
        except OSError as exc:
            logger.warning("Could not delete checkpoint file %s: %s", model_path, exc)

    # Remove from in-memory store
    del store.training_jobs[job_id]
    logger.info("Deleted training job %s", job_id)

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={"deleted_job_id": job_id}),
    )


# ---------------------------------------------------------------------------
# POST /api/train/finetune — start a fine-tuning job
# ---------------------------------------------------------------------------

@router.post("/finetune")
async def start_finetune(request: FinetuneRequest) -> JSONResponse:
    """
    Start training from an existing model's weights instead of random init.

    Loads the base model checkpoint, validates that the current training data
    has matching classes, then launches a normal training job with the base
    weights pre-loaded. Returns a new job_id — the original model is unchanged.
    """
    # Validate base model exists
    base_path = os.path.join(EXPORTS_DIR, f"{request.base_job_id}.pt")
    if not os.path.exists(base_path):
        return JSONResponse(
            status_code=404,
            content=api_response(
                False,
                error=f"Base model '{request.base_job_id}' not found. Train a model first.",
            ),
        )

    # Load the base checkpoint to get its config + classes for validation
    try:
        base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"Could not load base model: {exc}"),
        )

    base_classes = base_ckpt.get("class_names", [])

    # Validate session exists
    if request.session_id not in store.sessions:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Session '{request.session_id}' not found."),
        )

    # Gather labeled windows for class-list validation
    if request.run_id:
        labeled_windows = store.labeled_data.get(request.run_id, [])
    else:
        labeled_windows = [
            lw
            for lw_list in store.labeled_data.values()
            for lw in lw_list
            if lw.session_id == request.session_id
        ]

    if not labeled_windows:
        return JSONResponse(
            status_code=400,
            content=api_response(
                False,
                error="No labeled windows found for this session/run.",
            ),
        )

    # Exclude 'unknown' — same filter applied when training from scratch
    current_classes = sorted(set(lw.label for lw in labeled_windows if lw.label != 'unknown'))

    # Check class compatibility up front so the frontend gets instant feedback
    # instead of waiting for the thread to start and fail.
    if sorted(base_classes) != sorted(current_classes):
        return JSONResponse(
            status_code=400,
            content=api_response(
                False,
                error=(
                    f"Class mismatch: base model has {base_classes} "
                    f"but current labels have {current_classes}. "
                    "Fine-tuning requires identical class lists."
                ),
            ),
        )

    # Re-use the base model's architecture config exactly
    model_config = dict(base_ckpt.get("model_config", {}))
    model_config.setdefault("learning_rate", 0.001)
    model_config.setdefault("epochs",        20)
    model_config.setdefault("batch_size",    32)
    model_config["classes"] = base_classes

    job_id = str(uuid.uuid4())
    store.training_jobs[job_id] = {
        "job_id":           job_id,
        "session_id":       request.session_id,
        "run_id":           request.run_id,
        "model_name":       request.model_name,
        "model_config":     model_config,
        "status":           "running",
        "current_epoch":    0,
        "updates":          [],
        "history":          [],
        "model_path":       None,
        "started_at":       datetime.now(timezone.utc).isoformat(),
        "finetuned_from":   request.base_job_id,
    }

    thread = threading.Thread(
        target=run_training,
        args=(job_id, request.session_id, model_config),
        kwargs={
            "run_id":               request.run_id,
            "finetune_from_job_id": request.base_job_id,
            "augmentation_config":  request.augmentation_config,
        },
        daemon=True,
        name=f"finetune-{job_id[:8]}",
    )
    thread.start()

    logger.info("Fine-tune job %s started from base %s", job_id, request.base_job_id)

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={
            "job_id":       job_id,
            "status":       "running",
            "model_config": model_config,
            "base_job_id":  request.base_job_id,
        }),
    )


# ---------------------------------------------------------------------------
# POST /api/train/import — import an externally trained .pt model file
# ---------------------------------------------------------------------------

@router.post("/import")
async def import_model(
    file: UploadFile,
    model_name: str = Form("imported"),
) -> JSONResponse:
    """
    Import an external .pt checkpoint file and register it as a training job.

    This lets users bring in a model they trained outside of Neutrol (or in a
    previous session) and use it directly for inference, export, or fine-tuning.

    Required checkpoint keys: model_state_dict, class_names.
    Optional (filled with defaults if missing): model_config, n_channels, n_samples.

    The file is copied to exports/{job_id}.pt and a 'done' training_jobs entry
    is created so the model appears in dropdowns immediately.
    """
    job_id = str(uuid.uuid4())

    # Write the upload to a temp file so we can torch.load it for validation
    suffix = ".pt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        try:
            checkpoint = torch.load(tmp_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content=api_response(False, error=f"Could not load .pt file: {exc}"),
            )

        # Validate minimum required keys
        required = ["model_state_dict", "class_names"]
        missing  = [k for k in required if k not in checkpoint]
        if missing:
            return JSONResponse(
                status_code=400,
                content=api_response(
                    False,
                    error=(
                        f"Checkpoint is missing required keys: {missing}. "
                        "A valid Neutrol checkpoint must contain 'model_state_dict' "
                        "and 'class_names'."
                    ),
                ),
            )

        class_names  = checkpoint["class_names"]
        ckpt_model_name = checkpoint.get("model_name", model_name) or model_name

        # Copy validated checkpoint to exports directory
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        dest_path = os.path.join(EXPORTS_DIR, f"{job_id}.pt")
        shutil.copy2(tmp_path, dest_path)

        # Build a training_jobs entry that matches what a native job produces.
        # Status 'done' makes it immediately available in all dropdowns.
        store.training_jobs[job_id] = {
            "job_id":         job_id,
            "session_id":     None,
            "run_id":         None,
            "model_name":     ckpt_model_name,
            "model_config":   checkpoint.get("model_config", {}),
            "class_names":    class_names,
            "n_channels":     checkpoint.get("n_channels"),
            "n_samples":      checkpoint.get("n_samples"),
            "status":         "done",
            "current_epoch":  checkpoint.get("epoch", 0),
            "updates":        [],
            "history":        [],
            "model_path":     dest_path,
            "started_at":     datetime.now(timezone.utc).isoformat(),
            "finetuned_from": checkpoint.get("finetuned_from"),
            "imported":       True,   # flag so UI can show "imported" badge
        }

        logger.info(
            "Imported model '%s' as job %s (classes: %s)",
            ckpt_model_name, job_id, class_names,
        )

        return JSONResponse(
            status_code=200,
            content=api_response(True, data={
                "job_id":      job_id,
                "model_name":  ckpt_model_name,
                "class_names": class_names,
                "n_channels":  checkpoint.get("n_channels"),
                "n_samples":   checkpoint.get("n_samples"),
            }),
        )

    finally:
        # Always clean up the temp file, even if an error occurred above
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# GET /api/train/history/{model_name} — all runs for a model name
# ---------------------------------------------------------------------------

@router.get("/history/{model_name}")
async def get_model_history(model_name: str) -> JSONResponse:
    """
    Return all training runs that share the given model_name, sorted by date.

    Scans store.training_jobs (in-memory) for matching entries.
    Each entry returns enough metadata to populate the history table:
    job_id, date, session_id, epochs, final val accuracy, finetuned_from.
    """
    runs = []
    for job_id, job in store.training_jobs.items():
        # Match on model_name — case-insensitive strip for robustness
        if job.get("model_name", "").strip().lower() != model_name.strip().lower():
            continue
        if job.get("status") != "done":
            continue

        history = job.get("history", [])
        final   = history[-1] if history else {}

        runs.append({
            "job_id":         job_id,
            "session_id":     job.get("session_id"),
            "started_at":     job.get("started_at"),
            "epochs_run":     len(history),
            "final_val_acc":  final.get("val_acc"),
            "final_train_acc": final.get("train_acc"),
            "finetuned_from": job.get("finetuned_from"),
            "model_path":     job.get("model_path"),
        })

    # Sort oldest → newest so a chart shows progression forward in time
    runs.sort(key=lambda r: r["started_at"] or "")

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={"model_name": model_name, "runs": runs}),
    )


# ---------------------------------------------------------------------------
# GET /api/train/classical/algorithms
# POST /api/train/classical/start
# ---------------------------------------------------------------------------

class ClassicalMLRequest(BaseModel):
    """Body for POST /api/train/classical/start."""
    session_id:  str
    run_id:      str | None = None
    algorithm:   str               # 'LDA', 'SVM', 'RandomForest', 'GBM'
    hyperparams: dict = {}         # algorithm-specific hyperparameter overrides
    model_name:  str  = ""


@router.get("/classical/algorithms")
async def list_classical_algorithms() -> JSONResponse:
    """
    Return the list of supported classical ML algorithms with descriptions.

    Used by the frontend to populate the algorithm dropdown.
    """
    algorithms = [
        {
            "id":          alg_id,
            "name":        name,
            "description": _CLASSICAL_DESCRIPTIONS.get(alg_id, ""),
            "hyperparams": _CLASSICAL_HYPERPARAMS.get(alg_id, []),
        }
        for alg_id, name in SUPPORTED_ALGORITHMS.items()
    ]
    return JSONResponse(status_code=200, content=api_response(True, data=algorithms))


@router.post("/classical/start")
async def start_classical_training(request: ClassicalMLRequest) -> JSONResponse:
    """
    Start a classical ML training job in a background thread.

    Returns a job_id immediately. The caller can then poll
    GET /api/train/{job_id}/status to check progress and get results.
    Classical ML is fast (seconds), so the job will typically be done
    before the frontend even polls once.
    """
    if request.algorithm not in SUPPORTED_ALGORITHMS:
        return JSONResponse(
            status_code=400,
            content=api_response(
                False,
                error=f"Unknown algorithm '{request.algorithm}'. "
                      f"Supported: {list(SUPPORTED_ALGORITHMS.keys())}",
            ),
        )

    job_id = str(uuid.uuid4())
    now    = datetime.now(timezone.utc).isoformat()

    store.training_jobs[job_id] = {
        "job_id":      job_id,
        "status":      "running",
        "started_at":  now,
        "session_id":  request.session_id,
        "run_id":      request.run_id,
        "model_name":  request.model_name,
        "algorithm":   request.algorithm,
        "ml_type":     "classical",
        "updates":     [],
        "class_names": [],
        "n_classes":   0,
        "history":     [],          # empty — classical ML has no epoch history
        "current_epoch": 0,
    }

    thread = threading.Thread(
        target=run_classical_training,
        args=(
            job_id,
            request.session_id,
            request.run_id,
            request.algorithm,
            request.hyperparams,
            request.model_name,
        ),
        daemon=True,
    )
    thread.start()

    logger.info(
        "Classical ML job %s started: algorithm=%s session=%s run=%s",
        job_id, request.algorithm, request.session_id, request.run_id,
    )

    return JSONResponse(
        status_code=200,
        content=api_response(True, data={"job_id": job_id}),
    )


# Per-algorithm human-readable descriptions shown in the UI
_CLASSICAL_DESCRIPTIONS: dict[str, str] = {
    "LDA": (
        "Fast, no hyperparameters. Best for P300, SSVEP, and simple paradigms. "
        "Often outperforms deep learning on small datasets (<200 windows)."
    ),
    "SVM": (
        "Support Vector Machine. Handles non-linear boundaries via the kernel trick. "
        "Good general-purpose choice. Set kernel='rbf' for most EEG tasks."
    ),
    "RandomForest": (
        "Ensemble of decision trees. Robust to noise and feature redundancy. "
        "Works well across frequency bands without careful preprocessing."
    ),
    "GBM": (
        "Gradient Boosting Machine. Often the highest accuracy of classical methods. "
        "Slower to train but worth it for final deployment models."
    ),
}

# Hyperparameter definitions for each algorithm — used by the frontend to render inputs
_CLASSICAL_HYPERPARAMS: dict[str, list[dict]] = {
    "LDA": [
        {"name": "solver",     "type": "select", "default": "svd",
         "options": ["svd", "lsqr", "eigen"],
         "description": "svd: no matrix inversion (stable). lsqr/eigen: support shrinkage."},
        {"name": "shrinkage",  "type": "float",  "default": 0.0,
         "description": "Regularisation (0=none, 'auto'=Ledoit-Wolf). Only for lsqr/eigen."},
    ],
    "SVM": [
        {"name": "kernel", "type": "select", "default": "rbf",
         "options": ["rbf", "linear", "poly", "sigmoid"],
         "description": "rbf: handles non-linear boundaries (recommended). linear: fast."},
        {"name": "C",      "type": "float",  "default": 1.0,
         "description": "Regularisation strength. Higher C = less regularisation."},
        {"name": "gamma",  "type": "select", "default": "scale",
         "options": ["scale", "auto"],
         "description": "Kernel coefficient. 'scale'=1/(n_features·X.var())."},
    ],
    "RandomForest": [
        {"name": "n_estimators",     "type": "int",   "default": 100,
         "description": "Number of trees. More = better but slower."},
        {"name": "max_depth",        "type": "int",   "default": 0,
         "description": "Maximum tree depth (0 = unlimited)."},
        {"name": "min_samples_split","type": "int",   "default": 2,
         "description": "Minimum samples to split a node."},
    ],
    "GBM": [
        {"name": "n_estimators",  "type": "int",   "default": 100,
         "description": "Number of boosting rounds."},
        {"name": "learning_rate", "type": "float", "default": 0.1,
         "description": "Shrinkage per round. Lower = slower but more robust."},
        {"name": "max_depth",     "type": "int",   "default": 3,
         "description": "Maximum depth of each tree. 3-5 is typical."},
    ],
}


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
