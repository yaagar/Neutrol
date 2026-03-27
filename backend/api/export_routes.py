"""
export_routes.py — /api/export/* endpoints.

Three download endpoints:
  GET /api/export/pipeline/{run_id}  → pipeline config as .json file download
  GET /api/export/model/{job_id}     → trained model weights as .pt file download
  GET /api/export/script/{job_id}    → auto-generated Python inference script (.py)

Each endpoint returns a FileResponse or a plain text response with the
appropriate Content-Disposition header so the browser triggers a download.

Why use run_id for pipeline export instead of pipeline_id?
  Users interact with pipeline RUNS, not saved pipeline configs. Runs already
  store the full pipeline_config dict (added in Session 8), so we can produce
  a JSON download without requiring the user to have explicitly saved the config.

Registered in main.py with prefix /api/export.
"""

from __future__ import annotations

import json
import logging
import os

import torch
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from core.export import generate_inference_script
from sessions.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter()
store  = SessionStore()

# Where trainer.py saves model weights
EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exports")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def api_response(success: bool, data=None, error: str | None = None) -> dict:
    """Standard APIResponse shape from CLAUDE.md."""
    return {"success": success, "data": data, "error": error}


# ---------------------------------------------------------------------------
# GET /api/export/pipeline/{run_id}
# ---------------------------------------------------------------------------

@router.get("/pipeline/{run_id}", response_model=None)
async def export_pipeline(run_id: str) -> JSONResponse:
    """
    Return the full pipeline config for a run as a JSON file download.

    The response uses Content-Disposition: attachment so the browser saves it
    as a file rather than displaying it inline.

    The exported JSON matches the PipelineConfig schema from CLAUDE.md so it
    can be loaded back into Neutrol's PreprocessPage in a future session.
    """
    run = store.pipeline_runs.get(run_id)
    if run is None or not isinstance(run, dict):
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Pipeline run '{run_id}' not found."),
        )

    # Build the exportable config dict
    config = run.get("pipeline_config", {})
    if not config:
        # Fallback: reconstruct from steps_applied names (no parameters)
        config = {"steps": [{"step": s} for s in run.get("steps_applied", [])]}

    export_dict = {
        "run_id":     run_id,
        "name":       run.get("name", ""),
        "session_id": run.get("session_id", ""),
        "created_at": run.get("created_at", ""),
        "steps":      config.get("steps", []),
    }

    filename = f"pipeline_{(run.get('name') or run_id[:8]).replace(' ', '_')}.json"

    return JSONResponse(
        status_code=200,
        content=export_dict,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /api/export/model/{job_id}
# ---------------------------------------------------------------------------

@router.get("/model/{job_id}", response_model=None)
async def export_model(job_id: str) -> FileResponse | JSONResponse:
    """
    Download the trained model weights (.pt file) for a completed training job.

    The .pt file is a PyTorch checkpoint containing:
      - model_state_dict  — the trained weights
      - model_config      — the architecture config (layers, classes, input shape)
      - class_names       — the class labels
      - n_channels, n_samples — input dimensions
      - trained_at        — ISO timestamp

    The inference script (GET /api/export/script/{job_id}) knows how to load
    this file and rebuild the model.
    """
    job = store.training_jobs.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Training job '{job_id}' not found."),
        )

    if job.get("status") != "done":
        return JSONResponse(
            status_code=400,
            content=api_response(False, error="Training has not completed yet. Wait for training to finish."),
        )

    model_path = job.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return JSONResponse(
            status_code=404,
            content=api_response(False, error="Model file not found on disk. It may have been deleted."),
        )

    model_name = job.get("model_name", "")
    filename   = f"{(model_name or job_id[:8]).replace(' ', '_')}.pt"

    logger.info("Serving model file for job %s: %s", job_id, model_path)

    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /api/export/script/{job_id}
# ---------------------------------------------------------------------------

@router.get("/script/{job_id}", response_model=None)
async def export_script(job_id: str) -> PlainTextResponse | JSONResponse:
    """
    Generate and download a standalone Python inference script for a trained model.

    The script is a complete .py file that:
      1. Contains all model-building code (no Neutrol import needed)
      2. Loads the .pt checkpoint by name from the same directory
      3. Exposes predict(window) → {class, confidence, probabilities}
      4. Can be run directly as a self-test

    The user should download both this script AND the model weights (.pt) and
    keep them in the same directory.
    """
    job = store.training_jobs.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Training job '{job_id}' not found."),
        )

    if job.get("status") != "done":
        return JSONResponse(
            status_code=400,
            content=api_response(False, error="Training has not completed yet."),
        )

    model_path = job.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return JSONResponse(
            status_code=404,
            content=api_response(False, error="Model file not found — cannot generate script."),
        )

    # Load the checkpoint to extract metadata for the script header
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.error("Failed to load checkpoint for job %s: %s", job_id, exc)
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"Could not read model file: {exc}"),
        )

    model_name  = job.get("model_name", "")
    script_code = generate_inference_script(job_id, checkpoint, model_name)

    script_stem = f"neutrol_inference_{job_id[:8]}"
    filename    = f"{script_stem}.py"

    logger.info("Generated inference script for job %s (%d bytes)", job_id, len(script_code))

    return PlainTextResponse(
        content=script_code,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
