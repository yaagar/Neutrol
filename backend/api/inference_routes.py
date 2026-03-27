"""
inference_routes.py — /api/inference/* and /ws/inference/* endpoints.

Endpoints:
  POST /api/inference/predict      — single-window prediction
  WS   /ws/inference/stream-run   — stream predictions for a full pipeline run

Streaming design:
  The frontend connects the WebSocket, sends { type: "start", job_id, run_id,
  delay_ms } and the backend streams one prediction per window at the requested
  speed. The backend has both the model (loaded from exports/) and the windows
  (in store.pipeline_runs) so no data needs to travel through the browser.

  Frontend can send { type: "stop" } to halt early.
  Backend sends { type: "prediction", window_index, predicted_class, confidence,
  scores, total } for each window, then { type: "done" } when finished.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.inference import load_model_for_inference, predict_window
from sessions.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter()
store  = SessionStore()

# Hard cap: never stream faster than 50 ms per window to avoid flooding the client
MIN_DELAY_MS = 50
MAX_DELAY_MS = 5000


def api_response(success: bool, data=None, error: str | None = None) -> dict:
    return {"success": success, "data": data, "error": error}


# ---------------------------------------------------------------------------
# POST /api/inference/predict — single-window prediction
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Body for single-window prediction."""
    job_id:      str
    window_data: list[list[float]]  # 2D list: (n_channels, n_samples)


@router.post("/predict")
async def predict_single(request: PredictRequest) -> JSONResponse:
    """
    Run the model on one window and return the predicted class + confidence.

    The window_data must match the input shape the model was trained on
    (same number of channels and samples).
    """
    try:
        model, checkpoint = load_model_for_inference(request.job_id)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content=api_response(False, error=str(e)))

    window = np.array(request.window_data, dtype=np.float32)
    result = predict_window(model, window, checkpoint["class_names"])

    return JSONResponse(status_code=200, content=api_response(True, data={
        **result,
        "job_id":      request.job_id,
        "class_names": checkpoint["class_names"],
    }))


# ---------------------------------------------------------------------------
# WS /ws/inference/stream-run — stream all windows in a pipeline run
# ---------------------------------------------------------------------------

@router.websocket("/ws/stream-run")
async def stream_run(websocket: WebSocket) -> None:
    """
    Stream predictions for every window in a pipeline run.

    Protocol:
      Client → Server:  { "type": "start", "job_id": "...", "run_id": "...", "delay_ms": 500 }
      Server → Client:  { "type": "ready", "class_names": [...], "total": N }
      Server → Client:  { "type": "prediction", "window_index": i, "predicted_class": "T1",
                          "confidence": 0.89, "scores": {...}, "total": N }
      ...
      Server → Client:  { "type": "done", "total": N }
      Client → Server:  { "type": "stop" }  ← client can send this at any time to halt
    """
    await websocket.accept()
    logger.info("Inference stream WebSocket connected")

    try:
        # Wait for the start message
        msg = await websocket.receive_json()

        if msg.get("type") != "start":
            await websocket.send_json({"type": "error", "message": "First message must be {type: 'start'}"})
            return

        job_id   = msg["job_id"]
        run_id   = msg["run_id"]
        delay_ms = max(MIN_DELAY_MS, min(MAX_DELAY_MS, int(msg.get("delay_ms", 500))))

        # Validate job exists and is complete
        job = store.training_jobs.get(job_id)
        if not job:
            await websocket.send_json({"type": "error", "message": f"Training job '{job_id}' not found."})
            return
        if job.get("status") != "done":
            await websocket.send_json({"type": "error", "message": f"Job '{job_id}' is not complete (status: {job.get('status')})."})
            return

        # Validate pipeline run exists — store.pipeline_runs holds a dict per run,
        # the actual window arrays are under the "windows" key
        run_entry = store.pipeline_runs.get(run_id)
        if not run_entry:
            await websocket.send_json({"type": "error", "message": f"Pipeline run '{run_id}' not found."})
            return
        windows = run_entry.get("windows", [])
        if not windows:
            await websocket.send_json({"type": "error", "message": f"Pipeline run '{run_id}' not found."})
            return

        # Load model — this may take a moment on first call
        try:
            model, checkpoint = load_model_for_inference(job_id)
        except Exception as e:
            await websocket.send_json({"type": "error", "message": f"Failed to load model: {e}"})
            return

        class_names = checkpoint["class_names"]
        total       = len(windows)

        await websocket.send_json({
            "type":        "ready",
            "class_names": class_names,
            "total":       total,
        })

        # Stream predictions window by window
        for i, window in enumerate(windows):
            # Check if client sent a stop message (non-blocking peek)
            try:
                stop_msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if stop_msg.get("type") == "stop":
                    logger.info("Inference stream stopped by client at window %d/%d", i, total)
                    break
            except asyncio.TimeoutError:
                pass  # No message from client — continue streaming

            # Run inference on this window
            result = predict_window(model, np.array(window, dtype=np.float32), class_names)

            await websocket.send_json({
                "type":            "prediction",
                "window_index":    i,
                "predicted_class": result["predicted_class"],
                "confidence":      result["confidence"],
                "scores":          result["scores"],
                "total":           total,
            })

            # Wait between windows to simulate real-time streaming speed
            await asyncio.sleep(delay_ms / 1000.0)

        await websocket.send_json({"type": "done", "total": total})
        logger.info("Inference stream complete: %d windows", total)

    except WebSocketDisconnect:
        logger.info("Inference stream WebSocket disconnected by client")
    except Exception as exc:
        logger.exception("Inference stream error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
