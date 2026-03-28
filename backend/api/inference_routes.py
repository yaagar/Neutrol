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

from core.inference import HardwareStreamer, load_model_for_inference, predict_window
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


# ---------------------------------------------------------------------------
# GET /api/inference/hardware/boards — list all supported BrainFlow boards
# ---------------------------------------------------------------------------

@router.get("/hardware/boards")
async def list_hardware_boards() -> JSONResponse:
    """
    Return all boards that Neutrol supports via BrainFlow.

    Also indicates whether BrainFlow is installed, so the frontend can show
    a clear install message if it isn't.
    """
    try:
        boards = HardwareStreamer.list_boards()
        return JSONResponse(status_code=200, content=api_response(True, data={
            "boards":            boards,
            "brainflow_available": True,
        }))
    except Exception as exc:
        return JSONResponse(status_code=200, content=api_response(True, data={
            "boards":            [],
            "brainflow_available": False,
            "message": str(exc),
        }))


# ---------------------------------------------------------------------------
# POST /api/inference/hardware/connect — start streaming from a device
# ---------------------------------------------------------------------------

class HardwareConnectRequest(BaseModel):
    """Body for connecting to a BrainFlow device."""
    board_name:  str
    serial_port: str = ""
    ip_address:  str = ""


@router.post("/hardware/connect")
async def hardware_connect(request: HardwareConnectRequest) -> JSONResponse:
    """
    Connect to a BrainFlow device and begin streaming.

    Only one hardware device can be connected at a time. If one is already
    running, disconnect it first.

    The synthetic board (board_name="synthetic") requires no physical hardware
    and is useful for testing the live inference UI.
    """
    # Stop any existing streamer before creating a new one
    if store.hardware_streamer and store.hardware_streamer.is_running:
        store.hardware_streamer.stop()

    try:
        streamer = HardwareStreamer(
            board_name=request.board_name,
            serial_port=request.serial_port,
            ip_address=request.ip_address,
        )
        streamer.start()
        store.hardware_streamer = streamer

        return JSONResponse(status_code=200, content=api_response(True, data={
            "board_name":  streamer.board_name,
            "n_channels":  streamer.n_channels,
            "sample_rate": streamer.sample_rate,
            "status":      "connected",
        }))
    except ImportError as exc:
        return JSONResponse(status_code=422, content=api_response(
            False,
            error=f"BrainFlow not installed. Run: pip install brainflow  ({exc})",
        ))
    except Exception as exc:
        return JSONResponse(status_code=500, content=api_response(
            False, error=f"Failed to connect to {request.board_name}: {exc}",
        ))


# ---------------------------------------------------------------------------
# POST /api/inference/hardware/disconnect — stop the current hardware stream
# ---------------------------------------------------------------------------

@router.post("/hardware/disconnect")
async def hardware_disconnect() -> JSONResponse:
    """Stop and release the currently connected hardware device."""
    streamer = store.hardware_streamer
    if not streamer:
        return JSONResponse(status_code=200, content=api_response(True, data={
            "status": "no device connected",
        }))

    streamer.stop()
    store.hardware_streamer = None
    return JSONResponse(status_code=200, content=api_response(True, data={
        "status": "disconnected",
    }))


# ---------------------------------------------------------------------------
# WS /ws/inference/hardware — live hardware predictions via WebSocket
# ---------------------------------------------------------------------------

@router.websocket("/ws/hardware")
async def stream_hardware(websocket: WebSocket) -> None:
    """
    Stream live predictions from a connected BrainFlow device.

    Protocol:
      Client → Server: { "type": "start", "job_id": "...", "n_samples": 256, "delay_ms": 500 }
      Server → Client: { "type": "ready", "class_names": [...], "board_name": "...",
                         "sample_rate": 256, "n_channels": 8 }
      Server → Client: { "type": "prediction", "predicted_class": "T1",
                         "confidence": 0.87, "scores": {...} }
      ...
      Client → Server: { "type": "stop" }
      Server → Client: { "type": "done" }

    The backend polls the hardware buffer every delay_ms milliseconds, extracts
    the latest n_samples window, runs inference, and pushes the result.
    If the buffer doesn't have enough samples yet, the cycle is skipped silently.
    """
    await websocket.accept()
    logger.info("Hardware inference WebSocket connected")

    try:
        msg = await websocket.receive_json()

        if msg.get("type") != "start":
            await websocket.send_json({"type": "error", "message": "First message must be {type: 'start'}"})
            return

        job_id   = msg["job_id"]
        n_samples = int(msg.get("n_samples", 256))
        delay_ms  = max(MIN_DELAY_MS, min(MAX_DELAY_MS, int(msg.get("delay_ms", 500))))

        # Validate training job
        job = store.training_jobs.get(job_id)
        if not job or job.get("status") != "done":
            await websocket.send_json({"type": "error", "message": f"Training job '{job_id}' not ready."})
            return

        # Validate hardware device is connected
        streamer = store.hardware_streamer
        if not streamer or not streamer.is_running:
            await websocket.send_json({"type": "error",
                                       "message": "No hardware device connected. Use POST /hardware/connect first."})
            return

        # Load model
        try:
            model, checkpoint = load_model_for_inference(job_id)
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": f"Failed to load model: {exc}"})
            return

        class_names = checkpoint["class_names"]

        await websocket.send_json({
            "type":        "ready",
            "class_names": class_names,
            "board_name":  streamer.board_name,
            "sample_rate": streamer.sample_rate,
            "n_channels":  streamer.n_channels,
        })

        # Continuously poll the hardware buffer and push predictions
        while True:
            # Non-blocking check for stop message from client
            try:
                stop_msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if stop_msg.get("type") == "stop":
                    logger.info("Hardware inference stopped by client")
                    break
            except asyncio.TimeoutError:
                pass

            # Try to get a fresh window from the hardware buffer
            window = streamer.get_window(n_samples)
            if window is not None:
                result = predict_window(model, window, class_names)
                await websocket.send_json({
                    "type":            "prediction",
                    "predicted_class": result["predicted_class"],
                    "confidence":      result["confidence"],
                    "scores":          result["scores"],
                })

            await asyncio.sleep(delay_ms / 1000.0)

        await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("Hardware inference WebSocket disconnected by client")
    except Exception as exc:
        logger.exception("Hardware inference stream error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
