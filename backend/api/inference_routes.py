"""
inference_routes.py — /api/inference/* and /ws/inference/* endpoints.

Endpoints:
  POST /api/inference/predict           — single-window prediction
  WS   /ws/inference/stream-run         — stream: simulated OR file playback
  GET  /api/inference/hardware/boards   — list supported BrainFlow boards
  POST /api/inference/hardware/connect  — connect to live device
  POST /api/inference/hardware/disconnect
  WS   /ws/inference/hardware           — stream from live hardware
  GET  /api/inference/output/serial-ports — list available COM ports
  POST /api/inference/output/config     — configure output destinations
  GET  /api/inference/output/config     — get current output config + status

/ws/stream-run now supports two modes:
  "run"      — original behaviour: stream stored windows from a pipeline run
  "playback" — slice windows from a raw DataSession signal in real time

All prediction WebSocket handlers call store.output_dispatcher.dispatch() after
each prediction so results flow to external WebSocket clients and/or serial.
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
# WS /ws/inference/stream-run — run mode OR playback mode
# ---------------------------------------------------------------------------

@router.websocket("/ws/stream-run")
async def stream_run(websocket: WebSocket) -> None:
    """
    Stream predictions via WebSocket. Supports two modes:

    Mode "run" (original behaviour):
      Streams all stored windows from a completed pipeline run.
      Client sends: { "type": "start", "mode": "run", "job_id": "...",
                      "run_id": "...", "delay_ms": 500 }

    Mode "playback" (new):
      Replays a raw DataSession's signal array in real time, slicing windows
      sequentially with the same window_sec and overlap_ratio as a chosen run.
      Client sends: { "type": "start", "mode": "playback", "job_id": "...",
                      "session_id": "...", "run_id": "...", "delay_ms": 200 }
      The run_id is used to get the window shape. The session signal is re-sliced
      on the fly — not the stored windows. This lets users replay any recording
      as if it were live data.

    Common server → client messages:
      { "type": "ready", "class_names": [...], "total": N }
      { "type": "prediction", "window_index": i, "predicted_class": "T1",
        "confidence": 0.89, "scores": {...}, "total": N }
      { "type": "done", "total": N }
      { "type": "error", "message": "..." }
    """
    await websocket.accept()
    logger.info("Inference stream WebSocket connected")

    try:
        msg = await websocket.receive_json()

        if msg.get("type") != "start":
            await websocket.send_json({"type": "error", "message": "First message must be {type: 'start'}"})
            return

        mode     = msg.get("mode", "run")   # "run" or "playback"
        job_id   = msg["job_id"]
        delay_ms = max(MIN_DELAY_MS, min(MAX_DELAY_MS, int(msg.get("delay_ms", 500))))

        # Validate training job
        job = store.training_jobs.get(job_id)
        if not job:
            await websocket.send_json({"type": "error", "message": f"Training job '{job_id}' not found."})
            return
        if job.get("status") != "done":
            await websocket.send_json({"type": "error", "message": f"Job '{job_id}' not complete."})
            return

        # Load model
        try:
            model, checkpoint = load_model_for_inference(job_id)
        except Exception as e:
            await websocket.send_json({"type": "error", "message": f"Failed to load model: {e}"})
            return

        class_names = checkpoint["class_names"]

        # ------------------------------------------------------------------
        # Build the window list depending on mode
        # ------------------------------------------------------------------

        if mode == "playback":
            # Playback: re-slice the raw session signal.
            # window shape comes from the reference run (so it matches training).
            session_id = msg.get("session_id")
            run_id     = msg.get("run_id", "")

            session = store.sessions.get(session_id)
            if not session:
                await websocket.send_json({"type": "error", "message": f"Session '{session_id}' not found."})
                return

            # Get window size from the reference run or fall back to checkpoint
            run_entry    = store.pipeline_runs.get(run_id, {})
            window_shape = run_entry.get("window_shape")
            if window_shape:
                n_ch, n_samp = window_shape[0], window_shape[1]
            else:
                # Fall back to the model's expected input size
                n_ch   = checkpoint.get("n_channels", session.signals.shape[0])
                n_samp = checkpoint.get("n_samples", 256)

            # Slice the session signal into non-overlapping windows
            signals = session.signals  # (total_channels, total_samples)
            # Only keep the first n_ch channels to match the model
            signals = signals[:n_ch, :]

            windows = []
            start   = 0
            while start + n_samp <= signals.shape[1]:
                windows.append(signals[:, start:start + n_samp])
                start += n_samp

            if not windows:
                await websocket.send_json({"type": "error",
                                           "message": "Session signal too short to produce any windows."})
                return

        else:
            # Original "run" mode: use stored pipeline windows
            run_id    = msg["run_id"]
            run_entry = store.pipeline_runs.get(run_id)
            if not run_entry:
                await websocket.send_json({"type": "error", "message": f"Pipeline run '{run_id}' not found."})
                return
            windows = run_entry.get("windows", [])
            if not windows:
                await websocket.send_json({"type": "error", "message": f"No windows in run '{run_id}'."})
                return

        total = len(windows)

        await websocket.send_json({
            "type":        "ready",
            "class_names": class_names,
            "total":       total,
            "mode":        mode,
        })

        # ------------------------------------------------------------------
        # Stream predictions window by window
        # ------------------------------------------------------------------
        for i, window in enumerate(windows):
            # Non-blocking check for stop message
            try:
                stop_msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if stop_msg.get("type") == "stop":
                    logger.info("Stream stopped by client at window %d/%d", i, total)
                    break
            except asyncio.TimeoutError:
                pass

            result = predict_window(model, np.array(window, dtype=np.float32), class_names)

            await websocket.send_json({
                "type":            "prediction",
                "window_index":    i,
                "predicted_class": result["predicted_class"],
                "confidence":      result["confidence"],
                "scores":          result["scores"],
                "total":           total,
            })

            # Dispatch to external outputs (WebSocket server + serial)
            await store.output_dispatcher.dispatch(result["predicted_class"], result["confidence"])

            await asyncio.sleep(delay_ms / 1000.0)

        await websocket.send_json({"type": "done", "total": total})
        logger.info("Stream complete: %d windows (mode=%s)", total, mode)

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
            "boards":              boards,
            "brainflow_available": True,
        }))
    except Exception as exc:
        return JSONResponse(status_code=200, content=api_response(True, data={
            "boards":              [],
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
    """
    await websocket.accept()
    logger.info("Hardware inference WebSocket connected")

    try:
        msg = await websocket.receive_json()

        if msg.get("type") != "start":
            await websocket.send_json({"type": "error", "message": "First message must be {type: 'start'}"})
            return

        job_id    = msg["job_id"]
        n_samples = int(msg.get("n_samples", 256))
        delay_ms  = max(MIN_DELAY_MS, min(MAX_DELAY_MS, int(msg.get("delay_ms", 500))))

        job = store.training_jobs.get(job_id)
        if not job or job.get("status") != "done":
            await websocket.send_json({"type": "error", "message": f"Training job '{job_id}' not ready."})
            return

        streamer = store.hardware_streamer
        if not streamer or not streamer.is_running:
            await websocket.send_json({"type": "error",
                                       "message": "No hardware device connected. Use POST /hardware/connect first."})
            return

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

        while True:
            try:
                stop_msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if stop_msg.get("type") == "stop":
                    logger.info("Hardware inference stopped by client")
                    break
            except asyncio.TimeoutError:
                pass

            window = streamer.get_window(n_samples)
            if window is not None:
                result = predict_window(model, window, class_names)
                await websocket.send_json({
                    "type":            "prediction",
                    "predicted_class": result["predicted_class"],
                    "confidence":      result["confidence"],
                    "scores":          result["scores"],
                })
                # Dispatch to external outputs
                await store.output_dispatcher.dispatch(result["predicted_class"], result["confidence"])

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


# ---------------------------------------------------------------------------
# GET /api/inference/output/serial-ports — list available COM/serial ports
# ---------------------------------------------------------------------------

@router.get("/output/serial-ports")
async def list_serial_ports() -> JSONResponse:
    """
    Return all serial ports visible to the OS.

    Uses serial.tools.list_ports from pyserial — the same library used to
    send data. Returns an empty list (not an error) if pyserial is missing,
    so the frontend can gracefully show "no ports found".
    """
    try:
        from serial.tools.list_ports import comports
        ports = [
            {"port": p.device, "description": p.description or p.device}
            for p in comports()
        ]
        return JSONResponse(status_code=200, content=api_response(True, data={
            "ports":            ports,
            "pyserial_available": True,
        }))
    except ImportError:
        return JSONResponse(status_code=200, content=api_response(True, data={
            "ports":            [],
            "pyserial_available": False,
        }))
    except Exception as exc:
        return JSONResponse(status_code=200, content=api_response(True, data={
            "ports":            [],
            "pyserial_available": True,
            "error": str(exc),
        }))


# ---------------------------------------------------------------------------
# POST /api/inference/output/config — configure output destinations
# ---------------------------------------------------------------------------

class OutputConfigRequest(BaseModel):
    """Body for configuring output destinations."""
    ws_enabled:    bool          = False
    ws_port:       int           = 8765
    serial_enabled: bool         = False
    serial_port:   str           = ""
    serial_baud:   int           = 9600
    # Maps class label → string to send over serial, e.g. {"left": "L\n"}
    serial_map:    dict[str, str] = {}


@router.post("/output/config")
async def set_output_config(request: OutputConfigRequest) -> JSONResponse:
    """
    Apply a new output configuration.

    This may start/stop the WebSocket server and open/close the serial port.
    The configuration is saved in store.output_config so GET /output/config
    can return the current state on the next page load.
    """
    dispatcher = store.output_dispatcher

    # ---- WebSocket output ----
    if request.ws_enabled and not dispatcher.enabled_ws:
        # Start the WS server on the requested port
        await dispatcher.start_ws_server(request.ws_port)
    elif not request.ws_enabled and dispatcher.enabled_ws:
        await dispatcher.stop_ws_server()
    elif request.ws_enabled and dispatcher.ws_port != request.ws_port:
        # Port changed — restart on new port
        await dispatcher.stop_ws_server()
        await dispatcher.start_ws_server(request.ws_port)

    # ---- Serial output ----
    if request.serial_enabled and request.serial_port:
        try:
            dispatcher.start_serial(request.serial_port, request.serial_baud, request.serial_map)
        except ImportError as exc:
            return JSONResponse(status_code=422, content=api_response(
                False, error=f"pyserial not installed. Run: pip install pyserial  ({exc})"
            ))
        except Exception as exc:
            return JSONResponse(status_code=500, content=api_response(
                False, error=f"Could not open serial port {request.serial_port}: {exc}"
            ))
    elif not request.serial_enabled and dispatcher.enabled_serial:
        dispatcher.stop_serial()

    # Persist the config dict for GET /output/config
    store.output_config = request.model_dump()

    return JSONResponse(status_code=200, content=api_response(True, data={
        **dispatcher.status,
        "config": store.output_config,
    }))


# ---------------------------------------------------------------------------
# GET /api/inference/output/config — current output config and status
# ---------------------------------------------------------------------------

@router.get("/output/config")
async def get_output_config() -> JSONResponse:
    """
    Return the current output configuration and live status.

    `config` reflects what was last POSTed.
    The status fields (ws_running, ws_clients, serial_open) reflect live state.
    """
    return JSONResponse(status_code=200, content=api_response(True, data={
        **store.output_dispatcher.status,
        "config": store.output_config,
    }))
