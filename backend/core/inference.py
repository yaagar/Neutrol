"""
inference.py — Load a trained model and run predictions on EEG windows.

Public functions:
  load_model_for_inference(job_id) — loads weights from exports/, returns
      (model, checkpoint) where checkpoint has class_names, n_channels, etc.

  predict_window(model, window, class_names) — runs one forward pass and
      returns the predicted class, confidence, and per-class softmax scores.

Public class:
  HardwareStreamer — wraps BrainFlow BoardShim to stream live EEG from a
      hardware device. Collects samples into a rolling buffer; get_window()
      returns the latest (n_channels, n_samples) window for inference.

Why dummy forward before load_state_dict?
  build_model() creates Lazy modules that don't allocate weights until the
  first forward pass. load_state_dict() requires keys to exist, so we must
  run a dummy tensor through the model to materialize all lazy layers first.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn as nn

from core.models import EEGNet, build_model

logger = logging.getLogger(__name__)

EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exports")

# BrainFlow board IDs for the boards we expose in the UI.
# Full list: https://brainflow.readthedocs.io/en/stable/SupportedBoards.html
SUPPORTED_BOARDS = {
    "synthetic":       -1,   # BrainFlow synthetic board — no hardware needed, great for testing
    "cyton":            0,   # OpenBCI Cyton (8-channel, 250 Hz)
    "cyton_daisy":      2,   # OpenBCI Cyton + Daisy (16-channel, 125 Hz)
    "ganglion":         1,   # OpenBCI Ganglion (4-channel, 200 Hz)
    "muse_s":          39,   # Muse S (4-channel, 256 Hz)
    "muse_2":          38,   # Muse 2 (4-channel, 256 Hz)
    "neurosity_notion": 13,  # Neurosity Notion 1
}


def load_model_for_inference(job_id: str) -> tuple[nn.Module, dict]:
    """
    Load a trained model from disk and return it in eval mode.

    1. Read the .pt checkpoint file saved by trainer.py
    2. Reconstruct the model architecture from the saved config
    3. Run a dummy forward pass to materialize any Lazy modules
    4. Load the saved weights into the materialized model

    Raises:
        FileNotFoundError: if no .pt file exists for this job_id.
    """
    model_path = os.path.join(EXPORTS_DIR, f"{job_id}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found for job '{job_id}'. Expected: {model_path}"
        )

    checkpoint   = torch.load(model_path, map_location="cpu", weights_only=False)
    n_channels   = checkpoint["n_channels"]
    n_samples    = checkpoint["n_samples"]
    n_classes    = checkpoint["n_classes"]
    model_config = checkpoint["model_config"]

    if "layers" in model_config and model_config["layers"]:
        model = build_model(model_config)
        dummy = torch.zeros(1, 1, n_channels, n_samples)
        with torch.no_grad():
            model.eval()
            model(dummy)
        logger.info("Reconstructed dynamic model for job %s (%d layers)",
                    job_id, len(model_config["layers"]))
    else:
        model = EEGNet(n_channels, n_samples, n_classes)
        logger.info("Reconstructed EEGNet for job %s", job_id)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded model for job %s — classes: %s", job_id, checkpoint["class_names"])
    return model, checkpoint


def predict_window(
    model:       nn.Module,
    window:      np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Run inference on a single EEG window.

    Args:
        model:       Loaded model in eval mode.
        window:      Shape (n_channels, n_samples).
        class_names: Ordered list matching the model's output neurons.

    Returns:
        { predicted_class, confidence (0-1), scores {class: prob} }
    """
    x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0)  # (1,1,ch,samp)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx   = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())
    scores     = {name: round(float(probs[i].item()), 4) for i, name in enumerate(class_names)}

    return {
        "predicted_class": class_names[pred_idx],
        "confidence":      round(confidence, 4),
        "scores":          scores,
    }


# ---------------------------------------------------------------------------
# HardwareStreamer — BrainFlow live EEG acquisition
# ---------------------------------------------------------------------------

class HardwareStreamer:
    """
    Wraps BrainFlow's BoardShim to stream live EEG from a hardware device.

    BrainFlow is a unified Python API for 30+ EEG/BCI devices. It handles
    the device-specific USB/Bluetooth/serial protocols internally and exposes
    a single `get_board_data()` call regardless of which device is connected.

    Usage:
        streamer = HardwareStreamer("cyton", serial_port="COM3")
        streamer.start()
        window = streamer.get_window(n_samples=256)  # (n_channels, n_samples)
        streamer.stop()

    The internal buffer accumulates samples from the device. get_window()
    returns the latest `n_samples` from that buffer. If fewer samples have
    arrived than requested, it returns None (caller should wait and retry).
    """

    def __init__(self, board_name: str, serial_port: str = "", ip_address: str = "") -> None:
        """
        Args:
            board_name:  Key from SUPPORTED_BOARDS (e.g. "cyton", "synthetic").
            serial_port: COM port for USB/serial boards (e.g. "COM3" or "/dev/ttyUSB0").
            ip_address:  IP address for WiFi boards (e.g. "192.168.4.1").
        """
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams
        except ImportError as exc:
            raise ImportError(
                "brainflow is required for hardware streaming. "
                "Install with: pip install brainflow"
            ) from exc

        if board_name not in SUPPORTED_BOARDS:
            raise ValueError(
                f"Unknown board '{board_name}'. Supported: {sorted(SUPPORTED_BOARDS)}"
            )

        self.board_name  = board_name
        self.board_id    = SUPPORTED_BOARDS[board_name]
        self._running    = False
        self._BoardShim  = BoardShim

        # BrainFlowInputParams carries connection details.
        # Only the fields relevant to the chosen board are used;
        # unused fields are silently ignored by BrainFlow.
        params = BrainFlowInputParams()
        if serial_port:
            params.serial_port = serial_port
        if ip_address:
            params.ip_address = ip_address

        self._board      = BoardShim(self.board_id, params)
        self._eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sample_rate   = BoardShim.get_sampling_rate(self.board_id)
        self.n_channels    = len(self._eeg_channels)

        logger.info(
            "HardwareStreamer: board=%s id=%d ch=%d sr=%d Hz",
            board_name, self.board_id, self.n_channels, self.sample_rate,
        )

    def start(self) -> None:
        """
        Prepare and begin streaming from the device.

        prepare_session() initialises the driver / opens the connection.
        start_stream() begins filling BrainFlow's internal ring buffer.
        """
        self._board.prepare_session()
        self._board.start_stream()
        self._running = True
        logger.info("HardwareStreamer started: %s", self.board_name)

    def stop(self) -> None:
        """Stop streaming and release the device connection."""
        if not self._running:
            return
        try:
            self._board.stop_stream()
            self._board.release_session()
        except Exception as exc:
            logger.warning("HardwareStreamer stop error: %s", exc)
        finally:
            self._running = False
            logger.info("HardwareStreamer stopped: %s", self.board_name)

    def get_window(self, n_samples: int) -> np.ndarray | None:
        """
        Return the latest n_samples from the device buffer.

        BrainFlow's get_board_data(n) returns the last n samples and removes
        them from the buffer. We request exactly n_samples to get a fresh,
        non-overlapping window each time.

        Returns:
            np.ndarray of shape (n_channels, n_samples), or None if fewer
            than n_samples have accumulated since the last call.
        """
        if not self._running:
            return None

        # Check how many samples are in the buffer without consuming them.
        # get_board_data_count() is available in BrainFlow >= 5.x.
        try:
            count = self._board.get_board_data_count()
        except Exception:
            count = n_samples  # fallback: attempt to read anyway

        if count < n_samples:
            return None

        # get_board_data returns shape (total_channels, n_samples).
        # We select only the EEG channels (not accelerometer, timestamp, etc.)
        data = self._board.get_board_data(n_samples)
        eeg  = data[self._eeg_channels, :]  # (n_eeg_channels, n_samples)

        return eeg.astype(np.float32)

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def list_boards() -> list[dict]:
        """Return display info for all supported boards."""
        try:
            from brainflow.board_shim import BoardShim
        except ImportError:
            return []

        result = []
        for name, board_id in SUPPORTED_BOARDS.items():
            try:
                sr  = BoardShim.get_sampling_rate(board_id)
                chs = len(BoardShim.get_eeg_channels(board_id))
                result.append({
                    "name":        name,
                    "board_id":    board_id,
                    "sample_rate": sr,
                    "n_channels":  chs,
                })
            except Exception:
                result.append({"name": name, "board_id": board_id})
        return result


# ---------------------------------------------------------------------------
# OutputDispatcher — fan-out predictions to external destinations
# ---------------------------------------------------------------------------

class OutputDispatcher:
    """
    Receives predictions and dispatches them to configured output destinations.

    Two independent outputs (both optional, both can run simultaneously):
      1. WebSocket server — Neutrol acts as a WS server on a configurable port.
         External scripts connect as clients and receive predictions as JSON.
         Best for Python, Node.js, or any language that supports WebSocket.

      2. Serial port — sends a short string to a COM/USB serial port.
         Best for Arduino, ESP32, or any microcontroller reading Serial.

    Called after every prediction regardless of inference source
    (simulated, file playback, or live hardware).

    Usage:
        dispatcher = OutputDispatcher()
        await dispatcher.start_ws_server(8765)
        dispatcher.start_serial("COM3", 9600, {"left": "L\\n", "right": "R\\n"})
        await dispatcher.dispatch("left", 0.92)
    """

    def __init__(self) -> None:
        self.ws_server: object | None = None   # asyncio WebSocket server
        self.ws_clients: set           = set() # connected external WS clients
        self.serial_conn: object | None = None  # pyserial Serial instance
        self.serial_map: dict[str, str] = {}   # label → string to send
        self.ws_port: int               = 8765
        self.enabled_ws: bool           = False
        self.enabled_serial: bool       = False

    # ---- WebSocket server ----

    async def start_ws_server(self, port: int) -> None:
        """
        Start a WebSocket server on the given port.

        External clients connect to ws://localhost:{port} and receive
        every prediction as: { "label": "left", "confidence": 0.87, "timestamp": ... }

        We keep a set of connected clients and broadcast to all of them.
        Clients that have disconnected are removed from the set silently.
        """
        try:
            import websockets  # already in requirements.txt
        except ImportError as exc:
            raise ImportError("websockets is required for WS output") from exc

        if self.ws_server is not None:
            await self.stop_ws_server()

        async def handler(websocket) -> None:
            # Add this new external client to our broadcast set
            self.ws_clients.add(websocket)
            logger.info("Output WS client connected (total: %d)", len(self.ws_clients))
            try:
                # Keep the connection alive until the client disconnects
                await websocket.wait_closed()
            finally:
                self.ws_clients.discard(websocket)
                logger.info("Output WS client disconnected (total: %d)", len(self.ws_clients))

        self.ws_server = await websockets.serve(handler, "0.0.0.0", port)
        self.ws_port    = port
        self.enabled_ws = True
        logger.info("Output WebSocket server started on port %d", port)

    async def stop_ws_server(self) -> None:
        """Stop the WebSocket server and disconnect all clients."""
        if self.ws_server:
            self.ws_server.close()
            await self.ws_server.wait_closed()
            self.ws_server   = None
            self.ws_clients.clear()
            self.enabled_ws  = False
            logger.info("Output WebSocket server stopped")

    # ---- Serial port ----

    def start_serial(self, port: str, baud: int, label_map: dict[str, str]) -> None:
        """
        Open a serial connection and configure the per-class message map.

        Args:
            port:      COM port string, e.g. "COM3" or "/dev/ttyUSB0".
            baud:      Baud rate, e.g. 9600 or 115200.
            label_map: Maps class label → string to send, e.g. {"left": "L\\n"}.
                       The string is sent verbatim (\\n becomes an actual newline).
        """
        try:
            import serial  # pyserial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for serial output. Install with: pip install pyserial"
            ) from exc

        if self.serial_conn and self.serial_conn.is_open:
            self.stop_serial()

        self.serial_conn    = serial.Serial(port, baud, timeout=1)
        # Interpret escape sequences in map values so "L\n" becomes "L" + newline
        self.serial_map     = {k: v.encode("utf-8").decode("unicode_escape") for k, v in label_map.items()}
        self.enabled_serial = True
        logger.info("Serial output opened: port=%s baud=%d map=%s", port, baud, label_map)

    def stop_serial(self) -> None:
        """Close the serial connection."""
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception as exc:
                logger.warning("Serial close error: %s", exc)
            finally:
                self.serial_conn    = None
                self.enabled_serial = False
                logger.info("Serial output closed")

    # ---- Dispatch ----

    async def dispatch(self, label: str, confidence: float) -> None:
        """
        Send a prediction to all enabled output destinations.

        Called after every prediction in the inference WebSocket handlers.
        Failures are logged but never propagate — a broken serial cable should
        not interrupt the live inference stream.
        """
        import time
        payload = {"label": label, "confidence": round(confidence, 4), "timestamp": int(time.time() * 1000)}

        # Broadcast to all connected WebSocket clients
        if self.enabled_ws and self.ws_clients:
            import json
            message = json.dumps(payload)
            dead = set()
            for client in list(self.ws_clients):
                try:
                    await client.send(message)
                except Exception:
                    # Client disconnected mid-stream — mark for removal
                    dead.add(client)
            self.ws_clients -= dead

        # Send string to serial port
        if self.enabled_serial and self.serial_conn and self.serial_conn.is_open:
            text = self.serial_map.get(label, "")
            if text:
                try:
                    self.serial_conn.write(text.encode("utf-8"))
                except Exception as exc:
                    logger.warning("Serial write error: %s", exc)

    @property
    def status(self) -> dict:
        """Return a JSON-serialisable status snapshot for the API."""
        return {
            "ws_enabled":      self.enabled_ws,
            "ws_port":         self.ws_port,
            "ws_clients":      len(self.ws_clients),
            "ws_running":      self.ws_server is not None,
            "serial_enabled":  self.enabled_serial,
            "serial_open":     bool(self.serial_conn and self.serial_conn.is_open),
            "serial_map":      self.serial_map,
        }
