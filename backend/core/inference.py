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
