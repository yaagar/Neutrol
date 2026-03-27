"""
inference.py — Load a trained model and run predictions on EEG windows.

Two public functions:
  load_model_for_inference(job_id) — loads weights from exports/, returns
      (model, checkpoint) where checkpoint has class_names, n_channels, etc.

  predict_window(model, window, class_names) — runs one forward pass and
      returns the predicted class, confidence, and per-class softmax scores.

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
