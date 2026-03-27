"""
export.py — Generates standalone Python inference scripts for trained models.

The generate_inference_script() function returns a .py file as a string.
The script is fully standalone: it only requires numpy and torch, and can
be run from any machine that has those packages — no Neutrol code needed.

How the generated script works:
  1. Contains the full model builder (copy of core/models.py logic)
  2. Loads the .pt checkpoint file by name (user must keep them together)
  3. Rebuilds the model using the saved config, materializes lazy modules,
     then loads the trained weights
  4. Exposes a predict(window) function: takes (n_channels, n_samples) numpy
     array, returns {class, confidence, probabilities}

Why include the model builder instead of torch.jit?
  torch.jit.trace/script both have limitations with custom lazy modules and
  conditional logic. The copy-and-rebuild approach is simpler, more
  transparent for learners, and guaranteed to produce identical output.
"""

from __future__ import annotations

from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Script template — written as a format string with {placeholders}
# ---------------------------------------------------------------------------

_SCRIPT_TEMPLATE = '''\
#!/usr/bin/env python3
"""
Neutrol Auto-Generated Inference Script
Model: {model_name}
Classes: {classes_str}
Trained: {trained_at}
Generated: {generated_at}

Requirements:
    pip install torch numpy

Usage:
    import numpy as np
    from {script_name} import predict

    # window must be a pre-processed numpy array: shape (n_channels, n_samples)
    # Apply the same preprocessing pipeline you used during training.
    window = np.zeros(({n_channels}, {n_samples}), dtype=np.float32)
    result = predict(window)
    # result = {{'class': 'T1', 'confidence': 0.87,
    #            'probabilities': {{'T0': 0.05, 'T1': 0.87, 'T2': 0.08}}}}

To run a quick self-test:
    python {script_name}.py
"""

import os
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Model architecture utilities (copied from Neutrol core/models.py)
# These are included verbatim so this script has no Neutrol dependency.
# ---------------------------------------------------------------------------

_ACTIVATIONS = {{
    "ELU":       nn.ELU,
    "ReLU":      nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh":      nn.Tanh,
    "Sigmoid":   nn.Sigmoid,
    "GELU":      nn.GELU,
}}


def _get_activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{{name}}'. Valid: {{sorted(_ACTIVATIONS.keys())}}")
    return _ACTIVATIONS[name]()


def _resolve_kernel(kernel_size, n_channels: int):
    """Replace the \'n_channels\' keyword with the actual channel count."""
    if isinstance(kernel_size, list):
        return tuple(n_channels if k == "n_channels" else int(k) for k in kernel_size)
    if kernel_size == "n_channels":
        return n_channels
    return int(kernel_size)


class _LazyDepthwiseConv2d(nn.Module):
    """Depthwise separable conv — one filter per input channel. Lazily infers in_channels."""
    def __init__(self, depth_multiplier: int, kernel_size, bias: bool = False) -> None:
        super().__init__()
        self.depth_multiplier = depth_multiplier
        self.kernel_size      = kernel_size
        self.bias             = bias
        self._conv            = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._conv is None:
            in_ch  = x.shape[1]
            out_ch = in_ch * self.depth_multiplier
            self._conv = nn.Conv2d(
                in_ch, out_ch,
                kernel_size=self.kernel_size,
                groups=in_ch,
                bias=self.bias,
            ).to(x.device)
        return self._conv(x)


class _LSTMWrapper(nn.Module):
    """LSTM that accepts flat input from a preceding Flatten layer."""
    def __init__(self, hidden_size: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self._dropout    = dropout
        self._lstm       = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self._lstm is None:
            lstm_drop  = self._dropout if self.num_layers > 1 else 0.0
            self._lstm = nn.LSTM(
                x.shape[2], self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=lstm_drop,
            ).to(x.device)
        out, _ = self._lstm(x)
        return out[:, -1, :]


def _build_layer_modules(layer_cfg: dict, n_channels: int, n_classes: int) -> list:
    t       = layer_cfg["type"]
    modules = []

    if t == "Conv2d":
        out_ch  = int(layer_cfg["out_channels"])
        kernel  = _resolve_kernel(layer_cfg["kernel_size"], n_channels)
        padding = layer_cfg.get("padding", 0)
        modules.append(nn.LazyConv2d(out_ch, kernel, padding=padding, bias=False))
        if layer_cfg.get("batch_norm", False):
            modules.append(nn.LazyBatchNorm2d())
        if "activation" in layer_cfg:
            modules.append(_get_activation(layer_cfg["activation"]))

    elif t == "DepthwiseConv2d":
        dm     = int(layer_cfg.get("depth_multiplier", 1))
        kernel = _resolve_kernel(layer_cfg["kernel_size"], n_channels)
        modules.append(_LazyDepthwiseConv2d(dm, kernel, bias=False))
        if layer_cfg.get("batch_norm", False):
            modules.append(nn.LazyBatchNorm2d())
        if "activation" in layer_cfg:
            modules.append(_get_activation(layer_cfg["activation"]))

    elif t == "AvgPool2d":
        modules.append(nn.AvgPool2d(_resolve_kernel(layer_cfg["kernel_size"], n_channels)))

    elif t == "MaxPool2d":
        modules.append(nn.MaxPool2d(_resolve_kernel(layer_cfg["kernel_size"], n_channels)))

    elif t == "Dropout":
        modules.append(nn.Dropout(float(layer_cfg.get("rate", 0.5))))

    elif t == "BatchNorm2d":
        modules.append(nn.LazyBatchNorm2d())

    elif t == "Flatten":
        modules.append(nn.Flatten())

    elif t == "Linear":
        out_feat = layer_cfg["out_features"]
        if out_feat == "n_classes":
            out_feat = n_classes
        modules.append(nn.LazyLinear(int(out_feat)))

    elif t == "LSTM":
        modules.append(_LSTMWrapper(
            hidden_size=int(layer_cfg.get("hidden_size", 64)),
            num_layers=int(layer_cfg.get("num_layers", 1)),
            dropout=float(layer_cfg.get("dropout", 0.0)),
        ))

    return modules


def _build_model(config: dict) -> nn.Module:
    """Rebuild model from a saved ModelConfig dict."""
    n_channels = config.get("input_channels", 1)
    n_classes  = len(config.get("classes", [1]))
    layers_cfg = config.get("layers", [])

    modules = []
    for layer_cfg in layers_cfg:
        modules.extend(_build_layer_modules(layer_cfg, n_channels, n_classes))
    return nn.Sequential(*modules)


# ---------------------------------------------------------------------------
# Load checkpoint + model
# ---------------------------------------------------------------------------

# Path to the .pt file — must be in the same directory as this script.
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{job_id}.pt")

_cached_model   = None
_cached_classes = None


def _get_model():
    """Load and cache the model. Materializes lazy modules on first call."""
    global _cached_model, _cached_classes
    if _cached_model is not None:
        return _cached_model, _cached_classes

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    model_config = checkpoint["model_config"]
    n_channels   = checkpoint["n_channels"]
    n_samples    = checkpoint["n_samples"]
    classes      = checkpoint["class_names"]

    # Build model with lazy modules
    model = _build_model(model_config)

    # Materialize lazy modules with a dummy forward pass before loading weights.
    # Lazy modules record their input shapes on the first call and allocate weights.
    dummy = torch.zeros(1, 1, n_channels, n_samples)
    with torch.no_grad():
        model.eval()
        model(dummy)

    # Load the trained weights — shapes now match because we used the same dummy shape
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _cached_model   = model
    _cached_classes = classes
    return model, classes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(window: np.ndarray) -> dict:
    """
    Run inference on one pre-processed EEG window.

    Args:
        window: numpy array of shape (n_channels, n_samples).
                This must already be preprocessed (bandpass filtered, normalized, etc.)
                using the same pipeline that was applied during training.

    Returns:
        dict with:
            "class"         — predicted class label (str)
            "confidence"    — probability of the predicted class (float 0-1)
            "probabilities" — dict mapping every class name to its probability
    """
    model, classes = _get_model()

    # Add batch and spatial dimensions:
    # (n_channels, n_samples) -> (1, 1, n_channels, n_samples)
    x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        # Softmax converts raw logits to probabilities that sum to 1
        probs  = torch.softmax(logits, dim=1).squeeze(0).numpy()

    pred_idx = int(probs.argmax())
    return {{
        "class":         classes[pred_idx],
        "confidence":    float(probs[pred_idx]),
        "probabilities": {{c: float(p) for c, p in zip(classes, probs)}},
    }}


# ---------------------------------------------------------------------------
# Self-test — run this file directly to verify the model loads correctly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    n_channels = checkpoint["n_channels"]
    n_samples  = checkpoint["n_samples"]
    classes    = checkpoint["class_names"]

    print(f"Model loaded from: {{CHECKPOINT_PATH}}")
    print(f"  Classes : {{classes}}")
    print(f"  Input   : ({{n_channels}} channels, {{n_samples}} samples)")
    print()

    # Run a zero-filled dummy window through the model
    dummy_window = np.zeros((n_channels, n_samples), dtype=np.float32)
    result = predict(dummy_window)

    print(f"Test prediction (zero input):")
    print(f"  Predicted class : {{result[\'class\']}}")
    print(f"  Confidence      : {{result[\'confidence\']:.1%}}")
    print(f"  All probabilities:")
    for cls, prob in result["probabilities"].items():
        bar = "#" * int(prob * 20)
        print(f"    {{cls:12s}}  {{prob:.1%}}  {{bar}}")
'''


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_inference_script(
    job_id: str,
    checkpoint: dict,
    model_name: str = "",
) -> str:
    """
    Generate a standalone Python inference script for a trained model.

    The returned string is a complete .py file. Save it alongside the .pt
    checkpoint file and it will work without any Neutrol installation.

    Args:
        job_id:      Training job UUID — also the stem of the .pt filename.
        checkpoint:  The loaded torch checkpoint dict (from torch.load).
        model_name:  Human-readable name for the model (shown in the docstring).

    Returns:
        Python source code as a string.
    """
    classes    = checkpoint.get("class_names", [])
    trained_at = checkpoint.get("trained_at", "unknown")
    n_channels = checkpoint.get("n_channels", 0)
    n_samples  = checkpoint.get("n_samples",  0)

    script_name = f"neutrol_inference_{job_id[:8]}"

    return _SCRIPT_TEMPLATE.format(
        job_id       = job_id,
        model_name   = model_name or f"model_{job_id[:8]}",
        classes_str  = ", ".join(classes),
        trained_at   = trained_at,
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        script_name  = script_name,
        n_channels   = n_channels,
        n_samples    = n_samples,
    )
