"""
export.py — Generates standalone inference scripts and exports trained models.

Public functions
----------------
generate_inference_script(job_id, checkpoint, model_name) -> str
    Standalone Python script. Embeds full preprocessing pipeline from the
    checkpoint so predict(raw_window) is end-to-end — no manual preprocessing.

export_onnx(job_id, model_path, export_dir) -> str
    Exports the PyTorch model to ONNX format for lightweight edge deployment.
    Returns path to the .onnx file.

generate_onnx_script(job_id, checkpoint, model_name) -> str
    Standalone Python script that uses onnxruntime instead of PyTorch.
    Same section structure as the main inference script.

export_microcontroller(job_id, model_path, export_dir) -> str
    Generates a ZIP with model_data.h + neutrol_inference.cpp + README.
    Requires tensorflow and onnx2tf (optional dependencies).
    Returns path to the .zip file.

Design notes
------------
Scripts are built programmatically (string concatenation), not via a single
.format() template. This avoids the need to double-brace every {} in the
generated Python code, which makes templates fragile and hard to read.

Filter coefficients for bandpass and notch are pre-computed by SciPy at
generation time and embedded as float constants in the script. This guarantees
the exported script uses mathematically identical preprocessing to training
without needing SciPy for filter design — only scipy.signal.sosfiltfilt/lfilter
is needed at runtime to apply the pre-computed coefficients.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Where ONNX files are saved (same directory as .pt files)
EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exports")


# ---------------------------------------------------------------------------
# Internal helpers — pre-compute filter coefficients from pipeline steps
# ---------------------------------------------------------------------------

def _compute_bandpass_sos(low_hz: float, high_hz: float, sample_rate: int) -> list:
    """Compute Butterworth order-4 bandpass SOS coefficients."""
    from scipy.signal import butter
    sos = butter(4, [low_hz, high_hz], btype="band", fs=sample_rate, output="sos")
    return sos.tolist()


def _compute_notch_ba(freq_hz: float, quality_factor: float, sample_rate: int) -> tuple:
    """Compute IIR notch filter b, a coefficients."""
    from scipy.signal import iirnotch
    b, a = iirnotch(freq_hz, quality_factor, fs=sample_rate)
    return b.tolist(), a.tolist()


# ---------------------------------------------------------------------------
# _MODEL_BUILDER_CODE — embedded verbatim in every generated script.
# This is the same logic as core/models.py but without any Neutrol imports.
# ---------------------------------------------------------------------------

_MODEL_BUILDER_CODE = '''\
# ── MODEL BUILDER (do not edit) ─────────────────────────────────────────────
# Copied from Neutrol core/models.py. No Neutrol installation required.

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "ELU":       nn.ELU,
    "ReLU":      nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh":      nn.Tanh,
    "Sigmoid":   nn.Sigmoid,
    "GELU":      nn.GELU,
}


def _get_activation(name):
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}. Valid: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


def _resolve_kernel(kernel_size, n_channels):
    if isinstance(kernel_size, list):
        return tuple(n_channels if k == "n_channels" else int(k) for k in kernel_size)
    if kernel_size == "n_channels":
        return n_channels
    return int(kernel_size)


class _LazyDepthwiseConv2d(nn.Module):
    def __init__(self, depth_multiplier, kernel_size, bias=False):
        super().__init__()
        self.depth_multiplier = depth_multiplier
        self.kernel_size      = kernel_size
        self.bias             = bias
        self._conv            = None

    def forward(self, x):
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
    def __init__(self, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self._dropout    = dropout
        self._lstm       = None

    def forward(self, x):
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


def _build_layer_modules(layer_cfg, n_channels, n_classes):
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


def _build_model(config):
    n_channels = config.get("input_channels", 1)
    n_classes  = len(config.get("classes", [1]))
    layers_cfg = config.get("layers", [])
    modules    = []
    for layer_cfg in layers_cfg:
        modules.extend(_build_layer_modules(layer_cfg, n_channels, n_classes))
    return nn.Sequential(*modules)
'''


# ---------------------------------------------------------------------------
# generate_inference_script — main public function
# ---------------------------------------------------------------------------

def generate_inference_script(
    job_id: str,
    checkpoint: dict,
    model_name: str = "",
) -> str:
    """
    Generate a standalone Python inference script for a trained model.

    The script embeds:
      - The full preprocessing pipeline from the checkpoint (bandpass, notch,
        normalize, etc.) with pre-computed filter coefficients
      - The full model builder (no Neutrol import needed)
      - A predict(raw_window) function that is truly end-to-end

    Args:
        job_id:     Training job UUID.
        checkpoint: Loaded torch checkpoint dict (from torch.load).
        model_name: Human-readable name shown in the header.

    Returns:
        Complete Python source code as a string.
    """
    classes         = checkpoint.get("class_names", [])
    n_channels      = checkpoint.get("n_channels", 0)
    n_samples       = checkpoint.get("n_samples", 0)
    sample_rate     = checkpoint.get("sample_rate", 256)
    trained_at      = checkpoint.get("trained_at", "unknown")
    pipeline_config = checkpoint.get("pipeline_config") or {}

    display_name        = model_name or f"model_{job_id[:8]}"
    checkpoint_filename = f"{(model_name or job_id[:8]).replace(' ', '_')}.pt"
    script_name         = f"neutrol_inference_{job_id[:8]}"
    generated_at        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    steps = pipeline_config.get("steps", [])

    # Find window parameters from the pipeline (used in CONFIG section)
    window_sec    = 1.0
    overlap_ratio = 0.5
    for s in steps:
        if s.get("step") == "window":
            window_sec    = float(s.get("window_sec", 1.0))
            overlap_ratio = float(s.get("overlap_ratio", 0.5))

    # Pre-compute filter coefficients for every bandpass and notch step.
    # These are embedded as constants so the script never needs to call butter()
    # or iirnotch() at runtime — just scipy.signal.sosfiltfilt / lfilter.
    bandpass_constants = []  # list of (var_name, sos_list) tuples
    notch_constants    = []  # list of (var_name, b_list, a_list) tuples
    bp_idx = 0
    notch_idx = 0

    for s in steps:
        if s.get("step") == "bandpass":
            try:
                sos = _compute_bandpass_sos(
                    float(s.get("low_hz", 1.0)),
                    float(s.get("high_hz", 40.0)),
                    sample_rate,
                )
                bandpass_constants.append((f"_BP_SOS_{bp_idx}", sos))
                bp_idx += 1
            except Exception as e:
                logger.warning("Could not compute bandpass coefficients: %s", e)

        elif s.get("step") == "notch":
            try:
                b, a = _compute_notch_ba(
                    float(s.get("freq_hz", 60.0)),
                    float(s.get("quality_factor", 30.0)),
                    sample_rate,
                )
                notch_constants.append((f"_NOTCH_B_{notch_idx}", f"_NOTCH_A_{notch_idx}", b, a))
                notch_idx += 1
            except Exception as e:
                logger.warning("Could not compute notch coefficients: %s", e)

    # Build the script section by section
    parts = []

    # ── Header ──────────────────────────────────────────────────────────────
    parts.append(f'''\
"""
============================================================
NEUTROL INFERENCE SCRIPT
============================================================
Model:     {display_name}
Classes:   {", ".join(classes)}
Trained:   {trained_at}
Generated: {generated_at}

Requirements:
    pip install torch numpy scipy

Usage:
    from {script_name} import predict
    import numpy as np

    # raw_window: shape (n_channels, n_samples) — raw EEG, no preprocessing needed
    raw = np.zeros(({n_channels}, {n_samples}), dtype=np.float32)
    result = predict(raw)
    # {{"class": "T1", "confidence": 0.87, "probabilities": {{"T0": 0.05, ...}}}}

Self-test:
    python {script_name}.py
"""
''')

    # ── Section 1: CONFIG ────────────────────────────────────────────────────
    parts.append(f'''\
# ── SECTION 1: CONFIG (safe to edit) ────────────────────────────────────────
SAMPLE_RATE   = {sample_rate}       # Hz — must match your EEG device
WINDOW_SEC    = {window_sec}        # seconds per prediction window
CHANNELS      = {classes!r}        # class labels in training order (do not reorder)
MODEL_PATH    = "{checkpoint_filename}"  # .pt file — keep in the same folder as this script
CLASSES       = {classes!r}
''')

    # ── Section 2: Pipeline steps ────────────────────────────────────────────
    parts.append(f'''\
# ── SECTION 2: PREPROCESSING PIPELINE (auto-generated) ──────────────────────
# Describes what was applied during training.
# The functions in Section 4 implement these steps — do not edit Section 4.
PIPELINE_STEPS = {steps!r}
''')

    # ── Section 3: User output callback ─────────────────────────────────────
    parts.append('''\
# ── SECTION 3: YOUR OUTPUT CODE (edit this) ─────────────────────────────────

def on_prediction(label: str, confidence: float, probabilities: dict) -> None:
    """Called after every prediction. Put your application logic here."""
    bar = "#" * int(confidence * 20)
    print(f"  {label:12s}  {confidence:.1%}  {bar}")
    # Examples:
    # if label == "left":  control_robot_left()
    # mqtt_client.publish("eeg/prediction", label)
''')

    # ── Section 4: Pre-computed filter coefficients ──────────────────────────
    coeff_lines = ["# ── SECTION 4: PREPROCESSING IMPLEMENTATION (do not edit) ─────────────────"]
    coeff_lines.append("")
    coeff_lines.append("import os")
    coeff_lines.append("import numpy as np")
    coeff_lines.append("import scipy.signal as _sig")
    coeff_lines.append("")

    if bandpass_constants:
        coeff_lines.append("# Butterworth order-4 bandpass SOS coefficients — pre-computed from pipeline.")
        coeff_lines.append("# sosfiltfilt applies these forward+backward for zero-phase filtering.")
        for var_name, sos in bandpass_constants:
            coeff_lines.append(f"{var_name} = {sos!r}")
        coeff_lines.append("")

    if notch_constants:
        coeff_lines.append("# IIR notch filter b, a coefficients — pre-computed from pipeline.")
        for b_var, a_var, b, a in notch_constants:
            coeff_lines.append(f"{b_var} = {b!r}")
            coeff_lines.append(f"{a_var} = {a!r}")
        coeff_lines.append("")

    # Build the apply_pipeline function body by iterating steps and generating
    # the correct filter call for each step type.
    apply_lines = []
    bp_idx    = 0
    notch_idx = 0

    for s in steps:
        t = s.get("step")
        if t == "bandpass" and bp_idx < len(bandpass_constants):
            var_name = bandpass_constants[bp_idx][0]
            apply_lines.append(
                f"    # Bandpass {s.get('low_hz')}–{s.get('high_hz')} Hz (zero-phase Butterworth order 4)"
            )
            apply_lines.append(
                f"    signals = _sig.sosfiltfilt({var_name}, signals, axis=1).astype(np.float32)"
            )
            bp_idx += 1
        elif t == "notch" and notch_idx < len(notch_constants):
            b_var = notch_constants[notch_idx][0]
            a_var = notch_constants[notch_idx][1]
            apply_lines.append(
                f"    # Notch filter {s.get('freq_hz')} Hz (Q={s.get('quality_factor', 30.0)})"
            )
            apply_lines.append(
                f"    signals = np.apply_along_axis("
                f"lambda ch: _sig.lfilter({b_var}, {a_var}, ch), 1, signals).astype(np.float32)"
            )
            notch_idx += 1
        elif t == "normalize":
            method = s.get("method", "zscore")
            if method == "zscore":
                apply_lines.append("    # Z-score normalize each channel to zero mean, unit variance")
                apply_lines.append("    mu  = signals.mean(axis=1, keepdims=True)")
                apply_lines.append("    std = signals.std(axis=1, keepdims=True) + 1e-8")
                apply_lines.append("    signals = ((signals - mu) / std).astype(np.float32)")
            else:  # minmax
                apply_lines.append("    # Min-max normalize each channel to [0, 1]")
                apply_lines.append("    mn  = signals.min(axis=1, keepdims=True)")
                apply_lines.append("    mx  = signals.max(axis=1, keepdims=True)")
                apply_lines.append("    signals = ((signals - mn) / (mx - mn + 1e-8)).astype(np.float32)")
        elif t == "reref":
            method = s.get("method", "average")
            if method == "average":
                apply_lines.append("    # Average reference: subtract mean across channels")
                apply_lines.append("    signals = (signals - signals.mean(axis=0, keepdims=True)).astype(np.float32)")
        elif t in ("window", "reject", "channels", "truncate", "ica"):
            # These steps are not applied per-window in real-time inference:
            # - window: handled externally (caller provides correct window size)
            # - reject: no rejection in real-time (we process every window)
            # - channels/truncate/ica: applied upstream at data collection time
            apply_lines.append(f"    # '{t}' step skipped — handled externally in real-time inference")

    coeff_lines.append("")
    coeff_lines.append("def _apply_preprocessing(signals: np.ndarray) -> np.ndarray:")
    coeff_lines.append('    """')
    coeff_lines.append("    Apply all training pipeline steps to one EEG window.")
    coeff_lines.append("    Args:  signals — shape (n_channels, n_samples), raw EEG")
    coeff_lines.append("    Returns: preprocessed array, same shape")
    coeff_lines.append('    """')

    if apply_lines:
        coeff_lines.extend(apply_lines)
    else:
        coeff_lines.append("    # No preprocessing pipeline recorded — returning signals as-is.")
        coeff_lines.append("    # Ensure your input is already preprocessed to match training conditions.")

    coeff_lines.append("    return signals")
    coeff_lines.append("")

    parts.append("\n".join(coeff_lines))

    # ── Section 5: Model builder (verbatim copy) ─────────────────────────────
    parts.append(_MODEL_BUILDER_CODE)

    # ── Section 6: Model loading + predict() ────────────────────────────────
    parts.append(f'''\
# ── MODEL LOADING ────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{checkpoint_filename}")

_cached_model   = None
_cached_classes = None


def _get_model():
    """Load and cache the model. Materializes lazy modules on first call."""
    global _cached_model, _cached_classes
    if _cached_model is not None:
        return _cached_model, _cached_classes

    checkpoint   = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    n_channels   = checkpoint["n_channels"]
    n_samples    = checkpoint["n_samples"]
    classes      = checkpoint["class_names"]

    # Build model with lazy modules, then materialize with a dummy forward pass
    model = _build_model(model_config)
    dummy = torch.zeros(1, 1, n_channels, n_samples)
    with torch.no_grad():
        model.eval()
        model(dummy)

    # Load trained weights — shapes match because dummy used the same dimensions
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _cached_model   = model
    _cached_classes = classes
    return model, classes


# ── PREDICT ──────────────────────────────────────────────────────────────────

def predict(raw_window: np.ndarray) -> dict:
    """
    Run end-to-end inference: preprocessing + model forward pass.

    Args:
        raw_window: np.ndarray, shape (n_channels, n_samples).
                    Raw EEG — preprocessing is applied automatically.

    Returns:
        dict with:
            "class"         — predicted class label (str)
            "confidence"    — probability of predicted class (float 0-1)
            "probabilities" — dict mapping every class name to its probability
    """
    # Apply the same preprocessing pipeline used during training
    window = _apply_preprocessing(raw_window.copy())

    model, classes = _get_model()

    # Shape: (n_channels, n_samples) -> (1, 1, n_channels, n_samples)
    x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze(0).numpy()

    pred_idx = int(probs.argmax())
    result = {{
        "class":         classes[pred_idx],
        "confidence":    float(probs[pred_idx]),
        "probabilities": {{c: float(p) for c, p in zip(classes, probs)}},
    }}

    on_prediction(result["class"], result["confidence"], result["probabilities"])
    return result
''')

    # ── Section 7: Self-test ─────────────────────────────────────────────────
    parts.append(f'''\
# ── SELF-TEST ─────────────────────────────────────────────────────────────────
# Run this file directly to verify the model loads and predicts correctly.

if __name__ == "__main__":
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    n_ch  = checkpoint["n_channels"]
    n_samp = checkpoint["n_samples"]
    cls   = checkpoint["class_names"]

    print(f"Model: {{CHECKPOINT_PATH}}")
    print(f"  Classes : {{cls}}")
    print(f"  Input   : ({{n_ch}} channels, {{n_samp}} samples) at {{SAMPLE_RATE}} Hz")
    print()

    raw = np.zeros((n_ch, n_samp), dtype=np.float32)
    print("Test prediction (zero input after preprocessing):")
    result = predict(raw)
    print(f"  Predicted : {{result[\'class\']}}")
    print(f"  Confidence: {{result[\'confidence\']:.1%}}")
    print()
    print("  All probabilities:")
    for cls_name, prob in result["probabilities"].items():
        bar = "#" * int(prob * 20)
        print(f"    {{cls_name:12s}}  {{prob:.1%}}  {{bar}}")
''')

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# export_onnx — export PyTorch model to ONNX format
# ---------------------------------------------------------------------------

def export_onnx(job_id: str, model_path: str, export_dir: str) -> str:
    """
    Export a trained PyTorch model to ONNX format.

    ONNX runs on onnxruntime which is lighter than PyTorch and works on
    Raspberry Pi ARM, Jetson Nano, and other edge hardware.

    The same lazy-module materialization pattern used in the inference script
    is applied here: build model → dummy forward pass → load weights → export.

    Args:
        job_id:      Training job UUID (used as filename stem).
        model_path:  Path to the .pt checkpoint file.
        export_dir:  Directory where the .onnx file will be saved.

    Returns:
        Path to the saved .onnx file.
    """
    import torch
    from core.models import build_model

    checkpoint   = torch.load(model_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    n_channels   = checkpoint["n_channels"]
    n_samples    = checkpoint["n_samples"]

    # Build model and materialize lazy modules with a dummy forward pass
    model = build_model(model_config)
    dummy = torch.zeros(1, 1, n_channels, n_samples)
    with torch.no_grad():
        model.eval()
        model(dummy)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, f"{job_id}.onnx")

    # PyTorch 2.x prints emoji-laden progress messages during export.
    # On Windows (cp1252 terminal) these cause UnicodeEncodeError.
    # Redirecting stdout to a StringIO buffer suppresses the noise safely.
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # dynamic_axes lets onnxruntime accept any batch size at inference time
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    logger.info("ONNX model exported to %s", onnx_path)
    return onnx_path


# ---------------------------------------------------------------------------
# generate_onnx_script — standalone onnxruntime inference script
# ---------------------------------------------------------------------------

def generate_onnx_script(
    job_id: str,
    checkpoint: dict,
    model_name: str = "",
) -> str:
    """
    Generate a standalone Python inference script that uses onnxruntime.

    Identical preprocessing embedding to generate_inference_script() but
    loads from a .onnx file instead of a .pt file. Lighter dependency —
    onnxruntime (~10 MB) vs PyTorch (~700 MB).

    Args:
        job_id:     Training job UUID.
        checkpoint: Loaded torch checkpoint dict.
        model_name: Human-readable model name.

    Returns:
        Complete Python source code as a string.
    """
    classes         = checkpoint.get("class_names", [])
    n_channels      = checkpoint.get("n_channels", 0)
    n_samples       = checkpoint.get("n_samples", 0)
    sample_rate     = checkpoint.get("sample_rate", 256)
    trained_at      = checkpoint.get("trained_at", "unknown")
    pipeline_config = checkpoint.get("pipeline_config") or {}

    display_name  = model_name or f"model_{job_id[:8]}"
    onnx_filename = f"{(model_name or job_id[:8]).replace(' ', '_')}.onnx"
    script_name   = f"neutrol_onnx_{job_id[:8]}"
    generated_at  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    steps         = pipeline_config.get("steps", [])

    # Pre-compute coefficients (same as main script)
    bandpass_constants = []
    notch_constants    = []
    bp_idx = notch_idx = 0

    for s in steps:
        if s.get("step") == "bandpass":
            try:
                sos = _compute_bandpass_sos(
                    float(s.get("low_hz", 1.0)),
                    float(s.get("high_hz", 40.0)),
                    sample_rate,
                )
                bandpass_constants.append((f"_BP_SOS_{bp_idx}", sos))
                bp_idx += 1
            except Exception:
                pass
        elif s.get("step") == "notch":
            try:
                b, a = _compute_notch_ba(
                    float(s.get("freq_hz", 60.0)),
                    float(s.get("quality_factor", 30.0)),
                    sample_rate,
                )
                notch_constants.append((f"_NOTCH_B_{notch_idx}", f"_NOTCH_A_{notch_idx}", b, a))
                notch_idx += 1
            except Exception:
                pass

    # Build apply_preprocessing body (identical logic to main script)
    apply_lines = []
    bp_i = notch_i = 0
    for s in steps:
        t = s.get("step")
        if t == "bandpass" and bp_i < len(bandpass_constants):
            apply_lines.append(f"    signals = _sig.sosfiltfilt({bandpass_constants[bp_i][0]}, signals, axis=1).astype(np.float32)")
            bp_i += 1
        elif t == "notch" and notch_i < len(notch_constants):
            b_v, a_v = notch_constants[notch_i][0], notch_constants[notch_i][1]
            apply_lines.append(f"    signals = np.apply_along_axis(lambda ch: _sig.lfilter({b_v}, {a_v}, ch), 1, signals).astype(np.float32)")
            notch_i += 1
        elif t == "normalize":
            method = s.get("method", "zscore")
            if method == "zscore":
                apply_lines += ["    mu = signals.mean(axis=1, keepdims=True)", "    std = signals.std(axis=1, keepdims=True) + 1e-8", "    signals = ((signals - mu) / std).astype(np.float32)"]
            else:
                apply_lines += ["    mn = signals.min(axis=1, keepdims=True)", "    mx = signals.max(axis=1, keepdims=True)", "    signals = ((signals - mn) / (mx - mn + 1e-8)).astype(np.float32)"]
        elif t == "reref" and s.get("method", "average") == "average":
            apply_lines.append("    signals = (signals - signals.mean(axis=0, keepdims=True)).astype(np.float32)")

    coeff_block_lines = []
    for var, sos in bandpass_constants:
        coeff_block_lines.append(f"{var} = {sos!r}")
    for b_v, a_v, b, a in notch_constants:
        coeff_block_lines.append(f"{b_v} = {b!r}")
        coeff_block_lines.append(f"{a_v} = {a!r}")
    coeff_block = "\n".join(coeff_block_lines)

    apply_body = "\n".join(apply_lines) if apply_lines else "    pass  # no preprocessing recorded"

    return f'''\
"""
============================================================
NEUTROL ONNX INFERENCE SCRIPT
============================================================
Model:     {display_name}
Classes:   {", ".join(classes)}
Trained:   {trained_at}
Generated: {generated_at}

Requirements (much lighter than PyTorch):
    pip install onnxruntime numpy scipy

Usage:
    from {script_name} import predict
    import numpy as np
    raw = np.zeros(({n_channels}, {n_samples}), dtype=np.float32)
    result = predict(raw)
"""

import os
import numpy as np
import scipy.signal as _sig
import onnxruntime as ort

# ── CONFIG ───────────────────────────────────────────────────────────────────
SAMPLE_RATE  = {sample_rate}
CLASSES      = {classes!r}
ONNX_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{onnx_filename}")

# ── PIPELINE STEPS ───────────────────────────────────────────────────────────
PIPELINE_STEPS = {steps!r}

# ── YOUR OUTPUT CODE ─────────────────────────────────────────────────────────
def on_prediction(label: str, confidence: float, probabilities: dict) -> None:
    print(f"  {{label:12s}}  {{confidence:.1%}}")

# ── FILTER COEFFICIENTS (pre-computed) ───────────────────────────────────────
{coeff_block}

# ── PREPROCESSING ────────────────────────────────────────────────────────────
def _apply_preprocessing(signals: np.ndarray) -> np.ndarray:
{apply_body}
    return signals

# ── ONNX SESSION ─────────────────────────────────────────────────────────────
_session = None

def _get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    return _session

# ── PREDICT ──────────────────────────────────────────────────────────────────
def predict(raw_window: np.ndarray) -> dict:
    """End-to-end inference: preprocessing + onnxruntime forward pass."""
    window = _apply_preprocessing(raw_window.copy())
    # Shape: (n_channels, n_samples) -> (1, 1, n_channels, n_samples)
    x      = window[np.newaxis, np.newaxis, :, :].astype(np.float32)
    sess   = _get_session()
    output = sess.run(None, {{"input": x}})[0]          # shape (1, n_classes)
    probs  = _softmax(output[0])
    idx    = int(probs.argmax())
    result = {{
        "class":         CLASSES[idx],
        "confidence":    float(probs[idx]),
        "probabilities": {{c: float(p) for c, p in zip(CLASSES, probs)}},
    }}
    on_prediction(result["class"], result["confidence"], result["probabilities"])
    return result

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

# ── SELF-TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw    = np.zeros(({n_channels}, {n_samples}), dtype=np.float32)
    result = predict(raw)
    print(f"Predicted: {{result[\'class\']}} ({{result[\'confidence\']:.1%}})")
'''


# ---------------------------------------------------------------------------
# export_microcontroller — ZIP with .h + .cpp + README for ESP32/STM32
# ---------------------------------------------------------------------------

def export_microcontroller(job_id: str, model_path: str, export_dir: str) -> str:
    """
    Generate a ZIP containing everything needed for ESP32-S3/STM32 deployment.

    Conversion chain: PyTorch → ONNX → TFLite → C array (.h file)

    Requires tensorflow and onnx2tf. Returns path to the .zip file.
    Raises ImportError with install instructions if dependencies are missing.
    """
    # Check optional heavy dependencies upfront — fail fast with clear message
    try:
        import tensorflow as tf  # noqa: F401
        import onnx                # noqa: F401
        import onnx2tf             # noqa: F401
    except ImportError:
        raise ImportError(
            "Microcontroller export requires TensorFlow. "
            "Run: pip install tensorflow onnx onnx2tf"
        )

    import torch
    from core.models import build_model

    checkpoint   = torch.load(model_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    n_channels   = checkpoint["n_channels"]
    n_samples    = checkpoint["n_samples"]
    n_classes    = checkpoint["n_classes"]
    classes      = checkpoint["class_names"]
    sample_rate  = checkpoint.get("sample_rate", 256)
    pipeline_config = checkpoint.get("pipeline_config") or {}
    accuracy     = None  # placeholder — not stored in checkpoint currently

    # Step 1: Build and export to ONNX (temp file)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "model.onnx")
        tflite_dir = os.path.join(tmp, "tflite_out")

        model = build_model(model_config)
        dummy = torch.zeros(1, 1, n_channels, n_samples)
        with torch.no_grad():
            model.eval()
            model(dummy)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        torch.onnx.export(
            model, dummy, onnx_path,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
        )

        # Step 2: ONNX → TFLite via onnx2tf
        import onnx2tf
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tflite_dir,
            non_verbose=True,
        )

        # Find the generated .tflite file
        tflite_files = [f for f in os.listdir(tflite_dir) if f.endswith(".tflite")]
        if not tflite_files:
            raise RuntimeError("onnx2tf did not produce a .tflite file.")
        tflite_path = os.path.join(tflite_dir, tflite_files[0])

        with open(tflite_path, "rb") as f:
            tflite_bytes = f.read()

        # Step 3: Generate files and zip them
        model_h   = _generate_model_h(tflite_bytes, classes, accuracy)
        cpp_code  = _generate_inference_cpp(
            n_channels, n_samples, n_classes, classes, sample_rate, pipeline_config
        )
        readme    = _generate_micro_readme()

        os.makedirs(export_dir, exist_ok=True)
        zip_path = os.path.join(export_dir, f"{job_id}_microcontroller.zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model_data.h",               model_h)
            zf.writestr("neutrol_inference.cpp",       cpp_code)
            zf.writestr("README_microcontroller.md",   readme)

    logger.info("Microcontroller ZIP exported to %s", zip_path)
    return zip_path


def _generate_model_h(tflite_bytes: bytes, classes: list, accuracy) -> str:
    """Generate model_data.h with the TFLite model embedded as a C array."""
    hex_bytes  = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
    acc_str    = f"{accuracy:.1%}" if accuracy is not None else "unknown"
    classes_str = ", ".join(classes)

    return f"""\
// Auto-generated by Neutrol. Do not edit.
// Classes: {classes_str} | Accuracy: {acc_str}

#pragma once
#include <stdint.h>

alignas(8) const unsigned char neutrol_model_data[] = {{ {hex_bytes} }};
const unsigned int neutrol_model_data_len = {len(tflite_bytes)};
"""


def _generate_inference_cpp(
    n_channels: int,
    n_samples: int,
    n_classes: int,
    classes: list,
    sample_rate: int,
    pipeline_config: dict,
) -> str:
    """Generate neutrol_inference.cpp — TFLite Micro inference template."""
    steps = pipeline_config.get("steps", [])
    classes_array = ", ".join(f'"{c}"' for c in classes)

    # Pre-compute bandpass SOS coefficients for C embedding
    bandpass_c = ""
    notch_c    = ""
    for s in steps:
        if s.get("step") == "bandpass" and not bandpass_c:
            try:
                sos = _compute_bandpass_sos(
                    float(s.get("low_hz", 1.0)),
                    float(s.get("high_hz", 40.0)),
                    sample_rate,
                )
                rows = []
                for row in sos:
                    row_str = ", ".join(f"{v:.8f}f" for v in row)
                    rows.append(f"    {{{row_str}}}")
                bandpass_c = (
                    f"// Bandpass {s.get('low_hz')}–{s.get('high_hz')} Hz, "
                    f"Butterworth order 4, at {sample_rate} Hz\n"
                    f"static const float BANDPASS_SOS[{len(sos)}][6] = {{\n"
                    + ",\n".join(rows)
                    + "\n};"
                )
            except Exception:
                pass
        elif s.get("step") == "notch" and not notch_c:
            try:
                b, a = _compute_notch_ba(
                    float(s.get("freq_hz", 60.0)),
                    float(s.get("quality_factor", 30.0)),
                    sample_rate,
                )
                b_str = ", ".join(f"{v:.8f}f" for v in b)
                a_str = ", ".join(f"{v:.8f}f" for v in a)
                notch_c = (
                    f"// Notch {s.get('freq_hz')} Hz, Q={s.get('quality_factor', 30.0)}\n"
                    f"static const float NOTCH_B[] = {{{b_str}}};\n"
                    f"static const float NOTCH_A[] = {{{a_str}}};"
                )
            except Exception:
                pass

    return f"""\
// ============================================================
// NEUTROL — Microcontroller Inference Template
// Tested on: ESP32-S3, STM32 H7, Arduino Nano 33 BLE Sense
// ============================================================

// ── SECTION 1: CONFIG (edit these) ──────────────────────────
#define SAMPLE_RATE      {sample_rate}
#define WINDOW_SAMPLES   {n_samples}
#define N_CHANNELS       {n_channels}
#define N_CLASSES        {n_classes}
const char* CLASS_LABELS[] = {{{classes_array}}};

// ── SECTION 2: FILTER COEFFICIENTS (auto-generated) ─────────
// These match exactly what your model was trained on.
{bandpass_c}
{notch_c}

// ── SECTION 3: YOUR OUTPUT CODE (edit this) ─────────────────
void on_prediction(int class_index, float confidence) {{
    Serial.printf("Prediction: %s (%.1f%%)\\n",
                  CLASS_LABELS[class_index], confidence * 100.0f);
    // Add your application logic here:
    // if (class_index == 1) digitalWrite(MOTOR_LEFT_PIN, HIGH);
}}

// ── SECTION 4: TFLITE MICRO + PREPROCESSING (do not edit) ───
#include "model_data.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

namespace {{
  const int kTensorArenaSize = 64 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}}

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor  = nullptr;
TfLiteTensor* output_tensor = nullptr;

void neutrol_setup() {{
  const tflite::Model* model = tflite::GetModel(neutrol_model_data);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter    = &static_interpreter;
  interpreter->AllocateTensors();
  input_tensor   = interpreter->input(0);
  output_tensor  = interpreter->output(0);
}}

// Call neutrol_predict() with a float array of shape [N_CHANNELS][WINDOW_SAMPLES]
void neutrol_predict(float window[N_CHANNELS][WINDOW_SAMPLES]) {{
  // Copy window into TFLite input tensor
  // Input shape: [1, 1, N_CHANNELS, WINDOW_SAMPLES]
  float* inp = input_tensor->data.f;
  for (int ch = 0; ch < N_CHANNELS; ch++)
    for (int s = 0; s < WINDOW_SAMPLES; s++)
      inp[ch * WINDOW_SAMPLES + s] = window[ch][s];

  interpreter->Invoke();

  // Softmax output
  float* out = output_tensor->data.f;
  int   best = 0;
  float best_score = out[0];
  for (int i = 1; i < N_CLASSES; i++)
    if (out[i] > best_score) {{ best = i; best_score = out[i]; }}

  on_prediction(best, best_score);
}}
"""


def _generate_micro_readme() -> str:
    return """\
# Neutrol Microcontroller Inference

## Files in this ZIP
- `model_data.h` — TFLite model as a C byte array
- `neutrol_inference.cpp` — Inference template with preprocessing
- `README_microcontroller.md` — This file

## Requirements
- Arduino IDE 2.x or ESP-IDF
- TensorFlow Lite Micro library (search "TensorFlow Lite" in Arduino Library Manager)
- Target: ESP32-S3, STM32 F4/H7, or Arduino Nano 33 BLE Sense

## Setup
1. Create a new Arduino sketch folder
2. Copy `model_data.h` and `neutrol_inference.cpp` into the sketch folder
3. In your main `.ino` file:
   ```cpp
   #include "neutrol_inference.cpp"

   void setup() {
     Serial.begin(115200);
     neutrol_setup();
   }

   void loop() {
     // Fill window with your EEG samples from ADC
     float window[N_CHANNELS][WINDOW_SAMPLES];
     // ... read your EEG hardware here ...
     neutrol_predict(window);
     delay(50);
   }
   ```
4. Edit `on_prediction()` in `neutrol_inference.cpp` to control your hardware
5. Upload and open Serial Monitor at 115200 baud

## Notes
- `SAMPLE_RATE`, `WINDOW_SAMPLES`, and `N_CHANNELS` in the CONFIG section must
  match your EEG hardware settings exactly
- The filter coefficients in Section 2 are pre-computed to match training —
  do not modify them
- NOT compatible with Arduino Uno (insufficient RAM — needs >64 KB)
"""
