"""
models.py — EEG neural network models: hardcoded EEGNet + dynamic builder.

Dynamic builder:
  build_model(config) reads a 'layers' array from the ModelConfig and constructs
  a PyTorch nn.Sequential at runtime. This lets users design their own architecture
  from the UI without writing Python.

  Uses PyTorch Lazy modules (LazyConv2d, LazyLinear) which infer their input size
  on the first forward pass — so we never need to track channel counts manually
  between layers.

Input convention: (batch_size, 1, n_channels, n_samples)
  The '1' is a single input feature map (like grayscale in image CNNs).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, type] = {
    "ELU":       nn.ELU,
    "ReLU":      nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh":      nn.Tanh,
    "Sigmoid":   nn.Sigmoid,
    "GELU":      nn.GELU,
}


def get_activation(name: str) -> nn.Module:
    """Return a PyTorch activation module by name string."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Valid: {sorted(_ACTIVATIONS.keys())}")
    return _ACTIVATIONS[name]()


# ---------------------------------------------------------------------------
# Layer type schema — returned by GET /api/train/layer-types
# ---------------------------------------------------------------------------

LAYER_TYPES_SCHEMA: list[dict] = [
    {
        "type": "Conv2d",
        "description": "2D convolution — learns temporal and spatial filters",
        "supports_activation": True,
        "parameters": [
            {"name": "out_channels", "type": "int",        "default": 16,      "description": "Number of output feature maps"},
            {"name": "kernel_size",  "type": "list[int]",  "default": [1, 32], "description": "[height, width] — [1, N] for temporal-only filters"},
            {"name": "padding",      "type": "str_or_int", "default": "same",  "description": "'same' keeps output size unchanged"},
            {"name": "batch_norm",   "type": "bool",       "default": True,    "description": "Add batch normalization after conv"},
        ],
    },
    {
        "type": "DepthwiseConv2d",
        "description": "Per-channel convolution — learns channel-specific patterns without mixing channels",
        "supports_activation": True,
        "parameters": [
            {"name": "depth_multiplier", "type": "int",       "default": 2,                "description": "Output channels = input x depth_multiplier"},
            {"name": "kernel_size",      "type": "list[int]", "default": ["n_channels", 1], "description": "Use ['n_channels', 1] to span all EEG channels"},
            {"name": "batch_norm",       "type": "bool",      "default": True,             "description": "Add batch normalization after conv"},
        ],
    },
    {
        "type": "AvgPool2d",
        "description": "Average pooling — downsamples by averaging blocks (preferred for EEG)",
        "supports_activation": False,
        "parameters": [
            {"name": "kernel_size", "type": "list[int]", "default": [1, 4], "description": "[height, width] — [1, 4] downsamples time by 4x"},
        ],
    },
    {
        "type": "MaxPool2d",
        "description": "Max pooling — downsamples by taking the peak value in each block",
        "supports_activation": False,
        "parameters": [
            {"name": "kernel_size", "type": "list[int]", "default": [1, 4], "description": "[height, width]"},
        ],
    },
    {
        "type": "Dropout",
        "description": "Randomly zeros outputs during training — prevents memorizing the training set",
        "supports_activation": False,
        "parameters": [
            {"name": "rate", "type": "float", "default": 0.5, "description": "Fraction to zero out (0.0-1.0)"},
        ],
    },
    {
        "type": "BatchNorm2d",
        "description": "Normalizes layer outputs — stabilizes training and allows higher learning rates",
        "supports_activation": False,
        "parameters": [],
    },
    {
        "type": "Flatten",
        "description": "Converts 2D feature maps to a 1D vector — required before a Linear layer",
        "supports_activation": False,
        "parameters": [],
    },
    {
        "type": "Linear",
        "description": "Fully connected layer. Use out_features='n_classes' for the final output layer",
        "supports_activation": False,
        "parameters": [
            {"name": "out_features", "type": "int_or_string", "default": "n_classes",
             "description": "Number of outputs. Use 'n_classes' for the final classification layer"},
        ],
    },
    {
        "type": "LSTM",
        "description": "Recurrent layer for temporal patterns. Place after Flatten",
        "supports_activation": False,
        "parameters": [
            {"name": "hidden_size", "type": "int",   "default": 64,  "description": "Features in the hidden state"},
            {"name": "num_layers",  "type": "int",   "default": 1,   "description": "Stacked LSTM layers"},
            {"name": "dropout",     "type": "float", "default": 0.0, "description": "Dropout between layers (only if num_layers > 1)"},
        ],
    },
    {
        "type": "CustomLayer",
        "description": "User-defined PyTorch layer — paste your nn.Module class code",
        "supports_activation": False,
        "parameters": [
            {
                "name": "code",
                "type": "code",
                "default": (
                    "class CustomLayer(nn.Module):\n"
                    "    def __init__(self):\n"
                    "        super().__init__()\n"
                    "        # Add sub-layers here, e.g.:\n"
                    "        # self.fc = nn.LazyLinear(64)\n\n"
                    "    def forward(self, x):\n"
                    "        # x shape depends on where this layer is placed.\n"
                    "        # Return a tensor of any shape your next layer expects.\n"
                    "        return x\n"
                ),
                "description": "Python class named 'CustomLayer' extending nn.Module",
            },
        ],
    },
]

ACTIVATION_OPTIONS: list[dict] = [
    {"name": "ELU",       "description": "Default for EEG CNNs — smooth, allows small negative values"},
    {"name": "ReLU",      "description": "Most common — sets negatives to zero"},
    {"name": "LeakyReLU", "description": "Like ReLU but passes small negatives — prevents dead neurons"},
    {"name": "Tanh",      "description": "Outputs between -1 and 1 — good for recurrent layers"},
    {"name": "Sigmoid",   "description": "Outputs between 0 and 1 — binary output only"},
    {"name": "GELU",      "description": "Smooth ReLU variant — used in deep networks"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_kernel(kernel_size, n_channels: int):
    """Replace the 'n_channels' keyword with the actual channel count."""
    if isinstance(kernel_size, list):
        return tuple(n_channels if k == "n_channels" else int(k) for k in kernel_size)
    if kernel_size == "n_channels":
        return n_channels
    return int(kernel_size)


# ---------------------------------------------------------------------------
# Lazy custom modules
# ---------------------------------------------------------------------------

class _LazyDepthwiseConv2d(nn.Module):
    """
    Depthwise 2D convolution that lazily infers in_channels on first forward pass.

    Depthwise conv requires groups=in_channels which is unknown at construction
    time. This wrapper materializes the inner Conv2d on the first forward call.
    """

    def __init__(self, depth_multiplier: int, kernel_size, bias: bool = False) -> None:
        super().__init__()
        self.depth_multiplier = depth_multiplier
        self.kernel_size      = kernel_size
        self.bias             = bias
        self._conv: nn.Conv2d | None = None

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
    """
    LSTM that accepts flat 1D input from a preceding Flatten layer.

    Input:  (batch, features) from Flatten
    Output: (batch, hidden_size) — last timestep only
    """

    def __init__(self, hidden_size: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self._dropout    = dropout
        self._lstm: nn.LSTM | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
        if self._lstm is None:
            lstm_drop  = self._dropout if self.num_layers > 1 else 0.0
            self._lstm = nn.LSTM(
                x.shape[2], self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=lstm_drop,
            ).to(x.device)
        out, _ = self._lstm(x)
        return out[:, -1, :]  # take last timestep: (batch, hidden_size)


# ---------------------------------------------------------------------------
# _build_layer_modules
# ---------------------------------------------------------------------------

def _build_layer_modules(layer_cfg: dict, n_channels: int, n_classes: int) -> list[nn.Module]:
    """
    Translate one layer config dict into a list of PyTorch modules.

    A single layer card may expand into multiple modules — e.g. Conv2d with
    batch_norm=True becomes [LazyConv2d, LazyBatchNorm2d, ELU].
    """
    t       = layer_cfg["type"]
    modules: list[nn.Module] = []

    if t == "Conv2d":
        out_ch  = int(layer_cfg["out_channels"])
        kernel  = _resolve_kernel(layer_cfg["kernel_size"], n_channels)
        padding = layer_cfg.get("padding", 0)
        modules.append(nn.LazyConv2d(out_ch, kernel, padding=padding, bias=False))
        if layer_cfg.get("batch_norm", False):
            modules.append(nn.LazyBatchNorm2d())
        if "activation" in layer_cfg:
            modules.append(get_activation(layer_cfg["activation"]))

    elif t == "DepthwiseConv2d":
        dm     = int(layer_cfg.get("depth_multiplier", 1))
        kernel = _resolve_kernel(layer_cfg["kernel_size"], n_channels)
        modules.append(_LazyDepthwiseConv2d(dm, kernel, bias=False))
        if layer_cfg.get("batch_norm", False):
            modules.append(nn.LazyBatchNorm2d())
        if "activation" in layer_cfg:
            modules.append(get_activation(layer_cfg["activation"]))

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

    elif t == "CustomLayer":
        # Execute the user's class definition in a sandboxed namespace.
        # Only torch and nn are exposed — no file I/O or network access.
        code = layer_cfg.get("code", "")
        namespace: dict = {"nn": nn, "torch": torch}
        try:
            exec(code, namespace)  # noqa: S102  (exec is intentional — user-defined layer)
        except Exception as exc:
            raise ValueError(f"CustomLayer code error: {exc}") from exc

        cls = namespace.get("CustomLayer")
        if cls is None or not isinstance(cls, type):
            raise ValueError(
                "CustomLayer code must define a class named 'CustomLayer'."
            )
        if not issubclass(cls, nn.Module):
            raise ValueError(
                "CustomLayer class must extend nn.Module "
                "(e.g. 'class CustomLayer(nn.Module):')."
            )
        modules.append(cls())

    else:
        raise ValueError(
            f"Unknown layer type: '{t}'. "
            f"Valid: {[lt['type'] for lt in LAYER_TYPES_SCHEMA]}"
        )

    return modules


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """
    Build a PyTorch model dynamically from a ModelConfig dict.

    Reads config['layers'], resolves special keywords ('n_channels', 'n_classes'),
    and returns an nn.Sequential. Lazy modules materialize on first forward pass.

    Args:
        config: dict with 'layers', 'input_channels', 'input_samples', 'classes'.

    Returns:
        nn.Sequential ready for training.
    """
    n_channels = int(config["input_channels"])
    n_classes  = len(config["classes"])

    all_modules: list[nn.Module] = []
    for layer_cfg in config["layers"]:
        all_modules.extend(_build_layer_modules(layer_cfg, n_channels, n_classes))

    return nn.Sequential(*all_modules)


# ---------------------------------------------------------------------------
# validate_model_config
# ---------------------------------------------------------------------------

def validate_model_config(config: dict) -> list[str]:
    """
    Check a ModelConfig for common mistakes.

    Returns a list of human-readable error strings. Empty list = valid.
    """
    errors: list[str] = []
    layers = config.get("layers", [])

    if not layers:
        errors.append("Architecture must have at least one layer.")
        return errors

    last = layers[-1]
    if last.get("type") != "Linear":
        errors.append(
            f"The last layer must be Linear (produces class scores). "
            f"Currently it is '{last.get('type')}'."
        )
    elif str(last.get("out_features")) != "n_classes":
        errors.append(
            "The last Linear layer must have out_features='n_classes'. "
            f"Currently: '{last.get('out_features')}'."
        )

    NEEDS_2D = {"Conv2d", "DepthwiseConv2d", "AvgPool2d", "MaxPool2d", "BatchNorm2d"}
    seen_flatten = False
    for i, layer in enumerate(layers):
        t = layer.get("type", "")
        if t == "Flatten":
            seen_flatten = True
        elif seen_flatten and t in NEEDS_2D:
            errors.append(
                f"Layer {i + 1} ({t}) appears after Flatten but needs 2D input. "
                "Move it before the Flatten layer."
            )

    for i, layer in enumerate(layers):
        if layer.get("type") == "Dropout":
            rate = layer.get("rate", 0.5)
            try:
                if not (0.0 <= float(rate) <= 1.0):
                    errors.append(f"Layer {i + 1} Dropout: rate must be 0.0-1.0, got {rate}.")
            except (TypeError, ValueError):
                errors.append(f"Layer {i + 1} Dropout: rate must be a number, got '{rate}'.")

    REQUIRED: dict[str, list[str]] = {
        "Conv2d":          ["out_channels", "kernel_size"],
        "DepthwiseConv2d": ["kernel_size"],
        "AvgPool2d":       ["kernel_size"],
        "MaxPool2d":       ["kernel_size"],
        "Linear":          ["out_features"],
    }
    for i, layer in enumerate(layers):
        t = layer.get("type", "")
        for param in REQUIRED.get(t, []):
            if param not in layer:
                errors.append(f"Layer {i + 1} ({t}): missing required parameter '{param}'.")

    return errors


# ---------------------------------------------------------------------------
# estimate_output_shape
# ---------------------------------------------------------------------------

def estimate_output_shape(config: dict) -> list:
    """
    Run a zero tensor through the model layer-by-layer and record the output
    shape after each user-defined layer.

    Returns a list aligned with config['layers']. Each entry is a list of ints
    or None if that layer caused an error (stops at first failure).
    """
    n_channels = int(config["input_channels"])
    n_classes  = len(config["classes"])
    n_samples  = int(config["input_samples"])

    x      = torch.zeros(1, 1, n_channels, n_samples)
    shapes: list = []

    for i, layer_cfg in enumerate(config["layers"]):
        try:
            modules = _build_layer_modules(layer_cfg, n_channels, n_classes)
            seq     = nn.Sequential(*modules)
            seq.eval()
            with torch.no_grad():
                x = seq(x)
            shapes.append(list(x.shape))
        except Exception as exc:
            logger.warning("Shape estimation failed at layer %d (%s): %s",
                           i, layer_cfg.get("type"), exc)
            shapes.append(None)
            break

    return shapes


# ---------------------------------------------------------------------------
# EEGNet — original hardcoded architecture (backward compatibility)
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    """
    Original EEGNet-inspired CNN from Session 4.
    Kept as fallback when no 'layers' key is present in the training config.
    New training runs use build_model() instead.
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int) -> None:
        super().__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 32), padding="same", bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.5),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.5),
        )
        self.flatten_size = 32 * (n_samples // 32)
        self.classifier   = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
