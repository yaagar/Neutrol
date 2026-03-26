"""
models.py — EEGNet-inspired CNN for EEG classification.

Architecture defined exactly as in CLAUDE.md. Do not change layer sizes or order.

Input shape convention: (batch_size, 1, n_channels, n_samples)
  - The "1" is a single input feature map (like grayscale in image CNNs)
  - n_channels is the spatial dimension (electrodes)
  - n_samples is the temporal dimension (time steps in the window)

Why this architecture?
  The three conv blocks learn progressively higher-level features:
  1. Temporal block: learns which frequencies are present (like a learned bandpass filter)
  2. Spatial block: learns which electrodes activate together (spatial connectivity)
  3. Separable block: combines and refines both types of features
  Average pooling (not max pooling) is used because EEG amplitude distributions are
  smoother than image edges — average pooling preserves this statistical structure better.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet-inspired convolutional neural network for EEG window classification.

    Args:
        n_channels: Number of EEG channels (electrodes), e.g. 8.
        n_samples:  Number of time samples per window, e.g. 256 (1s at 256 Hz).
        n_classes:  Number of output classes, e.g. 3 (rest, left, right).
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Block 1 — Temporal convolution
        # Kernel (1, 32): slides along the time axis, one channel at a time.
        # Learns to detect frequency-band patterns (like a learned bandpass bank).
        # padding='same' keeps n_samples unchanged after this layer.
        # bias=False because BatchNorm has its own learnable shift parameter.
        # Output: (batch, 16, n_channels, n_samples)
        # ------------------------------------------------------------------
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 32), padding="same", bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )

        # ------------------------------------------------------------------
        # Block 2 — Spatial (depthwise) convolution
        # Kernel (n_channels, 1): mixes information across ALL channels at once.
        # This learns which electrode pairs co-activate — spatial connectivity.
        # AvgPool(1, 4) downsamples time by 4×: n_samples → n_samples/4.
        # Dropout(0.5) regularises heavily — EEG datasets are small.
        # Output: (batch, 32, 1, n_samples/4)
        # ------------------------------------------------------------------
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.5),
        )

        # ------------------------------------------------------------------
        # Block 3 — Separable temporal convolution
        # Now operating on (1 × n_samples/4) feature maps — much cheaper than block 1.
        # Kernel (1, 16): refines temporal patterns at the downsampled resolution.
        # AvgPool(1, 8) downsamples by 8×: n_samples/4 → n_samples/32.
        # Output: (batch, 32, 1, n_samples/32)
        # ------------------------------------------------------------------
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.5),
        )

        # ------------------------------------------------------------------
        # Classifier
        # After flattening: 32 feature maps × (n_samples // 32) time steps.
        # For a 1s window at 256 Hz: 32 × 8 = 256 features.
        # CrossEntropyLoss expects raw logits (no softmax here).
        # ------------------------------------------------------------------
        self.flatten_size = 32 * (n_samples // 32)
        self.classifier = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, n_channels, n_samples).

        Returns:
            Class logits of shape (batch_size, n_classes).
            Pass through CrossEntropyLoss during training — do NOT apply softmax here.
        """
        x = self.temporal_conv(x)    # (batch, 16, n_channels, n_samples)
        x = self.spatial_conv(x)     # (batch, 32, 1, n_samples // 4)
        x = self.separable_conv(x)   # (batch, 32, 1, n_samples // 32)
        x = x.flatten(1)             # (batch, 32 * n_samples // 32)
        x = self.classifier(x)       # (batch, n_classes)
        return x
