"""
model/network.py
----------------
Self-pruning network for CIFAR-10 image classification.

Architecture overview
=====================
Input  : 3 x 32 x 32 CIFAR-10 image
         |  Convolutional feature extractor (fixed, not prunable)
         v
  Conv(3->64) + BN + ReLU
  Conv(64->128) + BN + ReLU  + MaxPool(2)   --> 128 x 16 x 16
  Conv(128->256) + BN + ReLU + MaxPool(2)   --> 256 x  8 x  8
         |  Flatten --> 16384-dim vector
         v
  PrunableLinear(16384 -> 512) + ReLU + Dropout
  PrunableLinear(512   -> 10)  logits

Only the classifier (PrunableLinear) layers carry learnable gate
parameters and participate in the sparsity-learning mechanism.
The convolutional feature extractor uses standard nn layers so its
weights are optimised purely by the CE loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List

from model.prunable_linear import PrunableLinear


class SelfPruningNet(nn.Module):
    """
    CNN feature extractor + prunable linear classifier for CIFAR-10.

    The convolutional backbone is a standard (non-prunable) feature
    extractor that provides spatial representations.  The classifier
    head uses ``PrunableLinear`` layers so every weight connection
    participates in the sparsity-learning mechanism.

    Parameters
    ----------
    in_channels  : int   - input image channels (3 for CIFAR-10).
    image_size   : int   - spatial size of input image (32 for CIFAR-10).
    num_classes  : int   - number of output classes (10 for CIFAR-10).
    dropout_rate : float - dropout probability in the classifier head.
    """

    # Feature-map size after the two MaxPool(2) operations: 32 // 2 // 2 = 8
    _POOL_OUT = 8
    _CNN_CHANNELS = 256  # output channels of the last conv block

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        # kept for API compatibility but no longer used
        hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.image_size  = image_size
        self.num_classes = num_classes

        # ── Convolutional feature extractor (not prunable) ─────────────
        # Convolutional feature extractor (not prunable) + prunable classifier
        self.features = nn.Sequential(
            # Block 1: 3 x 32 x 32  -->  64 x 32 x 32
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 2: 64 x 32 x 32  -->  128 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 128 x 16 x 16  -->  256 x 8 x 8
            nn.Conv2d(128, self._CNN_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(self._CNN_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Flattened feature dimension after the two MaxPool(2) layers
        flat_dim = self._CNN_CHANNELS * self._POOL_OUT * self._POOL_OUT  # 16384

        # ── Prunable classifier head ───────────────────────────────────
        self.classifier = nn.Sequential(
            PrunableLinear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            PrunableLinear(512, num_classes),
        )

    # ──────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) image batch.

        Returns:
            logits: (N, num_classes) unnormalised scores.
        """
        # Convolutional feature extraction (spatial → descriptor)
        x = self.features(x)           # (N, 256, 8, 8)
        x = x.flatten(start_dim=1)     # (N, 16384)
        return self.classifier(x)      # (N, 10)

    # ──────────────────────────────────────────────────────────────────
    # Bonus — hard threshold & model export helpers
    # ──────────────────────────────────────────────────────────────────

    def apply_hard_pruning(self, threshold: float = 1e-2) -> None:
        """
        Walk all PrunableLinear sub-modules and permanently zero their
        gate scores that fall below ``threshold``. Call this after
        training to materialise the learned sparse mask.
        """
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                module.apply_hard_threshold(threshold)

    def export_torchscript(self, path: str) -> None:
        """
        Traces the model to TorchScript for deployment.

        Args:
            path: File path for the serialised ``.pt`` model.
        """
        self.eval()
        device = next(self.parameters()).device
        dummy = torch.zeros(1, self.in_channels, self.image_size, self.image_size, device=device)
        scripted = torch.jit.trace(self, dummy)
        scripted.save(path)

    def count_parameters(self) -> dict:
        """Return a breakdown of total vs. prunable parameters."""
        total = sum(p.numel() for p in self.parameters())
        gate_params = sum(
            m.gate_scores.numel()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        )
        # Count non-pruned (active) gate connections
        active_weights = sum(
            (m.get_gates() >= 0.01).sum().item()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        )
        total_gate_weights = sum(
            m.gate_scores.numel()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        )
        return {
            "total_params":         total,
            "gate_score_params":    gate_params,
            "weight_bias_params":   total - gate_params,
            "active_gate_weights":  int(active_weights),
            "pruned_gate_weights":  int(total_gate_weights - active_weights),
        }