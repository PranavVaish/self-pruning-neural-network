"""
model/prunable_linear.py
------------------------
Custom linear layer with learnable gate parameters for dynamic weight pruning.

Design goals
============
* Gradients flow through both ``weight`` and ``gate_scores`` via standard
  autograd — no custom backward pass is required because element-wise
  multiplication and ``torch.sigmoid`` are both differentiable.
* ``gate_scores`` are raw (un-bounded) scalars stored as ``nn.Parameter``
  so they receive gradient updates from any standard optimizer.
* The sigmoid transformation maps raw scores to [0, 1], giving a soft
  gate that drives towards either fully-open (1) or fully-closed (0).
* The L1 sparsity penalty on the *sigmoid* outputs (applied externally in
  the loss module) exerts a constant-magnitude gradient on each gate score,
  steadily pushing inactive gates to exactly zero.

                ┌──────────────┐
  gate_scores ──► sigmoid(·)  ├──► gates  ──┐
                └──────────────┘            │  element-wise ×
                                            ▼
  weight ─────────────────────────► pruned_weights
                                            │
  input ──────────────────────────► F.linear(input, pruned_weights, bias)
                                            │
                                           out
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A drop-in replacement for ``nn.Linear`` augmented with per-weight
    learnable gates that enable dynamic self-pruning.

    Parameters
    ----------
    in_features  : int   – size of each input sample.
    out_features : int   – size of each output sample.
    bias         : bool  – if ``True`` (default) adds a learnable bias.

    Attributes
    ----------
    weight      : Parameter (out_features, in_features)
    bias        : Parameter (out_features,) or ``None``
    gate_scores : Parameter (out_features, in_features)
                  Learnable raw scores passed through sigmoid to produce gates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # ── Primary weight matrix ──────────────────────────────────────
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # ── Bias term ─────────────────────────────────────────────────
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # ── Learnable gate scores (same shape as weight) ───────────────
        # Initialised to a small positive value so that sigmoid(score) ≈ 0.5
        # at the start of training — all gates begin "half open," giving the
        # network equal opportunity to keep or prune each connection.
        self.gate_scores = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self._init_parameters()

    # ──────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────

    def _init_parameters(self) -> None:
        """
        Kaiming uniform for weights (same as nn.Linear default).
        Gate scores initialised to 0 → sigmoid(0) = 0.5 (fully neutral).
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Zero-init gate scores so every gate starts at exactly 0.5.
        nn.init.zeros_(self.gate_scores)

    # ──────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gated linear transformation.

        Steps
        -----
        1. gates = sigmoid(gate_scores)          — soft mask in [0, 1]
        2. pruned_weights = weight * gates        — element-wise masking
        3. output = x @ pruned_weights.T + bias  — standard affine op

        Gradients propagate through all three steps automatically.

        Args:
            x: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).
        """
        # Step 1 — soft gates in [0, 1]
        gates: torch.Tensor = torch.sigmoid(self.gate_scores)

        # Step 2 — mask weights; gradients flow to both `weight` and `gate_scores`
        pruned_weights: torch.Tensor = self.weight * gates

        # Step 3 — standard linear operation (fully differentiable)
        return F.linear(x, pruned_weights, self.bias)

    # ──────────────────────────────────────────────────────────────────
    # Utility helpers
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Return current sigmoid gate values (no grad tracking)."""
        return torch.sigmoid(self.gate_scores)

    @torch.no_grad()
    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below ``threshold`` for this layer."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    @torch.no_grad()
    def apply_hard_threshold(self, threshold: float = 1e-2) -> None:
        """
        **Bonus — Hard Threshold Pruning**

        Permanently zero-out gate scores whose sigmoid value is below
        ``threshold``, converting soft gates into binary masks.
        This is applied post-training to get a truly sparse model.
        """
        mask = self.get_gates() < threshold          # True where pruned
        # Push those gate_scores to -100 so sigmoid ≈ 0 permanently
        self.gate_scores.data[mask] = -100.0

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )