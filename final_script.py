# ============================================================
# Self-Pruning Neural Network — Single Script Implementation
# ============================================================

"""
This script implements a self-pruning neural network using learnable
sigmoid gates with L1 sparsity regularization.

Contains:

1. PrunableLinear layer
2. SelfPruningNet architecture
3. Training loop
4. Evaluation
5. Visualization
6. Experiment runner

Fully self-contained.

Dataset:
CIFAR-10

Outputs:
- Accuracy
- Sparsity
- Model size
- Gate histogram
- Training curves
"""

# ============================================================
# IMPORTS
# ============================================================
from __future__ import annotations

import os
import sys
import io
import math
import time
import yaml
import random
import logging
import argparse
import cProfile
import pstats

from datetime import datetime
from copy import deepcopy
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast

from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ============================================================
# SEED
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducible training across runs.

    Args:
        seed: Integer seed value. Default: 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # multi-GPU safety

    # Ensure deterministic CUDA kernels (may reduce throughput slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    # disable auto-tuning

# ============================================================
# LOGGER
# ============================================================

def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Build (or retrieve) a named logger that writes to both console and file.

    Args:
        name:    Logger name — typically the calling module's __name__.
        log_dir: Directory where log files are stored.
        level:   Logging verbosity (default: INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding duplicate handlers when the same logger is requested
        # multiple times within a single Python process.
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (UTF-8 safe on Windows cp1252) ───────────────────
    # Use reconfigure() when available (Python 3.7+); fall back to TextIOWrapper.
    stdout = sys.stdout
    if hasattr(stdout, "reconfigure"):
        try:
            stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    else:
        stdout = io.TextIOWrapper(stdout.buffer, encoding="utf-8", errors="replace")
    console_handler = logging.StreamHandler(stream=stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# ============================================================
# METRICS
# ============================================================

"""
Evaluation and sparsity metrics for the self-pruning network.

All functions are stateless and accept plain tensors or lists so they
can be called from anywhere in the training pipeline without importing
any model-specific code.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Top-1 accuracy from raw logits.

    Args:
        logits:  (N, C) float tensor — model output before softmax.
        targets: (N,)   long  tensor — ground-truth class indices.

    Returns:
        Accuracy in [0, 1].
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Sparsity
# ─────────────────────────────────────────────────────────────────────────────

def compute_sparsity(model: nn.Module, threshold: float = 1e-2) -> dict[str, float]:
    """
    Compute the fraction of pruned weights across all PrunableLinear layers.

    A weight is considered *pruned* when its corresponding sigmoid gate
    value falls below ``threshold``.

    Args:
        model:     Any nn.Module that may contain PrunableLinear sub-modules.
        threshold: Gate value below which a weight is counted as pruned.

    Returns:
        Dictionary with keys:
            ``sparsity``      – global fraction of pruned weights in [0, 1].
            ``pruned_count``  – total number of pruned weights.
            ``total_count``   – total number of weights inspected.
            ``layer_sparsity``– per-layer sparsity dict {layer_name: float}.
    """
    # Import here to avoid circular imports at module load time.
    from model.prunable_linear import PrunableLinear

    pruned_total = 0
    weight_total = 0
    layer_stats: dict[str, float] = {}

    for name, module in model.named_modules():
        if isinstance(module, PrunableLinear):
            with torch.no_grad():
                gates = torch.sigmoid(module.gate_scores)
                pruned = (gates < threshold).sum().item()
                total = gates.numel()

            pruned_total += pruned
            weight_total += total
            layer_stats[name] = pruned / total if total > 0 else 0.0

    global_sparsity = pruned_total / weight_total if weight_total > 0 else 0.0

    return {
        "sparsity": global_sparsity,
        "pruned_count": pruned_total,
        "total_count": weight_total,
        "layer_sparsity": layer_stats,
    }


def collect_gate_values(model: nn.Module) -> torch.Tensor:
    """
    Collect all sigmoid gate values from every PrunableLinear layer
    into a single flat CPU tensor (useful for histogram visualisation).

    Args:
        model: A trained nn.Module containing PrunableLinear sub-modules.

    Returns:
        1-D float tensor of gate values in [0, 1].
    """
    from model.prunable_linear import PrunableLinear

    gate_list = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            with torch.no_grad():
                gates = torch.sigmoid(module.gate_scores).cpu().flatten()
            gate_list.append(gates)

    if not gate_list:
        return torch.tensor([])
    return torch.cat(gate_list)


def model_size_bytes(model: nn.Module) -> int:
    """
    Estimate the in-memory parameter footprint in bytes.

    Args:
        model: Any nn.Module.

    Returns:
        Total bytes occupied by all trainable parameters.
    """
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
        if p.requires_grad
    )

# ============================================================
# PRUNABLE LINEAR
# ============================================================

"""
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

# ============================================================
# NETWORK
# ============================================================

"""
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

# ============================================================
# LOSSES
# ============================================================

"""
Loss functions for the self-pruning neural network.

The total training objective is a weighted sum of two terms:

    Total Loss = CrossEntropyLoss(logits, targets)
               + λ × SparsityLoss(model)

CrossEntropyLoss
    Standard multi-class classification loss. Drives the network to
    correctly classify images.

SparsityLoss (L1 on gates)
    The L1 norm of all sigmoid gate values summed across every
    PrunableLinear layer in the model.

    Because sigmoid(gate_score) is always positive, the L1 norm is
    simply the *sum* of all gate values.  The gradient of this term
    w.r.t. each gate_score is:

        ∂(SparsityLoss)/∂(gate_score_i)  =  sigmoid(gate_score_i) × (1 − sigmoid(gate_score_i))

    This gradient is largest when gate_score_i ≈ 0 (i.e. gate ≈ 0.5)
    and approaches zero when the gate is already near 0 or 1.
    Combined with the classification loss, the optimiser settles gates
    at 0 (pruned) or a functional value > 0 (kept), producing a
    bimodal "spike-and-cluster" distribution.
"""

class SparsityLoss(nn.Module):
    """
    L1 penalty on the sigmoid gate values of all PrunableLinear layers.

    Summing (not averaging) the gates makes the penalty scale with model
    size, which is intentional: larger models have more gates to regulate.

    Args:
        lambda_sparse: Regularisation coefficient λ. Higher values
                       produce sparser networks at the potential cost of
                       classification accuracy.
    """

    def __init__(self, lambda_sparse: float = 1e-3) -> None:
        super().__init__()
        self.lambda_sparse = lambda_sparse

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute lambda × mean(sigmoid(gate_score)) over all PrunableLinear layers.

        Uses **mean** normalization (not sum) so that lambda is model-size-
        independent and directly comparable to the CE loss magnitude (~2.3 at
        random init for 10 classes).  The per-gate gradient is:

            d(SP)/d(gate_score_i) = lambda * sigmoid(g_i) * (1 - sigmoid(g_i))

        which is the same expression regardless of whether we use sum or mean,
        but with mean the lambda values are far easier to interpret and tune.

        Args:
            model: The full network (nn.Module) containing PrunableLinear
                   sub-modules.

        Returns:
            Scalar tensor — the weighted sparsity penalty.
        """
        from model.prunable_linear import PrunableLinear  # avoid circular import

        gate_tensors = []
        total_count = 0

        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                gate_tensors.append(gates.sum())
                total_count += gates.numel()

        if not gate_tensors or total_count == 0:
            return torch.zeros(1,
                               device=next(model.parameters()).device,
                               requires_grad=True).squeeze()

        # Mean over ALL gate values across every PrunableLinear layer
        gate_mean = torch.stack(gate_tensors).sum() / total_count
        return self.lambda_sparse * gate_mean


class TotalLoss(nn.Module):
    """
    Combines CrossEntropyLoss and SparsityLoss into a single criterion.

    Usage
    -----
    >>> criterion = TotalLoss(lambda_sparse=0.001)
    >>> loss, ce, sp = criterion(logits, targets, model)
    >>> loss.backward()

    Args:
        lambda_sparse: Regularisation coefficient λ for the sparsity term.
        label_smoothing: Optional label smoothing for the CE loss (0 = off).
    """

    def __init__(
        self,
        lambda_sparse: float = 1e-3,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.sparsity_loss = SparsityLoss(lambda_sparse)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss and its components.

        Args:
            logits:  (N, C) model output.
            targets: (N,)   ground-truth class indices.
            model:   The full network.

        Returns:
            Tuple of (total_loss, ce_loss, sparsity_loss) tensors.
        """
        ce = self.ce_loss(logits, targets)
        sp = self.sparsity_loss(model)
        total = ce + sp
        return total, ce, sp

# ============================================================
# TRAINER
# ============================================================

"""
Production-grade training loop for the self-pruning network.

Features
========
* Mixed-precision training (torch.cuda.amp) — falls back gracefully on CPU.
* Gradient clipping to stabilise gate_scores updates.
* Early stopping on validation accuracy with configurable patience.
* Per-epoch checkpointing with best-model tracking.
* Structured logging (loss components, sparsity, accuracy, LR).
* Cosine annealing LR scheduler.
"""

class Trainer:
    """
    Encapsulates the full train / validate / evaluate lifecycle.

    Parameters
    ----------
    model          : nn.Module — the SelfPruningNet (or any compatible model).
    train_loader   : DataLoader for training split.
    val_loader     : DataLoader for validation split.
    config         : dict — training configuration block from config.yaml.
    lambda_sparse  : float — λ for the sparsity regularisation term.
    device         : torch.device — target device.
    checkpoint_dir : str — directory to write model checkpoints.
    log_dir        : str — directory for log files.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        lambda_sparse: float = 1e-3,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.lambda_sparse = lambda_sparse
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.logger = get_logger(
            f"trainer_lambda{lambda_sparse}", log_dir=log_dir
        )

        # ── Loss ──────────────────────────────────────────────────────
        self.criterion = TotalLoss(
            lambda_sparse=lambda_sparse,
            label_smoothing=0.0,
        )

        # ── Optimiser ─────────────────────────────────────────────────
        lr = config.get("learning_rate", 1e-3)
        wd = config.get("weight_decay", 1e-4)

        # Separate gate_scores from regular parameters so they get:
        #  - Higher LR (converge faster to 0 or open)
        #  - No weight_decay (L2 wd opposes the L1 sparsity signal)
        from model.prunable_linear import PrunableLinear
        gate_param_ids = {
            id(m.gate_scores)
            for m in model.modules()
            if isinstance(m, PrunableLinear)
        }
        gate_params  = [p for p in model.parameters() if id(p) in gate_param_ids]
        other_params = [p for p in model.parameters() if id(p) not in gate_param_ids]

        self.optimizer = torch.optim.Adam([
            {"params": other_params, "lr": lr,     "weight_decay": wd},
            {"params": gate_params,  "lr": lr * 5, "weight_decay": 0.0},
        ])

        # ── LR Scheduler ──────────────────────────────────────────────
        epochs = config.get("epochs", 30)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        # ── Mixed precision ───────────────────────────────────────────
        self.use_amp = config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        # ── Hyper-params ──────────────────────────────────────────────
        self.epochs = epochs
        self.grad_clip = config.get("grad_clip", 1.0)
        self.log_interval = config.get("log_interval", 50)
        self.patience = config.get("early_stopping_patience", 10)
        self.min_delta = config.get("early_stopping_min_delta", 0.001)
        self.sparsity_threshold = config.get("sparsity_threshold", 1e-2)

        # ── State ─────────────────────────────────────────────────────
        self.best_val_acc: float = 0.0
        self.no_improve_count: int = 0
        self.history: list[dict] = []

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def fit(self) -> list[dict]:
        """
        Run the full training loop.

        Returns
        -------
        List of per-epoch metric dictionaries.
        """
        self.logger.info(
            f"Starting training | lambda={self.lambda_sparse} | "
            f"device={self.device} | AMP={self.use_amp} | epochs={self.epochs}"
        )
        self.model.to(self.device)

        for epoch in range(1, self.epochs + 1):
            t_start = time.perf_counter()

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch()

            elapsed = time.perf_counter() - t_start
            self.scheduler.step()
            lr_now = self.scheduler.get_last_lr()[0]

            sparsity_info = compute_sparsity(self.model, self.sparsity_threshold)
            sparsity_pct = sparsity_info["sparsity"] * 100

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_ce": train_metrics["ce"],
                "train_sp": train_metrics["sp"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "sparsity_pct": sparsity_pct,
                "lr": lr_now,
                "elapsed_s": elapsed,
            }
            self.history.append(epoch_record)

            self.logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"loss={train_metrics['loss']:.4f} "
                f"(CE={train_metrics['ce']:.4f} SP={train_metrics['sp']:.4f}) | "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"val_acc={val_metrics['acc']:.4f} | "
                f"sparsity={sparsity_pct:.1f}% | "
                f"lr={lr_now:.2e} | {elapsed:.1f}s"
            )

            # ── Checkpointing ──────────────────────────────────────────
            if val_metrics["acc"] > self.best_val_acc + self.min_delta:
                self.best_val_acc = val_metrics["acc"]
                self.no_improve_count = 0
                self._save_checkpoint(epoch, val_metrics["acc"], is_best=True)
            else:
                self.no_improve_count += 1

            # ── Early stopping ─────────────────────────────────────────
            if self.no_improve_count >= self.patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(val_acc did not improve by >{self.min_delta:.4f} "
                    f"for {self.patience} consecutive epochs)."
                )
                break

        self.logger.info(
            f"Training complete. Best val_acc={self.best_val_acc:.4f}"
        )
        return self.history

    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Evaluate on the test set and report accuracy + sparsity.

        Args:
            test_loader: DataLoader for the held-out test split.

        Returns:
            Dict with keys: test_acc, sparsity_pct, layer_sparsity,
                            pruned_count, total_count.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_acc = correct / total
        sparsity_info = compute_sparsity(self.model, self.sparsity_threshold)

        result = {
            "test_acc": test_acc,
            "sparsity_pct": sparsity_info["sparsity"] * 100,
            "layer_sparsity": sparsity_info["layer_sparsity"],
            "pruned_count": sparsity_info["pruned_count"],
            "total_count": sparsity_info["total_count"],
        }

        self.logger.info(
            f"Test accuracy: {test_acc:.4f} | "
            f"Sparsity: {result['sparsity_pct']:.2f}%"
        )
        return result

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = ce_accum = sp_accum = correct = n_samples = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)  # memory-efficient zero

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss, ce, sp = self.criterion(logits, labels, self.model)

            self.scaler.scale(loss).backward()

            # Gradient clipping — applied after unscaling
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ── Accumulate metrics ─────────────────────────────────────
            bs = labels.size(0)
            total_loss += loss.item() * bs
            ce_accum   += ce.item()   * bs
            sp_accum   += sp.item()   * bs
            correct    += (logits.argmax(1) == labels).sum().item()
            n_samples  += bs

            if (batch_idx + 1) % self.log_interval == 0:
                self.logger.debug(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f} ce={ce.item():.4f} sp={sp.item():.4f}"
                )

        return {
            "loss": total_loss / n_samples,
            "ce":   ce_accum   / n_samples,
            "sp":   sp_accum   / n_samples,
            "acc":  correct    / n_samples,
        }

    def _validate_epoch(self) -> dict:
        self.model.eval()
        total_loss = correct = n_samples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss, _, _ = self.criterion(logits, labels, self.model)

                bs = labels.size(0)
                total_loss += loss.item() * bs
                correct    += (logits.argmax(1) == labels).sum().item()
                n_samples  += bs

        return {
            "loss": total_loss / n_samples,
            "acc":  correct    / n_samples,
        }

    def _save_checkpoint(
        self, epoch: int, val_acc: float, is_best: bool = False
    ) -> None:
        tag = "best" if is_best else f"epoch{epoch}"
        path = os.path.join(
            self.checkpoint_dir,
            f"lambda{self.lambda_sparse}_{tag}.pt",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": val_acc,
                "lambda_sparse": self.lambda_sparse,
            },
            path,
        )
        self.logger.debug(f"Checkpoint saved → {path}")

# ============================================================
# MAIN
# ============================================================
"""
Entry point for the Self-Pruning Neural Network experiments.

Orchestrates the full pipeline:
  1. Load configuration.
  2. Prepare CIFAR-10 data loaders.
  3. For each λ value: train, evaluate, collect metrics.
  4. Print a summary results table.
  5. Generate gate-value histogram plots.
  6. (Bonus) Apply hard pruning + export TorchScript.
  7. Run a lightweight cProfile performance snapshot.

Usage
-----
    python main.py                        # uses config/config.yaml defaults
    python main.py --epochs 5 --lambda_values 0.001  # quick smoke-test

Dependencies (see requirements.txt):
    torch, torchvision, matplotlib, pyyaml, tqdm
"""
# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_data_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    val_fraction: float = 0.1,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test DataLoaders for CIFAR-10.

    Train set uses standard augmentation (random crop + horizontal flip).
    Val and test sets use only normalisation.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_ds = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_tf
    )

    # ── Train / val split ─────────────────────────────────────────────
    n_val   = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val])

    # Override transform for validation subset (no augmentation)
    val_ds.dataset = deepcopy(full_train)
    val_ds.dataset.transform = eval_tf

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_histogram(
    model: torch.nn.Module,
    lambda_val: float,
    output_dir: str = "plots",
    sparsity_pct: float = 0.0,
    test_acc: float = 0.0,
) -> str:
    """
    Save a histogram of all sigmoid gate values to ``output_dir``.

    A successful pruning run shows a large spike near 0 and a secondary
    cluster of active gates away from 0.

    Returns
    -------
    Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    gates = collect_gate_values(model).numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gates, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Gate Value (sigmoid output)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Gate Value Distribution  |  lambda={lambda_val}\n"
        f"Test Acc: {test_acc*100:.2f}%   Sparsity: {sparsity_pct:.1f}%",
        fontsize=13,
    )
    ax.set_xlim(0, 1)
    ax.axvline(x=0.01, color="#CC3311", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax.legend()

    path = os.path.join(output_dir, f"gate_histogram_lambda{lambda_val}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_curves(
    history: list[dict],
    lambda_val: float,
    output_dir: str = "plots",
) -> str:
    """Plot train/val accuracy and total loss over epochs."""
    os.makedirs(output_dir, exist_ok=True)
    epochs     = [r["epoch"]      for r in history]
    train_acc  = [r["train_acc"]  for r in history]
    val_acc    = [r["val_acc"]    for r in history]
    train_loss = [r["train_loss"] for r in history]
    sparsity   = [r["sparsity_pct"] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_acc, label="Train")
    axes[0].plot(epochs, val_acc,   label="Val", linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, train_loss, color="#CC3311")
    axes[1].set_title("Total Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")

    axes[2].plot(epochs, sparsity, color="#228833")
    axes[2].set_title("Sparsity %")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Sparsity (%)")

    fig.suptitle(f"Training Curves  |  lambda={lambda_val}", fontsize=13)
    fig.tight_layout()

    path = os.path.join(output_dir, f"training_curves_lambda{lambda_val}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_forward_pass(model: torch.nn.Module, device: torch.device) -> str:
    """
    Run a cProfile snapshot of a single forward + backward pass.

    Returns a formatted string of the top-20 hotspots.
    """
    model.train()
    dummy_x = torch.randn(64, 3, 32, 32, device=device)
    dummy_y = torch.randint(0, 10, (64,), device=device)
    criterion = torch.nn.CrossEntropyLoss()

    pr = cProfile.Profile()
    pr.enable()
    out = model(dummy_x)
    loss = criterion(out, dummy_y)
    loss.backward()
    pr.disable()

    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(20)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    lambda_sparse: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    logger,
    plots_dir: str,
    checkpoint_dir: str,
    log_dir: str,
) -> dict:
    """
    Train and evaluate a single model for the given λ value.

    Returns a summary dict for the results table.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  EXPERIMENT: lambda = {lambda_sparse}")
    logger.info(f"{'='*60}")

    # ── Fresh model for each experiment ───────────────────────────────
    set_seed(config["training"]["seed"])
    model_cfg = config["model"]
    model = SelfPruningNet(
        in_channels=model_cfg["in_channels"],
        hidden_dims=model_cfg["hidden_dims"],
        num_classes=model_cfg["num_classes"],
        dropout_rate=model_cfg["dropout_rate"],
    )
    model.to(device)

    param_info = model.count_parameters()
    logger.info(
        f"Model params: total={param_info['total_params']:,}  "
        f"gate_scores={param_info['gate_score_params']:,}"
    )

    # ── Profile (first experiment only to avoid repeated output) ──────
    if lambda_sparse == config["pruning"]["lambda_values"][0]:
        profile_txt = profile_forward_pass(model, device)
        logger.info(f"\ncProfile — forward+backward (top 20 calls):\n{profile_txt}")

    # ── Train ─────────────────────────────────────────────────────────
    train_cfg = {**config["training"], **config["pruning"], **config["logging"]}
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        lambda_sparse=lambda_sparse,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    t0 = time.perf_counter()
    history = trainer.fit()
    train_time = time.perf_counter() - t0

    # ── Evaluate ──────────────────────────────────────────────────────
    eval_result = trainer.evaluate(test_loader)
    test_acc    = eval_result["test_acc"]
    sparsity    = eval_result["sparsity_pct"]

    # ── Bonus — hard pruning + TorchScript export ─────────────────────
    model.apply_hard_pruning(threshold=config["pruning"]["sparsity_threshold"])
    script_path = os.path.join(
        checkpoint_dir, f"pruned_lambda{lambda_sparse}.pt"
    )
    try:
        model.export_torchscript(script_path)
        logger.info(f"TorchScript model saved → {script_path}")
    except Exception as exc:
        logger.warning(f"TorchScript export failed: {exc}")

    # ── Plots ─────────────────────────────────────────────────────────
    hist_path = plot_gate_histogram(
        model, lambda_sparse, plots_dir, sparsity, test_acc
    )
    curve_path = plot_training_curves(history, lambda_sparse, plots_dir)
    logger.info(f"Plots saved: {hist_path}, {curve_path}")

    model_bytes = model_size_bytes(model)

    # Count non-pruned weights only
    effective_params = sum(
        (m.get_gates() >= 0.01).sum().item()
        for m in model.modules() if isinstance(m, PrunableLinear)
    )
    # Estimate effective MB
    effective_mb = (model_bytes / (1024 ** 2)) * (1 - (sparsity / 100.0))

    return {
        "lambda":       lambda_sparse,
        "test_acc":     test_acc,
        "sparsity_pct": sparsity,
        "train_time_s": train_time,
        "model_mb":     model_bytes / (1024 ** 2),
        "effective_mb": effective_mb,
        "history":      history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print(f"{'Lambda':<12} {'Test Acc':>10} {'Sparsity %':>12} {'Train Time':>12} {'Model MB':>10} {'Effective MB':>14}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['lambda']:<12} "
            f"{r['test_acc']*100:>9.2f}% "
            f"{r['sparsity_pct']:>11.1f}% "
            f"{r['train_time_s']:>10.1f}s "
            f"{r['model_mb']:>9.2f}MB "
            f"{r['effective_mb']:>12.2f}MB"
        )
    print("=" * 90 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-Pruning Neural Network — CIFAR-10")
    p.add_argument("--config",        default=CONFIG_PATH, help="Path to config YAML")
    p.add_argument("--epochs",        type=int,   default=None, help="Override epochs")
    p.add_argument("--lambda_values", type=float, nargs="+", default=None,
                   help="Override λ values (space-separated)")
    p.add_argument("--data_dir",      default="./data",       help="CIFAR-10 data dir")
    p.add_argument("--output_dir",    default="./outputs",    help="Root output dir")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    # ── CLI overrides ──────────────────────────────────────────────────
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lambda_values is not None:
        config["pruning"]["lambda_values"] = args.lambda_values

    # ── Directories ────────────────────────────────────────────────────
    out_root  = args.output_dir
    plots_dir = os.path.join(out_root, config["visualization"]["output_dir"])
    ckpt_dir  = os.path.join(out_root, config["checkpoint"]["dir"])
    log_dir   = os.path.join(out_root, config["logging"]["log_dir"])
    os.makedirs(out_root, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────
    logger = get_logger("main", log_dir=log_dir)

    # -- Seed & device ---------------------------------------------------------
    set_seed(config["training"]["seed"])
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    if not cuda_available:
        logger.warning(
            "CUDA not available — running on CPU (this will be slow). "
            "If you have an NVIDIA GPU, reinstall PyTorch with CUDA support:\n"
            "  pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu121"
        )
    else:
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")

    # ── Data ──────────────────────────────────────────────────────────
    train_cfg = config["training"]
    train_loader, val_loader, test_loader = build_data_loaders(
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"] and cuda_available,  # no-op on CPU
        data_dir=args.data_dir,
    )
    logger.info(
        f"Data loaded — train={len(train_loader.dataset):,}  "
        f"val={len(val_loader.dataset):,}  "
        f"test={len(test_loader.dataset):,}"
    )

    # ── Run experiments for each λ ────────────────────────────────────
    lambda_values: List[float] = config["pruning"]["lambda_values"]
    all_results = []

    for lam in lambda_values:
        result = run_experiment(
            lambda_sparse=lam,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            logger=logger,
            plots_dir=plots_dir,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
        )
        all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────
    print_results_table(all_results)
    logger.info("All experiments complete.")

if __name__ == "__main__":
    main()