"""
utils/metrics.py
----------------
Evaluation and sparsity metrics for the self-pruning network.

All functions are stateless and accept plain tensors or lists so they
can be called from anywhere in the training pipeline without importing
any model-specific code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.prunable_linear import PrunableLinear


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