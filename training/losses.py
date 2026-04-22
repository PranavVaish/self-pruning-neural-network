"""
training/losses.py
------------------
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

from __future__ import annotations

import torch
import torch.nn as nn


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