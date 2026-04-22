"""
training/trainer.py
-------------------
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

from __future__ import annotations

import os
import time
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from training.losses import TotalLoss
from utils.metrics import compute_accuracy, compute_sparsity
from utils.logger import get_logger


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