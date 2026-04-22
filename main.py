"""
main.py
-------
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

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time
from copy import deepcopy
from typing import List

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model.network import SelfPruningNet
from model.prunable_linear import PrunableLinear
from training.trainer import Trainer
from utils.logger import get_logger
from utils.metrics import collect_gate_values, model_size_bytes
from utils.seed import set_seed


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