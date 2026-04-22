"""
utils/seed.py
-------------
Deterministic training utilities.

Sets seeds for Python, NumPy, PyTorch (CPU + CUDA) and configures
cuDNN for full reproducibility at the cost of a small speed penalty.
"""

import os
import random
import numpy as np
import torch


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