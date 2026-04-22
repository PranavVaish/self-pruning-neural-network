"""
utils/logger.py
---------------
Lightweight structured logger for training runs.

Wraps Python's built-in logging module and writes both to stdout
and a rotating file sink so that long experiments are never lost.
"""

import io
import logging
import os
import sys
from datetime import datetime


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