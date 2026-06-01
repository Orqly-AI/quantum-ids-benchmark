"""Logging configured to write to both the console and a per-run file.

Each run gets its own log file under results/logs/ named after the config slug
and a timestamp, so a sweep produces one log per experiment alongside the JSON
results. Call get_logger() once at the start of a run.
"""
from __future__ import annotations
import logging
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "..", "results", "logs")


def get_logger(name: str = "quantum_ids", run_slug: str | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to console and (if run_slug given) a file.

    Idempotent per (name, run_slug): re-calling won't stack duplicate handlers.
    """
    logger = logging.getLogger(name if run_slug is None else f"{name}.{run_slug}")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:  # already configured
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                            datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if run_slug is not None:
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"{run_slug}_{ts}.log")
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("Logging to %s", log_path)

    return logger
