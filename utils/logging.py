"""Logging helpers for CLA experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Create a process-wide logger with optional file output."""

    logger = logging.getLogger("cla")
    logger.setLevel(level.upper())
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        destination = Path(log_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(destination, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
