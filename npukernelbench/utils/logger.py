"""Logging infrastructure — no global config dependency."""

from __future__ import annotations

import logging
import os
from typing import Optional

_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


def init_logger(name: str) -> logging.Logger:
    """Get or create a logger under the 'npukernelbench' hierarchy."""
    if not name.startswith("npukernelbench."):
        logger = logging.getLogger(f"npukernelbench.{name}")
    else:
        logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger


def add_file_handler(
    log_file_path: str,
    level: str = "DEBUG",
    logger: Optional[logging.Logger] = None,
) -> logging.FileHandler:
    """Attach a file handler to a logger."""
    if logger is None:
        logger = logging.getLogger("npukernelbench")

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    fh = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(fh)
    return fh


# Module-level convenience logger
common_logger = init_logger("common")
