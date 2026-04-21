"""Shared runtime context passed through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineContext:
    """Mutable shared state for a pipeline run."""
    config: Any  # FullConfig, typed as Any to avoid circular import
    extra: dict = field(default_factory=dict)
