"""Pipeline stage protocol definition."""

from __future__ import annotations

from typing import Protocol, List, runtime_checkable

from npukernelbench.config.models import FullConfig
from npukernelbench.models.task import BuildHandle, TestHandle


@runtime_checkable
class PipelineStage(Protocol):
    """Interface for a pipeline stage."""

    @property
    def name(self) -> str: ...

    def execute(self, task, ctx: "PipelineContext") -> None:
        """Execute this stage on a single task, mutating task in-place."""
        ...


@runtime_checkable
class BatchPipelineStage(Protocol):
    """Interface for a stage that processes all tasks at once (e.g. parallel execution)."""

    @property
    def name(self) -> str: ...

    def execute_batch(self, tasks: list, ctx: "PipelineContext") -> None:
        """Execute this stage on a batch of tasks."""
        ...
