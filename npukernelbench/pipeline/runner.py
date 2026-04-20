"""Pipeline runner — orchestrates stages sequentially."""

from __future__ import annotations

from typing import List

from npukernelbench.config.models import FullConfig
from npukernelbench.pipeline.context import PipelineContext
from npukernelbench.pipeline.stage import PipelineStage, BatchPipelineStage


class PipelineRunner:
    """Execute a sequence of stages over a list of tasks."""

    def __init__(self, stages: List, config: FullConfig):
        self.stages = stages
        self.ctx = PipelineContext(config=config)

    def run(self, tasks: list) -> list:
        """Run all stages, filtering out ineligible tasks at each stage.

        When a stage is not in active_stages but has a `try_restore` method,
        the runner attempts to restore results from disk so downstream
        stages can proceed.
        """
        active_stages = {s for s in self.ctx.config.active_stages}

        for stage in self.stages:
            stage_name = stage.name

            if stage_name not in active_stages:
                # Stage is not active — try to restore results from disk
                if hasattr(stage, "try_restore"):
                    restored = 0
                    for task in tasks:
                        if _get_result(task, stage_name + "_result") is None:
                            if stage.try_restore(task):
                                restored += 1
                    if restored:
                        print(f"[Pipeline] Restored {restored} results from disk for stage: {stage_name}")
                print(f"[Pipeline] Skipping stage: {stage_name}")
                continue

            # Filter tasks that haven't failed previous stages
            eligible = [t for t in tasks if _is_eligible(t, stage_name)]
            if not eligible:
                print(f"[Pipeline] No eligible tasks for stage: {stage_name}")
                continue

            print(f"[Pipeline] Stage '{stage_name}': {len(eligible)} tasks")

            if isinstance(stage, BatchPipelineStage):
                stage.execute_batch(eligible, self.ctx)
            else:
                for task in eligible:
                    stage.execute(task, self.ctx)

        return tasks


def _get_result(task, attr: str):
    """Get a result attribute, handling both BuildHandle and TestHandle."""
    val = getattr(task, attr, None)
    if val is not None:
        return val
    build = getattr(task, "build", None)
    if build is not None:
        return getattr(build, attr, None)
    return None


def _is_eligible(task, stage_name: str) -> bool:
    """Check if a task can proceed to the given stage."""
    if stage_name == "code_gen":
        return True

    if stage_name == "compile":
        cg = _get_result(task, "code_gen_result")
        if cg is not None and not cg.success:
            return False
        return True

    if stage_name == "precision":
        comp = _get_result(task, "compile_result")
        if comp is None or not comp.success:
            return False
        return True

    if stage_name == "perf":
        prec = _get_result(task, "precision_result")
        if prec is None or not prec.success:
            return False
        return True

    return True
