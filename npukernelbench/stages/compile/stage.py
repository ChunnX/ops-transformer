"""Compile stage — prepare, build, verify — zero torch dependency."""

from __future__ import annotations

import os

from npukernelbench.config.models import FullConfig
from npukernelbench.models.task import BuildHandle
from npukernelbench.models.results import CompileResult
from npukernelbench.pipeline.context import PipelineContext
from npukernelbench.stages.compile.source_preparer import prepare_source_files
from npukernelbench.stages.compile.builder import run_build
from npukernelbench.stages.compile.verifier import verify_build
from npukernelbench.utils.logger import init_logger, add_file_handler


class CompileStage:
    """Compile each BuildHandle: prepare sources → build.sh → verify artifacts."""

    name = "compile"

    def __init__(self, config: FullConfig):
        self.config = config

    def execute(self, task: BuildHandle, ctx: PipelineContext) -> None:
        logger = init_logger(f"compile.{task.short_id}")
        os.makedirs(task.paths.log_dir, exist_ok=True)
        add_file_handler(task.paths.compile_log, logger=logger)

        logger.info(f"Starting compilation: {task.short_id}")

        try:
            os.makedirs(task.paths.build_dir, exist_ok=True)

            # Set custom opp path for runtime
            _set_custom_opp_path(task)

            # Prepare source files
            prepare_source_files(task, self.config, logger)

            # Build
            success = run_build(task, logger, self.config.skip_op_build)

            # Verify
            if success:
                overall, ops_so, opapi, kernel_o = verify_build(task)
                if not overall:
                    logger.error(f"Verification failed: so={ops_so} opapi={opapi} kernel={kernel_o}")
                    success = False
                else:
                    logger.info(f"Compiled successfully: {task.metadata.op_name} sample={task.sample_id}")

        except Exception as e:
            logger.error(f"Compilation error: {e}", exc_info=True)
            success = False

        task.compile_result = CompileResult(success=success, log_file=task.paths.compile_log)

    def execute_batch(self, tasks: list[BuildHandle], ctx: PipelineContext) -> None:
        """Execute compilation for multiple tasks (can be parallelized later)."""
        for task in tasks:
            self.execute(task, ctx)

    def try_restore(self, task: BuildHandle) -> bool:
        """Check disk for existing compiled artifacts and restore compile_result.

        Returns True if artifacts exist and compile_result was restored.
        """
        overall, _, _, _ = verify_build(task)
        if overall:
            task.compile_result = CompileResult(success=True, log_file=task.paths.compile_log)
            return True
        return False


def _set_custom_opp_path(build: BuildHandle) -> None:
    """Set ASCEND_CUSTOM_OPP_PATH environment variable."""
    from npukernelbench.stages.compile.verifier import _find_custom_lib_path
    lib_path = _find_custom_lib_path(build.paths.src_dir)
    if lib_path:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{lib_path}/"
