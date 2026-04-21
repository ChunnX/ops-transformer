"""Build execution — call build.sh in subprocess — zero torch dependency."""

from __future__ import annotations

import os
import subprocess

from npukernelbench.models.task import BuildHandle
from npukernelbench.stages.compile.verifier import op_artifacts_exist


def run_build(build: BuildHandle, logger=None, skip_op_build: bool = False) -> bool:
    """Run build.sh in the source directory. Returns True on success.

    When skip_op_build is True and operator artifacts already exist,
    only rebuilds the pybind Python extension (fast path).
    Otherwise performs a full compilation.
    """
    src = build.paths.src_dir
    old_cwd = os.getcwd()

    skip_op = skip_op_build and op_artifacts_exist(build)
    if skip_op and logger:
        logger.info("skip_op_build enabled and artifacts found — only rebuilding pybind extension")

    try:
        os.chdir(src)

        # Clean stale CMake cache
        if os.path.exists("CMakeCache.txt"):
            os.remove("CMakeCache.txt")

        cmd = ["bash", "build.sh"]
        if skip_op:
            cmd.append("--skip-op-build")

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if logger:
            if proc.stdout:
                logger.debug(proc.stdout)
            if proc.stderr:
                if proc.returncode != 0:
                    logger.error(proc.stderr)
                else:
                    logger.warning(proc.stderr)

        if proc.returncode != 0:
            if logger:
                logger.error(f"Build failed with exit code {proc.returncode}")
            return False

        return True

    finally:
        os.chdir(old_cwd)
