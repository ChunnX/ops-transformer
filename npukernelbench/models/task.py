"""Core task data models — BuildHandle (compile unit) + TestHandle (test unit)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional, List

from npukernelbench.models.results import (
    CodeGenResult,
    CompileResult,
    PrecisionResult,
    PerfResult,
)


@dataclass(frozen=True)
class TaskMetadata:
    """Immutable identity for an operator task."""
    op_name: str           # e.g. "Sqrt"
    kernel_name: str       # e.g. "sqrt" (snake_case)
    category: str          # e.g. "Math"
    level_id: str          # e.g. "1"
    op_dir_name: str       # e.g. "Sqrt_custom"
    problem_def_path: str  # absolute path to task definition directory

    @staticmethod
    def from_path(problem_def_path: str) -> TaskMetadata:
        """Parse metadata from a task definition directory path."""
        op_dir_name = os.path.basename(problem_def_path)
        op_name = op_dir_name.split("_")[0]
        kernel_name = _camel_to_snake(op_name)

        parent_dir = os.path.dirname(problem_def_path)
        category = os.path.basename(parent_dir)

        match = re.search(r"level(\d+)", parent_dir, re.IGNORECASE)
        level_id = match.group(1) if match else "0"

        return TaskMetadata(
            op_name=op_name,
            kernel_name=kernel_name,
            category=category,
            level_id=level_id,
            op_dir_name=op_dir_name,
            problem_def_path=problem_def_path,
        )

    @property
    def short_id(self) -> str:
        return f"lvl{self.level_id}_category{self.category}_{self.op_name}"


@dataclass(frozen=True)
class BuildPaths:
    """Paths for a compile unit (one per sample)."""
    work_dir: str
    src_dir: str
    build_dir: str
    log_dir: str

    compile_log: str
    code_gen_log: str
    prompt_save_file: str

    # Source locations
    module_path: str
    prepare_inputs_path: str
    test_cases_path: str
    tiling_cases_path: str
    template_path: str
    api_desc_path: str

    @staticmethod
    def from_metadata(
        meta: TaskMetadata,
        sample_id: int,
        run_dir: str,
        src_dir: str = "src",
        build_dir: str = "build",
        log_dir: str = "log",
        static_shape_mode: bool = True,
        case_id: Optional[int] = None,
    ) -> BuildPaths:
        """Compute all paths for a build unit."""
        short = meta.short_id

        if static_shape_mode and case_id is not None:
            work = f"{run_dir}/lvl{meta.level_id}/{meta.category}/{meta.op_dir_name}/fixed_case_{case_id}/sample{sample_id}"
        else:
            work = f"{run_dir}/lvl{meta.level_id}/{meta.category}/{meta.op_dir_name}/sample{sample_id}"

        src = f"{work}/{src_dir}"
        build = f"{work}/{build_dir}"
        log = f"{work}/{log_dir}"

        def _find_file(base: str, name: str, fallback_dir: str = "common") -> str:
            """Return primary path or common fallback."""
            primary = f"{base}/{name}"
            common = f"{os.path.dirname(base)}/../{fallback_dir}/{name}"
            return primary  # scanner verifies existence later

        base = meta.problem_def_path
        # Prefer test_cases.csv from answer/{sample_id}/, fallback to validation/
        answer_csv = f"{base}/answer/{sample_id}/test_cases.csv"
        test_cases = answer_csv if os.path.exists(answer_csv) else f"{base}/validation/test_cases.csv"

        return BuildPaths(
            work_dir=work,
            src_dir=src,
            build_dir=build,
            log_dir=log,
            compile_log=f"{log}/{short}_compile.log",
            code_gen_log=f"{log}/{short}_code_gen.log",
            prompt_save_file=f"{log}/{short}_rag_simple_code_gen_prompt.json",
            module_path=f"{base}/validation/module.py",
            prepare_inputs_path=f"{base}/validation/prepare_inputs.py",
            test_cases_path=test_cases,
            tiling_cases_path=f"{base}/validation/tiling_cases.csv",
            template_path=f"{base}/question",
            api_desc_path=f"{base}/question/api_desc.md",
        )


@dataclass(frozen=True)
class TestPaths:
    """Paths for a test unit (one per case)."""
    log_dir: str
    precision_log: str
    perf_log: str

    @staticmethod
    def from_build(build_work_dir: str, short_id: str, case_id: int) -> TestPaths:
        log_dir = f"{build_work_dir}/case_{case_id}/log"
        return TestPaths(
            log_dir=log_dir,
            precision_log=f"{log_dir}/{short_id}_case{case_id}_precision.log",
            perf_log=f"{log_dir}/{short_id}_case{case_id}_perf.log",
        )


@dataclass
class BuildHandle:
    """Compile unit: one compilation per (op, sample_id).

    In LLM mode, each sample is a separate LLM generation.
    In reference mode, sample_id maps to answer/{sample_id}/ directory.
    """
    metadata: TaskMetadata
    sample_id: int
    paths: BuildPaths

    # Filled by pipeline stages
    gen_content: Optional[str] = None
    tiling_def: Optional[str] = None
    tiling_content: Optional[str] = None
    kernel_content: Optional[str] = None

    code_gen_result: Optional[CodeGenResult] = None
    compile_result: Optional[CompileResult] = None

    @property
    def short_id(self) -> str:
        return f"{self.metadata.short_id}_sample{self.sample_id}"

    @property
    def answer_path(self) -> str:
        """Path to reference answer directory."""
        return f"{self.metadata.problem_def_path}/answer/{self.sample_id}"


@dataclass
class TestHandle:
    """Test unit: one precision/perf test per (sample_id, case_id).

    Shares the compiled artifacts from its parent BuildHandle.
    """
    build: BuildHandle
    case_id: int
    test_paths: TestPaths

    precision_result: Optional[PrecisionResult] = None
    perf_result: Optional[PerfResult] = None

    @property
    def short_id(self) -> str:
        return f"{self.build.short_id}_case{self.case_id}"


def _camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
