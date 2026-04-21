"""Create BuildHandle / TestHandle instances from task definitions."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Optional

from npukernelbench.config.models import FullConfig
from npukernelbench.models.task import (
    TaskMetadata,
    BuildPaths,
    BuildHandle,
    TestHandle,
    TestPaths,
)


def create_build_handles(
    task_paths: List[str],
    config: FullConfig,
    sample_ids: Optional[List[int]] = None,
) -> List[BuildHandle]:
    """Create BuildHandle for each (task, sample) combination.

    Args:
        task_paths: Absolute paths to operator definition directories.
        config: Full configuration.
        sample_ids: Explicit sample IDs. Defaults to range(config.n_sample).
    """
    if sample_ids is None:
        sample_ids = list(range(config.n_sample))

    handles: List[BuildHandle] = []
    for path in task_paths:
        meta = TaskMetadata.from_path(path)
        for sid in sample_ids:
            case_ids = _read_case_ids(meta, config, sid)
            # Use first case_id for path computation in static mode
            case_id = case_ids[0] if (config.static_shape_mode and case_ids) else None
            paths = BuildPaths.from_metadata(
                meta=meta,
                sample_id=sid,
                run_dir=config.run_dir,
                src_dir=config.src_dir,
                build_dir=config.build_dir,
                log_dir=config.log_dir,
                static_shape_mode=config.static_shape_mode,
                case_id=case_id,
            )
            handles.append(BuildHandle(metadata=meta, sample_id=sid, paths=paths))

    return handles


def expand_test_handles(
    builds: List[BuildHandle],
    config: FullConfig,
) -> List[TestHandle]:
    """Expand BuildHandles into TestHandles (one per case).

    In static_shape_mode, each case is a separate test.
    In dynamic mode, there is one test per build (case_id=0).
    """
    tests: List[TestHandle] = []
    for build in builds:
        case_ids = _read_case_ids(build.metadata, config, build.sample_id)
        for cid in case_ids:
            test_paths = TestPaths.from_build(
                build_work_dir=build.paths.work_dir,
                short_id=build.short_id,
                case_id=cid,
            )
            tests.append(TestHandle(build=build, case_id=cid, test_paths=test_paths))
    return tests


def _read_case_ids(meta: TaskMetadata, config: FullConfig, sample_id: int = 0) -> List[int]:
    """Read case IDs from test_cases.csv, respecting n_case limit.

    Prefers answer/{sample_id}/test_cases.csv, falls back to validation/test_cases.csv.
    """
    answer_csv = f"{meta.problem_def_path}/answer/{sample_id}/test_cases.csv"
    csv_path = answer_csv if os.path.exists(answer_csv) else f"{meta.problem_def_path}/validation/test_cases.csv"
    if not os.path.exists(csv_path):
        return [0]

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            rows = list(reader)
    except Exception:
        return [0]

    if not rows:
        return [0]

    case_ids = list(range(len(rows)))

    # Respect n_case limit (0 or None means all)
    n = config.n_case
    if n and n > 0:
        case_ids = case_ids[:n]

    return case_ids
