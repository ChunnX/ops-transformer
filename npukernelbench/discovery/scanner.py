"""Scan the tasks/ directory for operator definitions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


def scan_tasks(tasks_dir: str | Path) -> List[str]:
    """Walk tasks/level*/category/operator and return sorted absolute paths.

    Returns:
        List of absolute paths to operator definition directories.
    """
    tasks_dir = Path(tasks_dir).resolve()
    if not tasks_dir.is_dir():
        return []

    result: List[str] = []
    for level_dir in sorted(tasks_dir.iterdir()):
        if not level_dir.is_dir() or not level_dir.name.startswith("level"):
            continue
        for category_dir in sorted(level_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for op_dir in sorted(category_dir.iterdir()):
                if op_dir.is_dir():
                    # Verify it has a validation/ directory (basic sanity)
                    if (op_dir / "validation").is_dir():
                        result.append(str(op_dir))

    return result


def filter_by_name(task_paths: List[str], names: List[str]) -> List[str]:
    """Filter task paths by operator directory name."""
    if not names:
        return task_paths

    name_set = set(n.strip() for n in names)
    filtered = [p for p in task_paths if Path(p).name in name_set]

    found = {Path(p).name for p in filtered}
    missing = name_set - found
    if missing:
        print(f"[WARNING] Tasks not found: {', '.join(missing)}")

    return filtered
