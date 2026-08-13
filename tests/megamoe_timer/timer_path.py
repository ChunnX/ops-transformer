import os
from pathlib import Path
from typing import Optional


DEFAULT_TIMER_ROOT = Path(__file__).resolve().parents[1] / "megamoe_w4a8" / "timer"


def resolve_timer_dir(path: Optional[str], file_pattern: str) -> Path:
    if path:
        timer_root = Path(path).expanduser()
    else:
        timer_root = Path(
            os.environ.get("MEGAMOE_TIMER_DIR", str(DEFAULT_TIMER_ROOT))
        ).expanduser()
        case_name = os.environ.get("MEGAMOE_CASE_NAME")
        if case_name:
            timer_root = timer_root / case_name

    if not timer_root.exists():
        raise FileNotFoundError(f"timer directory does not exist: {timer_root}")
    if any(timer_root.glob(file_pattern)):
        return timer_root

    case_dirs = [
        child
        for child in timer_root.iterdir()
        if child.is_dir() and any(child.glob(file_pattern))
    ]
    if not case_dirs:
        raise FileNotFoundError(
            f"no {file_pattern} files found under timer directory: {timer_root}"
        )

    def latest_file_mtime(case_dir: Path) -> float:
        return max(path.stat().st_mtime for path in case_dir.glob(file_pattern))

    return max(case_dirs, key=latest_file_mtime)
