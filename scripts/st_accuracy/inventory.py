from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ASCENDC_FEATURES = (
    "kernel_operator",
    "__aicore__",
    "__global__",
    "GlobalTensor",
    "TPipe",
    "TQue",
    "GET_TILING_DATA",
)

DEFAULT_CATEGORIES = ("attention",)


@dataclass(frozen=True)
class Operator:
    name: str
    category: str
    path: Path


def _has_ascendc_kernel(op_dir: Path) -> bool:
    kernel_dir = op_dir / "op_kernel"
    if not kernel_dir.is_dir():
        return False

    for source in kernel_dir.rglob("*"):
        if source.suffix not in {".cpp", ".cc", ".cxx", ".h", ".hpp"}:
            continue
        try:
            text = source.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(feature in text for feature in ASCENDC_FEATURES):
            return True
    return False


def _is_optimizable_operator(op_dir: Path) -> bool:
    if not (op_dir / "CMakeLists.txt").is_file():
        return False
    if not list((op_dir / "op_host").rglob("*_def*.cpp")):
        return False
    return _has_ascendc_kernel(op_dir)


def discover_operators(root: Path | str, categories: Iterable[str] = DEFAULT_CATEGORIES) -> list[Operator]:
    repo_root = Path(root)
    operators: list[Operator] = []
    for category in categories:
        category_dir = repo_root / category
        if not category_dir.is_dir():
            continue
        for op_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            if _is_optimizable_operator(op_dir):
                operators.append(Operator(name=op_dir.name, category=category, path=op_dir.relative_to(repo_root)))
    return operators
