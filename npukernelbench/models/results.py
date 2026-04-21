"""Result dataclasses for each pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class CodeGenResult:
    success: bool
    log_file: str = ""
    response: str = ""
    gen_content: str = ""
    reasoning_content: str = ""


@dataclass
class CompileResult:
    success: bool
    log_file: str = ""


@dataclass
class PrecisionResult:
    success: bool
    n_success: int = 0
    n_failed: int = 0
    log_file: str = ""
    results_dict: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PerfResult:
    success: bool
    n_success: int = 0
    n_failed: int = 0
    log_file: str = ""
    results_dict: Dict[int, Any] = field(default_factory=dict)
