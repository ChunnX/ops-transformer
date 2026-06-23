from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any, Iterable

from .adapters import get_adapter
from .inventory import Operator
from .st_cases import StCase


def build_coverage_report(operators: Iterable[Operator], cases: Iterable[StCase]) -> dict:
    operator_list = list(operators)
    case_list = list(cases)
    by_op_cases: dict[str, list[StCase]] = defaultdict(list)
    by_api_cases: dict[str, list[StCase]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()

    for case in case_list:
        by_op_cases[case.op_name].append(case)
        by_api_cases[case.aclnn_name].append(case)
        supported, reason = _case_support(case)
        if not supported:
            reason_counts[reason] += 1

    by_op = []
    for operator in operator_list:
        op_cases = by_op_cases.get(operator.name, [])
        executable = sum(1 for case in op_cases if _case_support(case)[0])
        unsupported = len(op_cases) - executable
        zero_sized = sum(1 for case in op_cases if _case_has_zero_sized_tensor(case))
        by_op.append(
            {
                "op": operator.name,
                "category": operator.category,
                "path": operator.path.as_posix(),
                "apis": sorted({case.aclnn_name for case in op_cases}),
                "cases": len(op_cases),
                "executable": executable,
                "unsupported": unsupported,
                "zero_sized_cases": zero_sized,
                "unsupported_reasons": dict(_reason_counter(op_cases)),
            }
        )

    by_api = []
    for api_name in sorted(by_api_cases):
        api_cases = by_api_cases[api_name]
        executable = sum(1 for case in api_cases if _case_support(case)[0])
        unsupported = len(api_cases) - executable
        zero_sized = sum(1 for case in api_cases if _case_has_zero_sized_tensor(case))
        by_api.append(
            {
                "api": api_name,
                "ops": sorted({case.op_name for case in api_cases}),
                "cases": len(api_cases),
                "executable": executable,
                "unsupported": unsupported,
                "zero_sized_cases": zero_sized,
                "unsupported_reasons": dict(_reason_counter(api_cases)),
            }
        )

    executable_count = sum(row["executable"] for row in by_op)
    zero_sized_count = sum(1 for case in case_list if _case_has_zero_sized_tensor(case))
    return {
        "summary": {
            "operator_count": len(operator_list),
            "case_count": len(case_list),
            "executable_case_count": executable_count,
            "unsupported_case_count": len(case_list) - executable_count,
            "zero_sized_case_count": zero_sized_count,
            "unsupported_reason_counts": dict(reason_counts),
        },
        "by_op": by_op,
        "by_api": by_api,
    }


def write_coverage_report(report: dict, out_dir: Path | str) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "coverage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report["summary"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_jsonl(path / "coverage_by_op.jsonl", report["by_op"])
    _write_jsonl(path / "coverage_by_api.jsonl", report["by_api"])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _reason_counter(cases: list[StCase]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for case in cases:
        supported, reason = _case_support(case)
        if not supported:
            counter[reason] += 1
    return counter


def _case_support(case: StCase) -> tuple[bool, str]:
    adapter = get_adapter(case.aclnn_name)
    if adapter is None:
        return False, f"no adapter for {case.aclnn_name}"
    return adapter.supports(case)


def _case_has_zero_sized_tensor(case: StCase) -> bool:
    for item in case.flat_inputs:
        if item.get("type") != "tensor":
            continue
        shape = item.get("shape")
        if isinstance(shape, list) and _numel(shape) == 0:
            return True
    return False


def _numel(shape: Any) -> int:
    if not isinstance(shape, list):
        return 1
    if not shape:
        return 1
    try:
        return int(reduce(mul, (int(dim) for dim in shape), 1))
    except (TypeError, ValueError):
        return 0
