from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .adapters import get_adapter
from .inventory import Operator
from .st_cases import StCase

ACTION_ORDER = {
    "implement_adapter_from_examples": 0,
    "extend_existing_adapter": 1,
    "inspect_api_assets": 2,
    "review_package_or_soc_support": 3,
    "documented_exclusion": 4,
    "covered": 5,
}


def classify_unsupported_reason(reason: str) -> str:
    text = reason or ""
    lower = text.lower()
    if lower.startswith("no adapter for "):
        return "missing_adapter"
    if "not supported on ascend910b" in lower or "eager example fails" in lower:
        return "package_soc_unsupported"
    if lower.startswith("scalar dtype ") or " scalar non-numeric range " in lower:
        return "scalar_gate"
    if lower.startswith("dtype "):
        return "dtype_gate"
    if "non-numeric range" in lower:
        return "range_gate"
    if "zero-sized tensor" in lower:
        return "zero_sized_gate"
    if "broadcastable" in lower or "shape" in lower:
        return "shape_gate"
    if any(token in lower for token in ("dims", "keepdims", "noop", "dtype attr", " attr")):
        return "attr_gate"
    if "expects " in lower and ("tensor input" in lower or "scalar input" in lower):
        return "signature_gate"
    return "adapter_other_gate"


def recommended_action_for_api(row: dict) -> str:
    unsupported = int(row.get("unsupported", 0) or 0)
    if unsupported == 0:
        return "covered"

    categories = dict(row.get("unsupported_categories") or {})
    if categories and set(categories) == {"package_soc_unsupported"}:
        return "documented_exclusion"

    if categories.get("missing_adapter"):
        if row.get("has_exact_example") and row.get("has_exact_op_api_header"):
            return "implement_adapter_from_examples"
        return "inspect_api_assets"

    if row.get("cmake", {}).get("mentions_ascend910b") is False and categories.get("package_soc_unsupported"):
        return "review_package_or_soc_support"

    return "extend_existing_adapter"


def action_sort_key(row: dict) -> tuple[int, int, str]:
    action = str(row.get("recommended_action") or recommended_action_for_api(row))
    return (ACTION_ORDER.get(action, 99), -int(row.get("unsupported", 0) or 0), str(row.get("api", "")))


def build_taxonomy_report(repo_root: Path | str, operators: Iterable[Operator], cases: Iterable[StCase]) -> dict:
    root = Path(repo_root)
    operator_list = list(operators)
    case_list = list(cases)
    operator_by_name = {operator.name: operator for operator in operator_list}
    by_api_cases: dict[str, list[StCase]] = defaultdict(list)
    by_op_cases: dict[str, list[StCase]] = defaultdict(list)

    for case in case_list:
        by_api_cases[case.aclnn_name].append(case)
        by_op_cases[case.op_name].append(case)

    by_api = []
    for api in sorted(by_api_cases):
        api_cases = by_api_cases[api]
        ops = sorted({case.op_name for case in api_cases})
        support_rows = [_case_support(case) for case in api_cases]
        unsupported_reasons = _reason_counts(support_rows)
        unsupported_categories = _category_counts(unsupported_reasons)
        metadata = _api_metadata(root, api, [operator_by_name[op] for op in ops if op in operator_by_name])
        row = {
            "api": api,
            "ops": ops,
            "cases": len(api_cases),
            "executable": sum(1 for supported, _ in support_rows if supported),
            "unsupported": sum(1 for supported, _ in support_rows if not supported),
            "unsupported_categories": dict(unsupported_categories),
            "unsupported_reasons": dict(unsupported_reasons),
            **metadata,
        }
        row["recommended_action"] = recommended_action_for_api(row)
        row["priority_score"] = int(row["unsupported"])
        by_api.append(row)

    by_op = []
    for operator in operator_list:
        op_cases = by_op_cases.get(operator.name, [])
        support_rows = [_case_support(case) for case in op_cases]
        unsupported_reasons = _reason_counts(support_rows)
        unsupported_categories = _category_counts(unsupported_reasons)
        by_op.append(
            {
                "op": operator.name,
                "category": operator.category,
                "path": operator.path.as_posix(),
                "apis": sorted({case.aclnn_name for case in op_cases}),
                "cases": len(op_cases),
                "executable": sum(1 for supported, _ in support_rows if supported),
                "unsupported": sum(1 for supported, _ in support_rows if not supported),
                "unsupported_categories": dict(unsupported_categories),
                "unsupported_reasons": dict(unsupported_reasons),
                "cmake": _cmake_signals(root, operator),
            }
        )

    priority = sorted((row for row in by_api if row["unsupported"] > 0), key=action_sort_key)
    unsupported_category_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for row in by_api:
        unsupported_category_counts.update(row["unsupported_categories"])
        action_counts[row["recommended_action"]] += 1

    executable_count = sum(row["executable"] for row in by_op)
    return {
        "summary": {
            "operator_count": len(operator_list),
            "api_count": len(by_api),
            "case_count": len(case_list),
            "executable_case_count": executable_count,
            "unsupported_case_count": len(case_list) - executable_count,
            "unsupported_category_counts": dict(unsupported_category_counts),
            "recommended_action_counts": dict(action_counts),
        },
        "by_api": by_api,
        "by_op": by_op,
        "priority": priority,
    }


def write_taxonomy_report(report: dict, out_dir: Path | str) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "taxonomy_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report["summary"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_jsonl(path / "taxonomy_by_api.jsonl", report["by_api"])
    _write_jsonl(path / "taxonomy_by_op.jsonl", report["by_op"])
    _write_jsonl(path / "taxonomy_priority.jsonl", report["priority"])


def _case_support(case: StCase) -> tuple[bool, str]:
    adapter = get_adapter(case.aclnn_name)
    if adapter is None:
        return False, f"no adapter for {case.aclnn_name}"
    return adapter.supports(case)


def _reason_counts(support_rows: list[tuple[bool, str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for supported, reason in support_rows:
        if not supported:
            counter[reason] += 1
    return counter


def _category_counts(unsupported_reasons: Counter[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for reason, count in unsupported_reasons.items():
        counter[classify_unsupported_reason(reason)] += count
    return counter


def _api_metadata(root: Path, api: str, operators: list[Operator]) -> dict:
    api_snake = _api_to_snake(api)
    example_files: list[str] = []
    op_api_headers: list[str] = []
    cmake_units: set[str] = set()
    mentions_ascend910b = False
    has_aclnn_exclude = False
    has_disable_in_opp = False

    for operator in operators:
        op_dir = root / operator.path
        examples_dir = op_dir / "examples"
        if examples_dir.is_dir():
            for example in sorted(examples_dir.rglob(f"test_aclnn_{api_snake}.cpp")):
                example_files.append(example.relative_to(root).as_posix())

        header = op_dir / "op_api" / f"aclnn_{api_snake}.h"
        if header.is_file():
            op_api_headers.append(header.relative_to(root).as_posix())

        cmake = _cmake_signals(root, operator)
        cmake_units.update(cmake["support_compute_units"])
        mentions_ascend910b = mentions_ascend910b or bool(cmake["mentions_ascend910b"])
        has_aclnn_exclude = has_aclnn_exclude or bool(cmake["has_aclnn_exclude"])
        has_disable_in_opp = has_disable_in_opp or bool(cmake["has_disable_in_opp"])

    return {
        "example_files": sorted(example_files),
        "has_exact_example": bool(example_files),
        "op_api_headers": sorted(op_api_headers),
        "has_exact_op_api_header": bool(op_api_headers),
        "cmake": {
            "support_compute_units": sorted(cmake_units),
            "mentions_ascend910b": mentions_ascend910b,
            "has_aclnn_exclude": has_aclnn_exclude,
            "has_disable_in_opp": has_disable_in_opp,
        },
    }


def _cmake_signals(root: Path, operator: Operator) -> dict:
    cmake_file = root / operator.path / "CMakeLists.txt"
    try:
        text = cmake_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        text = ""

    units: set[str] = set()
    for match in re.finditer(r"set\s*\(\s*SUPPORT_COMPUTE_UNIT\s+([^)]*)\)", text, flags=re.IGNORECASE | re.MULTILINE):
        units.update(re.findall(r'"([^"]+)"', match.group(1)))
        bare = re.sub(r'"[^"]+"', " ", match.group(1))
        units.update(token for token in re.split(r"\s+", bare.strip()) if token and not token.startswith("$"))

    upper = text.upper()
    return {
        "support_compute_units": sorted(units),
        "mentions_ascend910b": "ascend910b" in text.lower(),
        "has_aclnn_exclude": "ACLNNTYPE" in upper and "ACLNN_EXCLUDE" in upper,
        "has_disable_in_opp": "DISABLE_IN_OPP" in upper and "TRUE" in upper,
    }


def _api_to_snake(api: str) -> str:
    name = api
    if name.startswith("aclnn") and len(name) > 5 and name[5].isupper():
        name = name[5:]
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", name)
    name = re.sub(r"([0-9])([A-Za-z])", r"\1_\2", name)
    return name.lower()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
