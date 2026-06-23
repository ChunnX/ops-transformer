from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable

from .inventory import Operator


@dataclass(frozen=True)
class StCase:
    op_name: str
    category: str
    op_path: Path
    api_dir: str
    aclnn_name: str
    json_path: Path
    case_id: int
    raw: dict[str, Any]

    @cached_property
    def inputs(self) -> list[Any]:
        values = self.raw.get("inputs")
        return values if isinstance(values, list) else []

    @cached_property
    def flat_inputs(self) -> list[dict[str, Any]]:
        return flatten_input_items(self.inputs)

    @cached_property
    def tensor_items(self) -> list[dict[str, Any]]:
        return [item for item in self.flat_inputs if item.get("type") == "tensor"]

    @cached_property
    def tensor_list_items(self) -> list[dict[str, Any]]:
        return [item for item in self.flat_inputs if item.get("type") == "tensors"]

    @cached_property
    def scalar_items(self) -> list[dict[str, Any]]:
        return [item for item in self.flat_inputs if item.get("type") == "scalar"]

    @cached_property
    def tensor_dtypes(self) -> list[str]:
        return [str(item.get("dtype")) for item in self.tensor_items if item.get("dtype")]

    @cached_property
    def all_dtypes(self) -> list[str]:
        return [str(item.get("dtype")) for item in self.flat_inputs if item.get("dtype")]

    def to_result(self, status: str, message: str = "", duration_ms: float | None = None, **extra: Any) -> "CaseResult":
        return CaseResult(
            op_name=self.op_name,
            api=self.aclnn_name,
            case_id=self.case_id,
            json_path=self.json_path.as_posix(),
            status=status,
            message=message,
            duration_ms=duration_ms,
            extra=extra,
        )


@dataclass(frozen=True)
class CaseResult:
    op_name: str
    api: str
    case_id: int
    json_path: str
    status: str
    message: str = ""
    duration_ms: float | None = None
    extra: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["duration_ms"] is None:
            payload.pop("duration_ms")
        if not payload.get("extra"):
            payload.pop("extra", None)
        return payload


@dataclass(frozen=True)
class StAssetDiagnostic:
    op_name: str
    category: str
    op_path: Path
    json_path: Path
    reason: str
    message: str
    case_index: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "op": self.op_name,
            "category": self.category,
            "path": self.op_path.as_posix(),
            "json_path": self.json_path.as_posix(),
            "reason": self.reason,
            "message": self.message,
            "case_index": self.case_index,
        }


@dataclass(frozen=True)
class CaseLoadResult:
    cases: list[StCase]
    skipped_assets: list[StAssetDiagnostic]


def _case_aclnn_name(api_dir: str, case: dict[str, Any]) -> str:
    value = case.get("aclnn_name")
    if value:
        return str(value)
    return api_dir.removeprefix("aclnn")


def flatten_input_items(values: Iterable[Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for item in values:
        if isinstance(item, dict):
            flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(child for child in item if isinstance(child, dict))
    return flattened


def load_cases(root: Path | str, operators: Iterable[Operator], json_pattern: str = "all_*.json") -> list[StCase]:
    return load_cases_with_diagnostics(root, operators, json_pattern=json_pattern).cases


def load_cases_with_diagnostics(
    root: Path | str, operators: Iterable[Operator], json_pattern: str = "all_*.json"
) -> CaseLoadResult:
    repo_root = Path(root)
    cases: list[StCase] = []
    skipped_assets: list[StAssetDiagnostic] = []
    for operator in operators:
        st_root = repo_root / operator.path / "tests" / "st"
        if not st_root.is_dir():
            skipped_assets.append(
                _diagnostic(
                    operator,
                    st_root.relative_to(repo_root),
                    "missing_st_dir",
                    "operator has no tests/st directory",
                )
            )
            continue
        json_files = sorted(st_root.glob(f"*/{json_pattern}"))
        if not json_files:
            skipped_assets.append(
                _diagnostic(
                    operator,
                    st_root.relative_to(repo_root),
                    "no_json_files",
                    f"tests/st has no files matching */{json_pattern}",
                )
            )
            continue
        for json_file in json_files:
            api_dir = json_file.parent.name
            json_path = json_file.relative_to(repo_root)
            try:
                with json_file.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
            except json.JSONDecodeError as exc:
                skipped_assets.append(_diagnostic(operator, json_path, "invalid_json", str(exc)))
                continue
            except OSError as exc:
                skipped_assets.append(_diagnostic(operator, json_path, "read_error", str(exc)))
                continue
            if not isinstance(payload, list):
                skipped_assets.append(_diagnostic(operator, json_path, "payload_not_list", "ST JSON payload is not a list"))
                continue
            valid_cases = 0
            for index, raw_case in enumerate(payload):
                if not isinstance(raw_case, dict):
                    skipped_assets.append(
                        _diagnostic(operator, json_path, "case_not_object", "ST case is not an object", case_index=index)
                    )
                    continue
                try:
                    case_id = int(raw_case.get("id", index))
                except (TypeError, ValueError):
                    skipped_assets.append(
                        _diagnostic(operator, json_path, "invalid_case_id", "ST case id is not an integer", case_index=index)
                    )
                    continue
                cases.append(
                    StCase(
                        op_name=operator.name,
                        category=operator.category,
                        op_path=operator.path,
                        api_dir=api_dir,
                        aclnn_name=_case_aclnn_name(api_dir, raw_case),
                        json_path=json_file.relative_to(repo_root),
                        case_id=case_id,
                        raw=raw_case,
                    )
                )
                valid_cases += 1
            if not valid_cases:
                skipped_assets.append(_diagnostic(operator, json_path, "empty_case_file", "ST JSON has no valid cases"))
    return CaseLoadResult(cases=cases, skipped_assets=skipped_assets)


def _diagnostic(
    operator: Operator,
    json_path: Path,
    reason: str,
    message: str,
    case_index: int | None = None,
) -> StAssetDiagnostic:
    return StAssetDiagnostic(
        op_name=operator.name,
        category=operator.category,
        op_path=operator.path,
        json_path=json_path,
        reason=reason,
        message=message,
        case_index=case_index,
    )
