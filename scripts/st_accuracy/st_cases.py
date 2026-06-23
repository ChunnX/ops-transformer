from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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

    @property
    def inputs(self) -> list[Any]:
        values = self.raw.get("inputs")
        return values if isinstance(values, list) else []

    @property
    def flat_inputs(self) -> list[dict[str, Any]]:
        return flatten_input_items(self.inputs)

    @property
    def tensor_dtypes(self) -> list[str]:
        return [str(item.get("dtype")) for item in self.flat_inputs if item.get("type") == "tensor" and item.get("dtype")]

    @property
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
    repo_root = Path(root)
    cases: list[StCase] = []
    for operator in operators:
        st_root = repo_root / operator.path / "tests" / "st"
        if not st_root.is_dir():
            continue
        for json_file in sorted(st_root.glob(f"*/{json_pattern}")):
            api_dir = json_file.parent.name
            with json_file.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                continue
            for index, raw_case in enumerate(payload):
                if not isinstance(raw_case, dict):
                    continue
                case_id = int(raw_case.get("id", index))
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
    return cases
