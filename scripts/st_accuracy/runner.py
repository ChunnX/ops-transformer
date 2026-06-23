from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .adapters import get_adapter
from .native import NativeRunner
from .st_cases import CaseResult, StCase
from .timing import TimingCollector


@dataclass(frozen=True)
class CaseSelector:
    ops: set[str] | None = None
    apis: set[str] | None = None
    dtypes: set[str] | None = None
    case_ids: set[int] | None = None
    limit: int | None = None

    def apply(self, cases: Iterable[StCase]) -> list[StCase]:
        selected: list[StCase] = []
        for case in cases:
            if self.ops and case.op_name not in self.ops:
                continue
            normalized_api = case.aclnn_name.removeprefix("aclnn")
            if (
                self.apis
                and case.aclnn_name not in self.apis
                and case.api_dir not in self.apis
                and normalized_api not in self.apis
            ):
                continue
            if self.case_ids and case.case_id not in self.case_ids:
                continue
            if self.dtypes and not (set(case.all_dtypes) & self.dtypes):
                continue
            selected.append(case)
            if self.limit is not None and len(selected) >= self.limit:
                break
        return selected


def run_cases(
    cases: Iterable[StCase],
    repo_root: Path,
    out_dir: Path,
    vendor: str,
    soc: str,
    device: int,
    execute: bool = True,
    timings: TimingCollector | None = None,
) -> list[CaseResult]:
    native_runner = NativeRunner(repo_root=repo_root, out_dir=out_dir, vendor=vendor, soc=soc, device=device, timings=timings)
    results: list[CaseResult | None] = []
    executable: list[tuple[int, StCase, object]] = []
    with timings.measure("support_gate_ms") if timings is not None else nullcontext():
        for case in cases:
            adapter = get_adapter(case.aclnn_name)
            if adapter is None:
                results.append(case.to_result(status="unsupported", message=f"no adapter for {case.aclnn_name}"))
                continue
            supported, reason = adapter.supports(case)
            if not supported:
                results.append(case.to_result(status="unsupported", message=reason))
                continue
            if not execute:
                results.append(case.to_result(status="skipped", message="execution disabled"))
                continue
            results.append(None)
            executable.append((len(results) - 1, case, adapter))

    if executable:
        batch_results = native_runner.run_many([(case, adapter) for _index, case, adapter in executable])
        for (index, _case, _adapter), result in zip(executable, batch_results):
            results[index] = result

    return [result for result in results if result is not None]


def parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_int_csv(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}
