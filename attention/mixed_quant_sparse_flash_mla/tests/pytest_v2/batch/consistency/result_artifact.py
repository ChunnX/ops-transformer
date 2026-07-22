#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Machine-readable live NPU result artifacts and deterministic aggregation."""

import argparse
from collections import Counter, defaultdict
from enum import Enum
import hashlib
import json
import os
import re
import tempfile
from typing import Any, Dict, Iterable, Optional

from .runtime_capability import (
    RuntimeCapabilityError,
    SUPPORTED,
    validate_runtime_capability_report,
)
from .suite_config import ResolvedCaseConfig, case_identity


SCHEMA_VERSION = 1
RESULT_KIND = "smla_mqsmla_batch_consistency_result"
SUMMARY_KIND = "smla_mqsmla_batch_consistency_result_summary"
SCENARIO = "same-source-cross-call"
EXPECTED_MODES = ("reorder", "split", "token-split", "shape-change")
SUPPLEMENTAL_SCENARIOS = ("independent-cross-case", "single-call-intra-case")


class ResultStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID_TRANSFORM = "INVALID_TRANSFORM"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CONFIG_ERROR = "CONFIG_ERROR"


class ResultArtifactError(ValueError):
    """Raised when a result could be mistaken for a valid level-3 verdict."""


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def _atomic_json_dump(path: str, value: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _deterministic_levels(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    level = report.get("deterministic_level")
    if isinstance(level, dict):
        yield level
    subcases = report.get("subcases", [])
    if not isinstance(subcases, (list, tuple)):
        return
    for subcase in subcases:
        if isinstance(subcase, dict):
            yield from _deterministic_levels(subcase)


def _validate_runtime_capability_evidence(evidence: Dict[str, Any]) -> None:
    if not isinstance(evidence, dict):
        raise ResultArtifactError("Runtime capability evidence must be an object")
    source = evidence.get("source_report")
    if not isinstance(source, dict):
        raise ResultArtifactError("Runtime capability source report is missing")
    if not isinstance(source.get("path"), str) or not os.path.isabs(source["path"]):
        raise ResultArtifactError("Runtime capability source path must be absolute")
    for field in ("sha256", "probe_sha256"):
        value = source.get(field) if field == "sha256" else evidence.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ResultArtifactError(f"Runtime capability {field} is not a SHA256 digest")
    try:
        probe = evidence.get("probe")
        validate_runtime_capability_report(probe)
    except RuntimeCapabilityError as error:
        raise ResultArtifactError(f"Invalid runtime capability evidence: {error}") from error
    canonical = json.dumps(
        probe, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != evidence["probe_sha256"]:
        raise ResultArtifactError("Runtime capability embedded probe digest mismatch")


def validate_result_artifact(artifact: Dict[str, Any]) -> None:
    if not isinstance(artifact, dict):
        raise ResultArtifactError("Result artifact root must be an object")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ResultArtifactError(
            f"Unsupported result schema_version={artifact.get('schema_version')}"
        )
    if artifact.get("kind") != RESULT_KIND:
        raise ResultArtifactError(f"Unexpected result kind={artifact.get('kind')!r}")
    try:
        status = ResultStatus(artifact.get("status"))
    except (TypeError, ValueError) as error:
        raise ResultArtifactError(f"Unknown result status={artifact.get('status')!r}") from error
    case = artifact.get("case")
    if not isinstance(case, dict):
        raise ResultArtifactError("Result case identity must be an object")
    fingerprint = case.get("input_content_fingerprint_sha256")
    source_pt = case.get("source_pt")
    if not fingerprint or not source_pt:
        raise ResultArtifactError("Result case identity is incomplete")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ResultArtifactError("Result case fingerprint is not a SHA256 hex digest")
    if not isinstance(source_pt, str) or not os.path.isabs(source_pt):
        raise ResultArtifactError("Result source_pt must be an absolute path")
    for field in ("case_name", "pt_basename", "detected_operator"):
        if not isinstance(case.get(field), str) or not case[field]:
            raise ResultArtifactError(f"Result case {field} must be a non-empty string")
    if artifact.get("operator") not in ("smla", "mqsmla"):
        raise ResultArtifactError(f"Unknown result operator={artifact.get('operator')!r}")
    if artifact.get("mode") not in EXPECTED_MODES:
        raise ResultArtifactError("Result operator/mode is incomplete")
    if artifact.get("scenario") != SCENARIO:
        raise ResultArtifactError(f"Unexpected result scenario={artifact.get('scenario')!r}")
    runtime_capability = artifact.get("runtime_capability")
    if runtime_capability is not None:
        _validate_runtime_capability_evidence(runtime_capability)
    report = artifact.get("report")
    if status is not ResultStatus.PASS:
        for field in ("reason_code", "reason"):
            if not isinstance(artifact.get(field), str) or not artifact[field]:
                raise ResultArtifactError(f"{status.value} requires {field}")
    if status is ResultStatus.PASS:
        if case.get("detected_operator") != artifact["operator"]:
            raise ResultArtifactError("PASS operator does not match the loaded PT")
        if not isinstance(report, dict) or report.get("pass") is not True:
            raise ResultArtifactError("PASS requires a successful execution report")
        levels = list(_deterministic_levels(report))
        if not levels or any(
            item.get("requested") != 3 or item.get("observed") != 3 for item in levels
        ):
            raise ResultArtifactError("PASS requires observed deterministic level 3")
        if runtime_capability is None:
            raise ResultArtifactError("PASS requires runtime capability evidence")
        if (
            runtime_capability["probe"]["status"] != SUPPORTED
        ):
            raise ResultArtifactError("PASS conflicts with a non-supported runtime capability probe")


def write_result_artifact(
    report_dir: str,
    *,
    operator: str,
    input_data: Dict[str, Any],
    pt_path: str,
    mode: str,
    status: ResultStatus,
    config: Optional[ResolvedCaseConfig] = None,
    report: Optional[Dict[str, Any]] = None,
    reason_code: Optional[str] = None,
    reason: Optional[str] = None,
    runtime_capability: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist one final status for exactly one operator/case/mode tuple."""
    identity = case_identity(input_data, pt_path)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "claim_scope": (
            "level-3 NPU batch-consistency result"
            if status is ResultStatus.PASS
            else "non-PASS execution/configuration status; no NPU PASS claim"
        ),
        "scenario": SCENARIO,
        "operator": operator,
        "mode": mode,
        "status": status.value,
        "case": {
            "case_name": identity["case_name"],
            "detected_operator": identity["operator"],
            "source_pt": os.path.abspath(pt_path),
            "pt_basename": identity["pt_basename"],
            "input_content_fingerprint_sha256": identity[
                "input_content_fingerprint_sha256"
            ],
        },
        "config": config.to_dict() if config is not None else None,
        "reason_code": reason_code,
        "reason": reason,
        "runtime_capability": runtime_capability,
        "report": report,
    }
    validate_result_artifact(artifact)
    file_name = (
        f"{_safe_name(operator)}__{_safe_name(identity['pt_basename'])}__"
        f"{_safe_name(identity['case_name'])}__"
        f"{identity['input_content_fingerprint_sha256'][:12]}__{_safe_name(mode)}.json"
    )
    path = os.path.join(os.path.abspath(report_dir), "results", file_name)
    _atomic_json_dump(path, artifact)
    return path


def aggregate_result_artifacts(
    report_dir: str,
    operator: Optional[str] = None,
    expected_case_count: Optional[int] = None,
    execution_failure_count: int = 0,
) -> Dict[str, Any]:
    """Aggregate only validated per-mode artifacts; pytest text is never parsed."""
    if expected_case_count is not None and expected_case_count < 0:
        raise ResultArtifactError("expected_case_count must be >= 0")
    if execution_failure_count < 0:
        raise ResultArtifactError("execution_failure_count must be >= 0")
    result_dir = os.path.join(os.path.abspath(report_dir), "results")
    candidates = []
    validation_errors = []
    foreign_artifacts = []
    if os.path.isdir(result_dir):
        for name in sorted(os.listdir(result_dir)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(result_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    artifact = json.load(file)
                validate_result_artifact(artifact)
            except (OSError, UnicodeError, json.JSONDecodeError, ResultArtifactError) as error:
                validation_errors.append(
                    {"artifact": os.path.abspath(path), "error": f"{type(error).__name__}: {error}"}
                )
                continue
            if operator is not None and artifact["operator"] != operator:
                foreign_artifacts.append(
                    {
                        "artifact": os.path.abspath(path),
                        "operator": artifact["operator"],
                    }
                )
                continue
            candidates.append((path, artifact))

    mode_counts: Dict[str, Counter] = defaultdict(Counter)
    operator_counts: Dict[str, Counter] = defaultdict(Counter)
    cases = set()
    tuples = set()
    artifacts = []
    records = []
    runtime_capability_reports: Dict[str, Dict[str, Any]] = {}
    for path, item in candidates:
        case_key = (
            item["operator"],
            item["case"]["source_pt"],
            item["case"]["input_content_fingerprint_sha256"],
        )
        result_key = case_key + (item["mode"],)
        if result_key in tuples:
            validation_errors.append(
                {
                    "artifact": os.path.abspath(path),
                    "error": f"Duplicate operator/case/mode result: {result_key}",
                }
            )
            continue
        tuples.add(result_key)
        cases.add(case_key)
        artifacts.append((path, item))
        mode_counts[item["mode"]][item["status"]] += 1
        operator_counts[item["operator"]][item["status"]] += 1
        records.append(
            {
                "operator": item["operator"],
                "case_name": item["case"]["case_name"],
                "source_pt": item["case"]["source_pt"],
                "input_content_fingerprint_sha256": item["case"][
                    "input_content_fingerprint_sha256"
                ],
                "mode": item["mode"],
                "status": item["status"],
                "reason_code": item.get("reason_code"),
                "runtime_capability_probe_sha256": (
                    item["runtime_capability"]["probe_sha256"]
                    if item.get("runtime_capability")
                    else None
                ),
                "artifact": os.path.abspath(path),
            }
        )
        capability = item.get("runtime_capability")
        if capability:
            probe_hash = capability["probe_sha256"]
            runtime_capability_reports[probe_hash] = {
                "probe_sha256": probe_hash,
                "source_report": capability["source_report"],
                "status": capability["probe"]["status"],
                "reason_code": capability["probe"].get("reason_code"),
                "failure_stage": capability["probe"].get("failure_stage"),
                "attempt_count": capability["probe"]["attempt_count_executed"],
                "restoration_pass": capability["probe"]["restoration_pass"],
            }

    capability_artifact_count = sum(
        1 for _, item in artifacts if item.get("runtime_capability") is not None
    )
    if capability_artifact_count not in (0, len(artifacts)):
        validation_errors.append(
            {
                "artifact": None,
                "error": "Runtime capability evidence is missing from part of the result matrix",
            }
        )
    if len(runtime_capability_reports) > 1:
        validation_errors.append(
            {
                "artifact": None,
                "error": "Result matrix is bound to multiple runtime capability probes",
            }
        )

    status_counts = Counter(item["status"] for _, item in artifacts)
    blocking = (
        status_counts[ResultStatus.FAIL.value]
        + status_counts[ResultStatus.CONFIG_ERROR.value]
    )
    partial = (
        status_counts[ResultStatus.UNSUPPORTED.value]
        + status_counts[ResultStatus.INVALID_TRANSFORM.value]
    )
    passed = status_counts[ResultStatus.PASS.value]
    expected_artifacts = (
        expected_case_count * len(EXPECTED_MODES)
        if expected_case_count is not None
        else None
    )
    missing_artifacts = (
        max(expected_artifacts - len(artifacts), 0)
        if expected_artifacts is not None
        else 0
    )
    unexpected_artifacts = (
        max(len(artifacts) - expected_artifacts, 0) + len(foreign_artifacts)
        if expected_artifacts is not None
        else len(foreign_artifacts)
    )
    missing_cases = (
        max(expected_case_count - len(cases), 0)
        if expected_case_count is not None
        else 0
    )
    unexpected_cases = (
        max(len(cases) - expected_case_count, 0)
        if expected_case_count is not None
        else 0
    )
    if blocking or execution_failure_count or validation_errors:
        overall = "FAIL"
    elif missing_artifacts or unexpected_artifacts or missing_cases or unexpected_cases:
        overall = "INCOMPLETE"
    elif passed and partial:
        overall = "PARTIAL"
    elif passed:
        overall = "PASS"
    elif status_counts[ResultStatus.UNSUPPORTED.value]:
        overall = "UNSUPPORTED"
    elif artifacts:
        overall = "INCOMPLETE"
    else:
        overall = "NO_RESULTS"

    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": SUMMARY_KIND,
        "claim_scope": (
            "PASS only when at least one level-3 result passed and no executable result "
            "is unsupported, invalid, failed, or misconfigured"
        ),
        "operator_filter": operator,
        "scenario_coverage": {
            "implemented": [SCENARIO],
            "supplemental_gaps": list(SUPPLEMENTAL_SCENARIOS),
        },
        "overall_status": overall,
        "pass": overall == "PASS",
        "artifact_count": len(artifacts),
        "case_count": len(cases),
        "expected_case_count": expected_case_count,
        "expected_artifact_count": expected_artifacts,
        "missing_artifact_count": missing_artifacts,
        "unexpected_artifact_count": unexpected_artifacts,
        "missing_case_count": missing_cases,
        "unexpected_case_count": unexpected_cases,
        "foreign_artifact_count": len(foreign_artifacts),
        "foreign_artifacts": foreign_artifacts,
        "artifact_validation_error_count": len(validation_errors),
        "artifact_validation_errors": validation_errors,
        "pytest_process_failure_count": execution_failure_count,
        "runtime_capability_reports": [
            runtime_capability_reports[key] for key in sorted(runtime_capability_reports)
        ],
        "status_counts": {
            status.value: status_counts[status.value] for status in ResultStatus
        },
        "mode_status_counts": {
            mode: {status.value: counts[status.value] for status in ResultStatus}
            for mode, counts in sorted(mode_counts.items())
        },
        "operator_status_counts": {
            name: {status.value: counts[status.value] for status in ResultStatus}
            for name, counts in sorted(operator_counts.items())
        },
        "records": records,
    }
    summary_path = os.path.join(os.path.abspath(report_dir), "consistency_result_summary.json")
    _atomic_json_dump(summary_path, summary)
    summary["summary_path"] = summary_path
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate consistency result artifacts")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--operator", choices=("smla", "mqsmla"))
    parser.add_argument("--expected-case-count", type=int)
    parser.add_argument("--execution-failure-count", type=int, default=0)
    args = parser.parse_args()
    summary = aggregate_result_artifacts(
        args.report_dir,
        operator=args.operator,
        expected_case_count=args.expected_case_count,
        execution_failure_count=args.execution_failure_count,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if summary["overall_status"] == "PASS":
        return 0
    if summary["overall_status"] == "FAIL":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
