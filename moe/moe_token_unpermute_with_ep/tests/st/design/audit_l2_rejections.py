#!/usr/bin/env python3
"""Audit that every ACLNN L2 case is rejected by this operator, not by TTK infrastructure."""

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),")
PID = re.compile(r"\[(\d+) [^]]+ OnProfiling]")
INFRA_FAILURES = (
    "Check nnopExecutor != nullptr failed",
    "operator package to which the MoeTokenUnpermuteWithEp operator belongs is not installed",
    "GOLDEN_FAILURE",
    "No module named",
    "plugin load failed",
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def text_since(path, since):
    kept = []
    active = False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = TIMESTAMP.match(line)
        if match:
            active = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S") >= since
        if active:
            kept.append(line)
    return "\n".join(kept)


def matching_plogs(root, pid, since):
    paths = []
    for path in Path(root).rglob(f"*{pid}*"):
        if path.is_file() and datetime.fromtimestamp(path.stat().st_mtime) >= since:
            paths.append(path)
    return sorted(paths)


def excerpt(text, needles):
    lines = []
    for line in text.splitlines():
        if any(needle in line for needle in needles):
            lines.append(line.strip())
    return lines[-8:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--plog-root", required=True)
    parser.add_argument("--since", required=True, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    since = datetime.strptime(args.since, "%Y-%m-%d %H:%M:%S")
    expected_rows = read_csv(args.input)
    result_rows = read_csv(args.result)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest_cases = manifest["layers"]["L2"]
    expected_names = [row["testcase_name"] for row in expected_rows]
    result_by_name = {row["testcase_name"]: row for row in result_rows}
    manifest_by_name = {case["testcase_name"]: case for case in manifest_cases}
    duplicate_input = sorted(name for name, count in Counter(expected_names).items() if count != 1)
    duplicate_result = sorted(
        name for name, count in Counter(row["testcase_name"] for row in result_rows).items() if count != 1
    )

    cases = []
    for name in expected_names:
        result = result_by_name.get(name)
        meta = manifest_by_name.get(name)
        errors = []
        if result is None:
            errors.append("missing result row")
        if meta is None:
            errors.append("missing manifest row")

        log_path = Path(args.log_dir) / f"ttk-error-{name}.log"
        current_log = ""
        if not log_path.is_file():
            errors.append("missing per-case TTK error log")
        else:
            current_log = text_since(log_path, since)
            if not current_log:
                errors.append("no error-log evidence from this run")
        infra_hits = [needle for needle in INFRA_FAILURES if needle in current_log]
        if infra_hits:
            errors.append(f"infrastructure failure markers: {infra_hits}")

        if result is not None:
            if result.get("precision") != "NO_OUTPUT" or result.get("precision_status") != "FAIL":
                errors.append(
                    f"TTK did not observe rejection: precision={result.get('precision')}, "
                    f"status={result.get('precision_status')}"
                )
            if str(result.get("is_enabled", "")).strip().lower() in {"false", "0"}:
                errors.append("case was skipped/disabled")

        evidence_class = "kernel_assert" if name.endswith(("indices_negative", "indices_upper_bound")) else "host_contract"
        source_evidence = []
        plog_paths = []
        if evidence_class == "host_contract":
            required = (
                "aclnn_moe_token_unpermute_with_ep.cpp:",
                "errno[161002]",
                "errno[561000]",
                "aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize",
            )
            missing_markers = [marker for marker in required if marker not in current_log]
            if missing_markers:
                errors.append(f"missing operator host-rejection markers: {missing_markers}")
            source_evidence = excerpt(current_log, required)
        else:
            if "VEC_ERROR encountered" not in current_log:
                errors.append("missing VEC_ERROR from device-side assertion")
            pids = PID.findall(current_log)
            if not pids:
                errors.append("missing failing worker PID")
            plog_text = ""
            for pid in sorted(set(pids)):
                for path in matching_plogs(args.plog_root, pid, since):
                    plog_paths.append(str(path))
                    plog_text += "\n" + path.read_text(encoding="utf-8", errors="replace")
            expected_value = "-1" if name.endswith("indices_negative") else "2134372523"
            required = (
                "The trap instruction reports an error",
                "fault kernel_name=MoeTokenUnpermuteWithEp_",
                f"sortedIndices value {expected_value} is outside [0, 2134372523)",
            )
            missing_markers = [marker for marker in required if marker not in plog_text]
            if missing_markers:
                errors.append(f"missing operator kernel-rejection markers: {missing_markers}")
            source_evidence = excerpt(plog_text, required)

        cases.append(
            {
                "testcase_name": name,
                "mutation": meta.get("mutation") if meta else None,
                "expected_error": meta.get("expected_error") if meta else None,
                "evidence_class": evidence_class,
                "ttk_result": {
                    "precision": result.get("precision") if result else None,
                    "precision_status": result.get("precision_status") if result else None,
                    "is_enabled": result.get("is_enabled") if result else None,
                },
                "ttk_error_log": str(log_path),
                "plog_paths": plog_paths,
                "source_evidence": source_evidence,
                "correctly_rejected": not errors,
                "errors": errors,
            }
        )

    unexpected = sorted(set(result_by_name) - set(expected_names))
    missing = sorted(set(expected_names) - set(result_by_name))
    missing_manifest = sorted(set(expected_names) - set(manifest_by_name))
    failed_cases = [case["testcase_name"] for case in cases if not case["correctly_rejected"]]
    report = {
        "input": str(Path(args.input)),
        "result": str(Path(args.result)),
        "run_since": args.since,
        "generated_count": len(expected_rows),
        "manifest_count": len(manifest_cases),
        "executed_count": len(result_rows),
        "correctly_rejected_count": len(cases) - len(failed_cases),
        "not_rejected_or_unproven_count": len(failed_cases),
        "skipped_count": sum(
            str(row.get("is_enabled", "")).strip().lower() in {"false", "0"} for row in result_rows
        ),
        "duplicate_input": duplicate_input,
        "duplicate_result": duplicate_result,
        "missing": missing,
        "unexpected": unexpected,
        "missing_manifest": missing_manifest,
        "failed_cases": failed_cases,
        "cases": cases,
    }
    report["passed"] = not any(
        (
            duplicate_input,
            duplicate_result,
            missing,
            unexpected,
            missing_manifest,
            failed_cases,
            len(expected_rows) != len(manifest_cases),
            len(expected_rows) != len(result_rows),
        )
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
