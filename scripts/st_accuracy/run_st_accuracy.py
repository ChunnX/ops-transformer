#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

from scripts.st_accuracy.coverage import build_coverage_report, write_coverage_report
from scripts.st_accuracy.inventory import discover_operators
from scripts.st_accuracy.prepare import prepare_pkg
from scripts.st_accuracy.report import ResultReporter
from scripts.st_accuracy.runner import CaseSelector, parse_csv, parse_int_csv, run_cases
from scripts.st_accuracy.st_cases import CaseLoadResult, load_cases_with_diagnostics
from scripts.st_accuracy.taxonomy import build_taxonomy_report, write_taxonomy_report
from scripts.st_accuracy.timing import TimingCollector


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_out_dir(repo_root: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return repo_root / "build" / "st_accuracy" / timestamp


def _filter_operators(operators, ops: set[str] | None):
    if not ops:
        return operators
    return [operator for operator in operators if operator.name in ops]


def _build_discovery_report(
    operators,
    load_result: CaseLoadResult,
    requested_ops: set[str] | None,
) -> dict:
    case_counts: dict[str, int] = {}
    api_counts: dict[str, set[str]] = {}
    for case in load_result.cases:
        case_counts[case.op_name] = case_counts.get(case.op_name, 0) + 1
        api_counts.setdefault(case.op_name, set()).add(case.aclnn_name)

    rows = [
        {
            "op": operator.name,
            "category": operator.category,
            "path": operator.path.as_posix(),
            "apis": sorted(api_counts.get(operator.name, set())),
            "cases": case_counts.get(operator.name, 0),
        }
        for operator in operators
    ]
    found_ops = {operator.name for operator in operators}
    missing_ops = sorted((requested_ops or set()) - found_ops)
    zero_case_ops = sorted(
        operator.name
        for operator in operators
        if requested_ops and operator.name in requested_ops and case_counts.get(operator.name, 0) == 0
    )
    return {
        "operators": rows,
        "operator_count": len(rows),
        "case_count": len(load_result.cases),
        "missing_ops": missing_ops,
        "zero_case_ops": zero_case_ops,
        "skipped_assets": [asset.to_json_dict() for asset in load_result.skipped_assets],
    }


def _measure(timings: TimingCollector | None, key: str):
    return timings.measure(key) if timings is not None else nullcontext()


def _discover_and_load(root: Path, ops: set[str] | None, json_pattern: str, timings: TimingCollector | None = None):
    with _measure(timings, "inventory_discovery_ms"):
        operators = _filter_operators(discover_operators(root), ops)
    with _measure(timings, "case_load_ms"):
        load_result = load_cases_with_diagnostics(root, operators, json_pattern=json_pattern)
    with _measure(timings, "discovery_report_ms"):
        report = _build_discovery_report(operators, load_result, requested_ops=ops)
    return operators, load_result, report


def _discovery_gate_failed(report: dict, requested_ops: set[str] | None) -> bool:
    if not requested_ops:
        return False
    return bool(report["missing_ops"] or report["zero_case_ops"] or report["skipped_assets"])


def _print_discovery_failure(report: dict) -> None:
    print(json.dumps({"discovery_error": report}, indent=2, sort_keys=True))


def _cmd_list(args: argparse.Namespace) -> int:
    root = _repo_root()
    timings = TimingCollector()
    ops = parse_csv(args.ops)
    _operators, _load_result, report = _discover_and_load(root, ops, args.json_pattern, timings=timings)
    report["timings"] = timings.to_json_dict()
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"operators={report['operator_count']} cases={report['case_count']}")
        for row in report["operators"]:
            print(f"{row['category']}/{row['op']}: apis={len(row['apis'])} cases={row['cases']}")
        if report["missing_ops"]:
            print(f"missing_ops={','.join(report['missing_ops'])}")
        if report["zero_case_ops"]:
            print(f"zero_case_ops={','.join(report['zero_case_ops'])}")
        if report["skipped_assets"]:
            print(f"skipped_assets={len(report['skipped_assets'])}")
    return 1 if _discovery_gate_failed(report, ops) else 0


def _cmd_run(args: argparse.Namespace) -> int:
    root = _repo_root()
    timings = TimingCollector()
    ops = parse_csv(args.ops)
    _operators, load_result, discovery_report = _discover_and_load(root, ops, args.json_pattern, timings=timings)
    if _discovery_gate_failed(discovery_report, ops):
        discovery_report["timings"] = timings.to_json_dict()
        _print_discovery_failure(discovery_report)
        return 1

    with _measure(timings, "case_select_ms"):
        selector = CaseSelector(
            ops=ops,
            apis=parse_csv(args.apis),
            dtypes=parse_csv(args.dtype),
            case_ids=parse_int_csv(args.case_ids),
            limit=args.limit,
        )
        selected = selector.apply(load_result.cases)
    if not args.no_execute and selected:
        package_ops = sorted({case.op_name for case in selected})
        with _measure(timings, "prepare_pkg_ms"):
            prepare_pkg(root, package_ops, args.soc, args.vendor, jobs=args.jobs, opt_level=args.opt)
    out_dir = Path(args.out) if args.out else _default_out_dir(root)
    run_metadata = {}
    results = run_cases(
        selected,
        repo_root=root,
        out_dir=out_dir,
        vendor=args.vendor,
        soc=args.soc,
        device=args.device,
        execute=not args.no_execute,
        timings=timings,
        run_metadata=run_metadata,
    )
    ResultReporter(out_dir).write(results, timings=timings, metadata=run_metadata)
    failed = sum(1 for result in results if result.status in {"failed", "error"})
    print(f"wrote {len(results)} result rows to {out_dir}")
    return 1 if failed else 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    root = _repo_root()
    timings = TimingCollector()
    ops = parse_csv(args.ops)
    operators, load_result, discovery_report = _discover_and_load(root, ops, args.json_pattern, timings=timings)
    if _discovery_gate_failed(discovery_report, ops):
        discovery_report["timings"] = timings.to_json_dict()
        _print_discovery_failure(discovery_report)
        return 1
    with _measure(timings, "support_gate_ms"):
        report = build_coverage_report(operators, load_result.cases)
    out_dir = Path(args.out) if args.out else _default_out_dir(root)
    write_coverage_report(report, out_dir, timings=timings)
    summary = report["summary"]
    print(
        "coverage "
        f"operators={summary['operator_count']} "
        f"cases={summary['case_count']} "
        f"executable={summary['executable_case_count']} "
        f"unsupported={summary['unsupported_case_count']} "
        f"out={out_dir}"
    )
    return 0


def _cmd_taxonomy(args: argparse.Namespace) -> int:
    root = _repo_root()
    timings = TimingCollector()
    ops = parse_csv(args.ops)
    operators, load_result, discovery_report = _discover_and_load(root, ops, args.json_pattern, timings=timings)
    if _discovery_gate_failed(discovery_report, ops):
        discovery_report["timings"] = timings.to_json_dict()
        _print_discovery_failure(discovery_report)
        return 1
    with _measure(timings, "support_gate_ms"):
        report = build_taxonomy_report(root, operators, load_result.cases)
    out_dir = Path(args.out) if args.out else _default_out_dir(root)
    write_taxonomy_report(report, out_dir, timings=timings)
    summary = report["summary"]
    print(
        "taxonomy "
        f"operators={summary['operator_count']} "
        f"cases={summary['case_count']} "
        f"unsupported={summary['unsupported_case_count']} "
        f"out={out_dir}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local ST accuracy checks for ops-transformer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List optimizable operators and ST case counts")
    list_parser.add_argument("--ops", help="Comma-separated operator names")
    list_parser.add_argument("--soc", default="ascend910b")
    list_parser.add_argument("--json-pattern", default="all_*.json")
    list_parser.add_argument("--format", choices=("text", "json"), default="text")
    list_parser.set_defaults(func=_cmd_list)

    coverage_parser = subparsers.add_parser("coverage", help="Write native runner coverage gap reports")
    coverage_parser.add_argument("--ops", help="Comma-separated operator names")
    coverage_parser.add_argument("--json-pattern", default="all_*.json")
    coverage_parser.add_argument("--out")
    coverage_parser.set_defaults(func=_cmd_coverage)

    taxonomy_parser = subparsers.add_parser("taxonomy", help="Write unsupported taxonomy and adapter priority reports")
    taxonomy_parser.add_argument("--ops", help="Comma-separated operator names")
    taxonomy_parser.add_argument("--json-pattern", default="all_*.json")
    taxonomy_parser.add_argument("--out")
    taxonomy_parser.set_defaults(func=_cmd_taxonomy)

    run_parser = subparsers.add_parser("run", help="Run selected ST accuracy cases")
    run_parser.add_argument("--ops", help="Comma-separated operator names")
    run_parser.add_argument("--apis", help="Comma-separated aclnn names, such as Abs,Add,Adds")
    run_parser.add_argument("--case-ids", help="Comma-separated ST case ids")
    run_parser.add_argument("--dtype", help="Comma-separated dtype filters")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--out")
    run_parser.add_argument("--soc", default="ascend910b")
    run_parser.add_argument("--vendor", default="custom")
    run_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ACL runtime logical device id for aclrtSetDevice(), not the npu-smi physical id",
    )
    run_parser.add_argument(
        "--prepare",
        choices=("none", "pkg"),
        default="pkg",
        help="Deprecated compatibility flag; real device run always builds and installs the custom package.",
    )
    run_parser.add_argument("--jobs", type=int, default=16)
    run_parser.add_argument("--opt", default="3")
    run_parser.add_argument("--json-pattern", default="all_*.json")
    run_parser.add_argument("--no-execute", action="store_true", help="Write selected rows without invoking the native runner")
    run_parser.set_defaults(func=_cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
