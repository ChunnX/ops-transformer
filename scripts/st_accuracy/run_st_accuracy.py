#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.st_accuracy.coverage import build_coverage_report, write_coverage_report
from scripts.st_accuracy.inventory import discover_operators
from scripts.st_accuracy.prepare import prepare_pkg
from scripts.st_accuracy.report import ResultReporter
from scripts.st_accuracy.runner import CaseSelector, parse_csv, parse_int_csv, run_cases
from scripts.st_accuracy.st_cases import load_cases
from scripts.st_accuracy.taxonomy import build_taxonomy_report, write_taxonomy_report


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_out_dir(repo_root: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return repo_root / "build" / "st_accuracy" / timestamp


def _filter_operators(operators, ops: set[str] | None):
    if not ops:
        return operators
    return [operator for operator in operators if operator.name in ops]


def _cmd_list(args: argparse.Namespace) -> int:
    root = _repo_root()
    operators = _filter_operators(discover_operators(root), parse_csv(args.ops))
    cases = load_cases(root, operators, json_pattern=args.json_pattern)
    case_counts: dict[str, int] = {}
    api_counts: dict[str, set[str]] = {}
    for case in cases:
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
    if args.format == "json":
        print(json.dumps({"operators": rows, "operator_count": len(rows), "case_count": len(cases)}, indent=2, sort_keys=True))
    else:
        print(f"operators={len(rows)} cases={len(cases)}")
        for row in rows:
            print(f"{row['category']}/{row['op']}: apis={len(row['apis'])} cases={row['cases']}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    root = _repo_root()
    ops = parse_csv(args.ops)
    if args.prepare == "pkg":
        prepare_pkg(root, sorted(ops or []), args.soc, args.vendor, jobs=args.jobs, opt_level=args.opt)

    operators = discover_operators(root)
    cases = load_cases(root, operators, json_pattern=args.json_pattern)
    selector = CaseSelector(
        ops=ops,
        apis=parse_csv(args.apis),
        dtypes=parse_csv(args.dtype),
        case_ids=parse_int_csv(args.case_ids),
        limit=args.limit,
    )
    selected = selector.apply(cases)
    out_dir = Path(args.out) if args.out else _default_out_dir(root)
    results = run_cases(
        selected,
        repo_root=root,
        out_dir=out_dir,
        vendor=args.vendor,
        soc=args.soc,
        device=args.device,
        execute=not args.no_execute,
    )
    ResultReporter(out_dir).write(results)
    failed = sum(1 for result in results if result.status in {"failed", "error"})
    print(f"wrote {len(results)} result rows to {out_dir}")
    return 1 if failed else 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    root = _repo_root()
    operators = _filter_operators(discover_operators(root), parse_csv(args.ops))
    cases = load_cases(root, operators, json_pattern=args.json_pattern)
    report = build_coverage_report(operators, cases)
    out_dir = Path(args.out) if args.out else _default_out_dir(root)
    write_coverage_report(report, out_dir)
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
    operators = _filter_operators(discover_operators(root), parse_csv(args.ops))
    cases = load_cases(root, operators, json_pattern=args.json_pattern)
    report = build_taxonomy_report(root, operators, cases)
    out_dir = Path(args.out) if args.out else _default_out_dir(root)
    write_taxonomy_report(report, out_dir)
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
    run_parser.add_argument("--prepare", choices=("none", "pkg"), default="none")
    run_parser.add_argument("--jobs", type=int, default=16)
    run_parser.add_argument("--opt", default="3")
    run_parser.add_argument("--json-pattern", default="all_*.json")
    run_parser.add_argument("--no-execute", action="store_true", help="Write selected rows without invoking the native runner")
    run_parser.set_defaults(func=_cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
