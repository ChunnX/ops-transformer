#!/usr/bin/env python3
"""Batch runner for attention2ffn_server.py without modifying the source test.

Examples:
    python3 batch_attention2ffn_server.py --list
    python3 batch_attention2ffn_server.py --dry-run
    python3 batch_attention2ffn_server.py
    python3 batch_attention2ffn_server.py --cases bf16_smoke_e8_k1,bf16_boundary_e1024_k16
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_SCRIPT = SCRIPT_DIR / "attention2ffn_server.py"

# Keep the communication topology unchanged and vary the operator inputs.  With
# two ranks, rank 0 is the FFN worker and rank 1 is the Attention worker.
COMMON_OVERRIDES: Dict[str, Any] = {
    "server_num": 1,
    "rank_per_dev": 2,
    "moe_rank_ffn_num": 1,
    "shared_expert_num": 0,
    "rank_num_per_shared_expert": 0,
    "is_performance": 0,
    "is_check_setting": True,
    "is_topk_balance": True,
}

# moe_expert_num is derived by the original script as
# moe_rank_ffn_num * moe_expert_per_rank, so only the latter is overridden.
TEST_CASES: List[Tuple[str, Dict[str, Any]]] = [
    (
        "original_default",
        {
            "bs": 512,
            "h": 8192,
            "k": 1,
            "moe_expert_per_rank": 1,
            "out_dtype": 0,
            "quant_mode": 0,
            "is_mask": 0,
            "sync_flag": 0,
        },
    ),
    (
        "bf16_smoke_e8_k1",
        {
            "bs": 8,
            "h": 1024,
            "k": 1,
            "moe_expert_per_rank": 8,
            "out_dtype": 0,
            "quant_mode": 0,
            "is_mask": 0,
            "sync_flag": 0,
        },
    ),
    (
        "fp16_e64_k4",
        {
            "bs": 32,
            "h": 4096,
            "k": 4,
            "moe_expert_per_rank": 64,
            "out_dtype": 1,
            "quant_mode": 0,
            "is_mask": 0,
            "sync_flag": 0,
        },
    ),
    (
        "bf16_e256_k8",
        {
            "bs": 16,
            "h": 7168,
            "k": 8,
            "moe_expert_per_rank": 256,
            "out_dtype": 0,
            "quant_mode": 0,
            "is_mask": 0,
            "sync_flag": 0,
        },
    ),
    (
        "bf16_boundary_e1024_k16",
        {
            "bs": 8,
            "h": 8192,
            "k": 16,
            "moe_expert_per_rank": 1024,
            "out_dtype": 0,
            "quant_mode": 0,
            "is_mask": 0,
            "sync_flag": 0,
        },
    ),
    (
        "dynamic_quant_e64_k4",
        {
            "bs": 16,
            "h": 4096,
            "k": 4,
            "moe_expert_per_rank": 64,
            "out_dtype": 0,
            "quant_mode": 2,
            "is_attn2ffn_scales": 0,
            "is_mask": 0,
            "sync_flag": 0,
        },
    ),
    (
        "mask_sync_e32_k4",
        {
            "bs": 32,
            "h": 2048,
            "k": 4,
            "moe_expert_per_rank": 32,
            "out_dtype": 0,
            "quant_mode": 0,
            "is_mask": 1,
            "sync_flag": 1,
        },
    ),
]


def _find_module_assignments(source: str) -> Dict[str, ast.Assign]:
    tree = ast.parse(source, filename=str(TEST_SCRIPT))
    assignments: Dict[str, ast.Assign] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            assignments[target.id] = node
    return assignments


def _make_case_source(source: str, case_name: str, overrides: Dict[str, Any]) -> str:
    """Replace top-level assignments in a temporary copy of the source test."""
    values = dict(COMMON_OVERRIDES)
    values.update(overrides)
    values["current_case_name"] = case_name

    assignments = _find_module_assignments(source)
    missing = sorted(set(values) - set(assignments))
    if missing:
        raise ValueError(f"Top-level settings not found in {TEST_SCRIPT.name}: {missing}")

    lines = source.splitlines(keepends=True)
    replacements = []
    for name, value in values.items():
        node = assignments[name]
        end_lineno = getattr(node, "end_lineno", node.lineno)
        replacements.append((node.lineno - 1, end_lineno, f"{name} = {value!r}\n"))

    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]

    case_source = "".join(lines)
    ast.parse(case_source, filename=f"{case_name}.py")
    return case_source


def _selected_cases(names: Iterable[str]) -> List[Tuple[str, Dict[str, Any]]]:
    case_map = dict(TEST_CASES)
    names = list(names)
    if not names:
        return TEST_CASES

    unknown = sorted(set(names) - set(case_map))
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")
    return [(name, case_map[name]) for name in names]


def _print_case(name: str, overrides: Dict[str, Any]) -> None:
    settings = dict(COMMON_OVERRIDES)
    settings.update(overrides)
    expert_num = settings["moe_rank_ffn_num"] * settings["moe_expert_per_rank"]
    print(
        f"{name}: bs={settings['bs']}, h={settings['h']}, k={settings['k']}, "
        f"moe_expert_num={expert_num}, quant_mode={settings['quant_mode']}, "
        f"mask={settings['is_mask']}, sync={settings['sync_flag']}"
    )


def _last_result_marker(log_text: str) -> Optional[str]:
    markers = re.findall(r"attn2ffn test (Success|Failed)\.", log_text)
    return markers[-1] if markers else None


def _tail(text: str, line_count: int = 40) -> str:
    return "\n".join(text.splitlines()[-line_count:])


def _write_batch_report(
    report_path: Path,
    started_at: str,
    cases: List[Tuple[str, Dict[str, Any]]],
    results: List[Dict[str, Any]],
) -> None:
    """Combine case settings, complete logs and results into one text report."""
    case_map = dict(cases)
    report_lines = [
        "AttentionToFfnV2 batch test report",
        f"Started at: {started_at}",
        f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source test: {TEST_SCRIPT}",
        "",
    ]

    for result in results:
        name = result["case"]
        settings = dict(COMMON_OVERRIDES)
        settings.update(case_map[name])
        settings["current_case_name"] = name
        report_lines.extend(
            [
                "=" * 80,
                f"CASE: {name}",
                f"STATUS: {result['status']}",
                f"DURATION_SECONDS: {result['seconds']}",
                "SETTINGS:",
                json.dumps(settings, indent=2, ensure_ascii=False),
                "OUTPUT:",
            ]
        )

        log_path_value = result.get("log")
        if log_path_value and Path(log_path_value).is_file():
            report_lines.append(Path(log_path_value).read_text(encoding="utf-8", errors="replace").rstrip())
        elif result.get("error"):
            report_lines.append(str(result["error"]))
        else:
            report_lines.append("No output was captured.")
        report_lines.append("")

    passed = sum(result["status"] == "PASS" for result in results)
    failed = len(results) - passed
    report_lines.extend(
        [
            "=" * 80,
            "SUMMARY",
            f"TOTAL: {len(results)}",
            f"PASS: {passed}",
            f"FAIL: {failed}",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _stop_process(process: subprocess.Popen) -> None:
    """Stop the case process and its multiprocessing children after a timeout."""
    if process.poll() is not None:
        return
    if os.name != "nt":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def _run_case(
    python: str,
    source: str,
    name: str,
    overrides: Dict[str, Any],
    log_dir: Path,
    timeout: int,
    dry_run: bool,
) -> Dict[str, Any]:
    case_source = _make_case_source(source, name, overrides)
    if dry_run:
        print(f"[DRY-RUN] {name}: generated source passed syntax validation")
        return {"case": name, "status": "DRY-RUN", "seconds": 0.0, "log": None}

    log_path = log_dir / f"{name}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(SCRIPT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    start = time.monotonic()
    return_code = -1
    timed_out = False
    with tempfile.TemporaryDirectory(prefix=f"attention2ffn_{name}_") as temp_dir:
        temp_script = Path(temp_dir) / f"{name}.py"
        temp_script.write_text(case_source, encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as log_file:
            process = None
            try:
                process = subprocess.Popen(
                    [python, "-u", str(temp_script)],
                    cwd=SCRIPT_DIR,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name != "nt"),
                )
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop_process(process)

    elapsed = time.monotonic() - start
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    marker = _last_result_marker(log_text)
    passed = not timed_out and return_code == 0 and marker == "Success"
    status = "PASS" if passed else "TIMEOUT" if timed_out else "FAIL"

    print(f"[{status}] {name} ({elapsed:.1f}s), log: {log_path}")
    if not passed:
        print("----- log tail -----")
        print(_tail(log_text))
        print("--------------------")

    return {
        "case": name,
        "status": status,
        "seconds": round(elapsed, 3),
        "return_code": return_code,
        "result_marker": marker,
        "log": str(log_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple attention2ffn_server.py cases without editing the original test script."
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated case names; the default runs every case.",
    )
    parser.add_argument("--list", action="store_true", help="List cases and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Generate and validate cases without running NPU tests.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed case.")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout in seconds for each case (default: 900).")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to run each case.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=SCRIPT_DIR / "batch_logs",
        help="Directory for per-case logs and summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    requested_names = [name.strip() for name in args.cases.split(",") if name.strip()]
    try:
        cases = _selected_cases(requested_names)
    except ValueError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2

    if args.list:
        for name, overrides in cases:
            _print_case(name, overrides)
        return 0

    if args.timeout <= 0:
        print("[ERROR] --timeout must be greater than zero", file=sys.stderr)
        return 2
    if not TEST_SCRIPT.is_file():
        print(f"[ERROR] Test script does not exist: {TEST_SCRIPT}", file=sys.stderr)
        return 2

    source = TEST_SCRIPT.read_text(encoding="utf-8")
    run_dir = args.log_dir / time.strftime("%Y%m%d_%H%M%S")
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source test: {TEST_SCRIPT}")
    print(f"Cases: {len(cases)}")
    results = []
    for index, (name, overrides) in enumerate(cases, start=1):
        print(f"\n========== [{index}/{len(cases)}] {name} ==========")
        _print_case(name, overrides)
        try:
            result = _run_case(
                args.python,
                source,
                name,
                overrides,
                run_dir,
                args.timeout,
                args.dry_run,
            )
        except (OSError, ValueError, SyntaxError) as error:
            print(f"[FAIL] {name}: {error}")
            result = {"case": name, "status": "FAIL", "seconds": 0.0, "error": str(error)}
        results.append(result)
        if args.stop_on_failure and result["status"] not in ("PASS", "DRY-RUN"):
            break

    if not args.dry_run:
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_path = run_dir / "batch_report.txt"
        _write_batch_report(report_path, started_at, cases, results)
        print(f"\nSummary: {summary_path}")
        print(f"Report: {report_path}")

    failed = [result for result in results if result["status"] not in ("PASS", "DRY-RUN")]
    print(f"Completed: {len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
