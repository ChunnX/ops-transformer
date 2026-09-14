#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quant_flash_mla_with_kvcache(全量化 MLA FA)NPU 性能测试入口。

逐 case 独立 msprof 进程(one-by-one), 规避批跑下 profiler 条目归属歧义。
链路: perf_cases(用例) -> perf_codegen(生成脚本) -> perf_profiler(msprof 采集)
      -> perf_parser(过滤 OP Type=QuantFlashMlaWithKvcache, 热启动均值) -> perf.csv

用法:
    conda activate fzj
    source /home/fanzijian/Ascend/ascend-toolkit/set_env.sh        # 基础 CANN 包
    source /home/fanzijian/custom/vendors/custom_transformer/bin/set_env.bash  # custom 算子包
    python3 run_perf.py --runs 5
    python3 run_perf.py --runs 5 --case-filter Decode,PA_NZ
    python3 run_perf.py --list
"""

import argparse
import csv
import time
from pathlib import Path

from perf_cases import load_cases
from perf_codegen import build_script
from perf_profiler import run_msprof
from perf_parser import parse_op_summary, compute_hot_avg

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="QuantFlashMlaWithKvcache perf benchmark")
    ap.add_argument("--runs", type=int, default=5,
                    help="每 case 迭代次数(第一条作预热丢弃)")
    ap.add_argument("--case-filter", type=str, default=None,
                    help="逗号分隔的用例名子串过滤")
    ap.add_argument("--output", type=str, default=None,
                    help="输出目录(默认 results/<时间戳>)")
    ap.add_argument("--list", action="store_true", help="只列出用例名")
    args = ap.parse_args()

    if args.runs < 2:
        ap.error("--runs 需 >= 2 (第一条为预热, 丢弃)")

    cases = load_cases()
    if args.case_filter:
        filters = [f.strip() for f in args.case_filter.split(",") if f.strip()]
        cases = {k: v for k, v in cases.items() if any(f in k for f in filters)}
        print(f"[info] Filtered to {len(cases)} cases")
    if not cases:
        print("[error] no cases matched")
        return 2

    if args.list:
        for name in cases:
            print(name)
        return 0

    print(f"[info] Loaded {len(cases)} cases, runs={args.runs}")
    output_dir = (Path(args.output).resolve() if args.output else
                  HERE / "results" / time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Output -> {output_dir}")

    results = {}   # name -> (avg_us, samples, status, message)
    total = len(cases)
    for idx, (name, case) in enumerate(cases.items()):
        print(f"\n{'=' * 70}")
        print(f"[{idx + 1}/{total}] [HOT] {name}")
        print(f"{'=' * 70}")
        case_dir = output_dir / name
        try:
            script = build_script(name, case, args.runs)
            run_msprof(script, case_dir, label=name)
            entries = parse_op_summary(case_dir / "op_summary.csv")
            durations = [d for _, d in entries]
            if len(entries) < args.runs:
                message = (f"captured {len(entries)}/{args.runs} entries, "
                           f"check {case_dir / name}.msprof.log")
                print(f"  => FAILED: {message}")
                results[name] = (-1.0, len(entries), "MISSING", message)
                continue
            avg = compute_hot_avg(durations)
            results[name] = (avg, args.runs, "PASS", "")
            print(f"  => {avg:.2f} us  ({args.runs} entries, discard 1st)")
        except Exception as e:
            print(f"  => FAILED: {e}")
            results[name] = (-1.0, 0, "ERROR", str(e))

    perf_csv = output_dir / "perf.csv"
    with open(perf_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_name", "duration_us", "samples", "status", "message"])
        for name, (avg, samples, status, message) in results.items():
            writer.writerow([name, f"{avg:.2f}" if avg > 0 else "",
                             samples, status, message])

    n_failed = sum(1 for r in results.values() if r[2] != "PASS")
    print(f"\n{'=' * 70}")
    print(f"[SUMMARY] {len(results)} cases, {n_failed} failed, results -> {perf_csv}")
    print(f"{'=' * 70}")
    for name, (avg, samples, status, _msg) in results.items():
        shown = f"{avg:.2f} us" if avg > 0 else f"FAILED({status})"
        print(f"  [HOT] {name:<64s} {shown}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
