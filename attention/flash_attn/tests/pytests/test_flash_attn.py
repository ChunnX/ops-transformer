#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import sys
import argparse
import torch
from pathlib import Path

# ── 新架构 modules ──
from core.data import flash_attn_inputs
from core.case_loader import load_case_modules, normalize_params, resolve_case_ids
from core.orchestrator import run_case
from core.reporter import Reporter

_DEFAULT_CASE_FILES = ["base", "fia_stc", "functional_stc",
                        "functional_redline_train_tnd", "functional_redline_train",
                        "functional_redline_infer", "performance_redline_infer"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttn 精度测试（新架构）")
    parser.add_argument("--case_files", type=str, default=",".join(_DEFAULT_CASE_FILES))
    parser.add_argument("--case_id",    type=str,  default="all")
    parser.add_argument("--device_id",  type=int,  default=0)
    parser.add_argument("--use_gpu",    action="store_true")
    parser.add_argument("--gpu_device", type=int,  default=0)
    parser.add_argument("--dump_tensors", action="store_true")
    parser.add_argument("--dump_dir",   type=str,  default="./dump_output")
    parser.add_argument("--verbose_diff", action="store_true")
    parser.add_argument("--visualize",  action="store_true")
    parser.add_argument("--viz_dir",    type=str,  default="./viz_output")
    parser.add_argument("--meta_only",  action="store_true")
    parser.add_argument("--load_gpu_dump", type=str, default=None)
    parser.add_argument("--load_npu_dump", type=str, default=None)
    parser.add_argument("--compare_mode", action="store_true")
    parser.add_argument("--graph_mode", action="store_true")
    parser.add_argument("--fail_analysis", action="store_true")
    parser.add_argument("--report_interval", type=int, default=20)
    parser.add_argument("--perf_mode",   action="store_true")
    parser.add_argument("--perf_runs",   type=int,  default=5)
    parser.add_argument("--perf_cold_thr", type=int, default=16)
    parser.add_argument("--perf_output", type=str,  default="./perf_output")
    parser.add_argument("--skip_accuracy", action="store_true")
    parser.add_argument("--result_csv", type=str, default=None, help="精度结果 CSV 输出路径")
    args = parser.parse_args()

    # ── 加载用例 ──
    module_names = ["test_cases." + x.strip() for x in args.case_files.split(",")]
    all_cases = load_case_modules(module_names)
    run_cases = resolve_case_ids(args.case_id, all_cases)
    if not run_cases:
        print("[ERROR] 没有可运行的 case。"); sys.exit(1)

    # ── 性能测试分支 ├─
    if args.perf_mode:
        from pathlib import Path
        from utils.perf_runner import run_all_perf
        from utils.perf_parser import parse_op_summary, compute_stats, _case_key

        cases_list = [normalize_params(all_cases[name]) for name in run_cases]
        out = Path(args.perf_output)

        csv_path = run_all_perf(cases_list, args.perf_runs, args.perf_cold_thr,
                                out, case_names=run_cases)

        if csv_path is None or not csv_path.exists():
            print("[ERROR] msprof 采集失败")
            sys.exit(1)

        entries = parse_op_summary(str(csv_path))
        stats, crashed = compute_stats(entries, cases_list, args.perf_runs,
                                         args.perf_cold_thr, log_dir=str(out))

        SEP = "─" * 110
        print(f"\n┌{SEP}┐")
        print(f"│  性能测试  ({len(stats)} cases, {len(crashed)} crashed)")
        print(f"├{SEP}┤")
        print(f"│  {'Case':<55}  {'Avg(us)':>9}  {'Min(us)':>9}  {'Max(us)':>9}  {'Mode':>6}  │")
        print(f"├{SEP}┤")
        for name in run_cases:
            s = stats.get(_case_key(normalize_params(all_cases[name])))
            if not s: continue
            short = name[:55] if len(name) <= 55 else name[:52] + "..."
            if s["mode"] == "CRASH":
                print(f"│  {short:<55}  {'-':>9}  {'-':>9}  {'-':>9}  {'CRASH':>6}  │")
            else:
                print(f"│  {short:<55}  {s['avg_us']:>9.2f}  {s['min_us']:>9.2f}  "
                      f"{s['max_us']:>9.2f}  {s['mode']:>6}  │")
        print(f"└{SEP}┘")

        import csv as csv_mod
        perf_csv = out / "perf.csv"
        with open(perf_csv, "w", newline="") as f:
            w = csv_mod.DictWriter(f, ["#","case","avg_us","min_us","max_us","mode","error"])
            w.writeheader()
            for idx, name in enumerate(run_cases, 1):
                s = stats.get(_case_key(normalize_params(all_cases[name])))
                if not s: continue
                is_crash = s["mode"] == "CRASH"
                w.writerow(dict(
                    **{"#": idx},
                    case=name,
                    avg_us="" if is_crash else s["avg_us"],
                    min_us="" if is_crash else s["min_us"],
                    max_us="" if is_crash else s["max_us"],
                    mode=s["mode"],
                    error=s.get("error", "") if is_crash else "",
                ))
        print(f"[perf] {perf_csv}")
        if crashed:
            print(f"\n[perf] {len(crashed)} cases crashed:")
            for key in crashed: print(f"  {key}")
        sys.exit(0)

    # ── 选择后端 ──
    if args.use_gpu:
        try:
            from core.backends.gpu import GPUBackend
            primary = GPUBackend(args.gpu_device)
        except ImportError:
            print("[ERROR] GPU backend 不可用"); sys.exit(1)
    else:
        try:
            from core.backends.npu import NPUBackend
            primary = NPUBackend(args.device_id)
        except ImportError:
            print("[ERROR] NPU backend 不可用"); sys.exit(1)

    from core.backends.cpu import CPUBackend
    golden = CPUBackend()
    comparators = []
    if args.compare_mode and args.use_gpu:
        try:
            from core.backends.npu import NPUBackend
            comparators = [NPUBackend(args.device_id)]
        except ImportError:
            pass
    elif args.compare_mode and not args.use_gpu:
        try:
            from core.backends.gpu import GPUBackend
            comparators = [GPUBackend(args.gpu_device)]
        except ImportError:
            pass

    # ── 逐 case 执行 ──
    reporter = Reporter()
    for i, name in enumerate(run_cases):
        params = normalize_params(all_cases[name])
        layout = params.get("input_layout", "BNSD")
        dtype = params.get("Dtype", "fp16")

        print(f"\n{'='*66}")
        print(f"  Case: {name}  "
              f"B={params.get('B')} N1={params['N1']} N2={params.get('N2')} "
              f"S1={params.get('S1')} S2={params.get('S2')} D={params['D']} "
              f"layout={layout} dtype={dtype}")
        print(f"{'='*66}")

        primary_available = primary.is_available()

        n1 = params.get("N1", 1)
        n2 = params.get("N2", n1)
        if not (n1 >= n2 and n1 % n2 == 0):
            print(f"  跳过: N1={n1}, N2={n2} 不满足 N1>=N2 且 N1%N2==0")
            reporter.record(name, {"attn": {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                                            "fail_cnt": 0, "total": 0, "fail_ratio": 0.0},
                                   "lse":  {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                                            "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}},
                            error="skipped: N1<N2 or N1%N2!=0")
            if args.report_interval > 0 and (i + 1) % args.report_interval == 0 \
                    and (i + 1) < len(run_cases):
                reporter.print_interim(len(run_cases))
            continue
        if not primary_available:
            print("[ERROR] primary backend 不可用")
            err = "primary backend not available"
            reporter.record(name, {"attn": {"passed": False, "max_abs": float('nan'),
                                            "mean_abs": float('nan'), "fail_cnt": -1,
                                            "total": -1, "fail_ratio": float('nan')},
                                   "lse":  {"passed": False, "max_abs": float('nan'),
                                            "mean_abs": float('nan'), "fail_cnt": -1,
                                            "total": -1, "fail_ratio": float('nan')}},
                            error=err)
            if args.report_interval > 0 and (i + 1) % args.report_interval == 0 \
                    and (i + 1) < len(run_cases):
                reporter.print_interim(len(run_cases))
            continue

        try:
            result = run_case(
                params=params,
                primary=primary,
                golden=golden,
                spec=flash_attn_inputs,
                comparators=comparators if comparators else None,
            )
            err = None
        except Exception as e:
            import traceback
            err = f"{e}\n{traceback.format_exc()}"
            print(f"[ERROR] {name} 运行异常: {e}")
            traceback.print_exc()
            result = {"attn": {"passed": False, "max_abs": float('nan'),
                               "mean_abs": float('nan'), "fail_cnt": -1,
                               "total": -1, "fail_ratio": float('nan')},
                      "lse":  {"passed": False, "max_abs": float('nan'),
                               "mean_abs": float('nan'), "fail_cnt": -1,
                               "total": -1, "fail_ratio": float('nan')}}

        reporter.record(name, result, error=err)
        if args.report_interval > 0 and (i + 1) % args.report_interval == 0 \
                and (i + 1) < len(run_cases):
            reporter.print_interim(len(run_cases))

    fail_cnt = reporter.print_final()
    if args.result_csv:
        reporter.save_csv(Path(args.result_csv))
        print(f"[result] {args.result_csv}")
    sys.exit(0 if fail_cnt == 0 else 1)
